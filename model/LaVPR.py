import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torchvision.transforms as transforms
import vpr_models
import os
from model.blip_model import BlipForImageTextRetrievalWrapper
from transformers import BlipProcessor, BlipModel
from sklearn.decomposition import PCA
import torch.cuda.amp as amp
from transformers import AutoModel, AutoProcessor
import open_clip

class LaVPR(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                vpr_model_name='mixvpr',
                vpr_model_backbone='ResNet50',
                vpr_encoder_dim=512,            
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False,
                text_model_name='BAAI/bge-large-en-v1.5',
                embeds_dim=512,
                is_freeze_vpr=True,
                is_freeze_text=True,
                fusion_type='none',
                is_encode_image=True,
                is_encode_text=True,
                is_trainable_text_encoder=False,
                is_text_pooling=False,
                is_image_pooling=False,
                is_orig_desc_mining=False,
                cross_modal=0,
                lora_all_linear=False,
                lora_target_modules=None,
                lora_r=64,
                text_encoder_dim=1024,
                 ):
        super().__init__()
        
        self.vpr_model_name = vpr_model_name
        self.vpr_model_backbone = vpr_model_backbone
        self.vpr_encoder_dim = vpr_encoder_dim
        self.text_model_name = text_model_name
        
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.faiss_gpu = faiss_gpu
        self.is_encode_image = is_encode_image
        self.is_encode_text = is_encode_text
        self.is_trainable_text_encoder = is_trainable_text_encoder
        self.is_text_pooling = is_text_pooling
        self.is_image_pooling = is_image_pooling
        self.is_orig_desc_mining = is_orig_desc_mining
        self.cross_modal = cross_modal
        self.lora_all_linear = lora_all_linear
        self.lora_target_modules = lora_target_modules
        self.lora_r = lora_r
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 
       
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        self.fusion_type = fusion_type
        self.embeds_dim = embeds_dim        
        self.text_encoder_dim = text_encoder_dim        
        if 'blip' in vpr_model_name or 'clip' in vpr_model_name:
            text_encoder_dim = vpr_encoder_dim

        if cross_modal == 4: # contrastive loss for cross modal retrieval
            self.contrastive_logit_scale = nn.Parameter(0.07*torch.ones([])) 
            self.contrastive_loss = utils.losses.contrastive_loss_cross_modal
            self.miner = None
            
        self.mse_loss = torch.nn.MSELoss()
        
        if is_encode_image and is_encode_text:
            if is_text_pooling==1: # CLS pooling
                 self.text_pooling = CLSReweightingPooler(text_encoder_dim)
            elif is_text_pooling==2: # Mean pooling
                self.text_pooling = MeanReweightingPooler(text_encoder_dim)
            if is_image_pooling:
                self.image_pooling = CLSReweightingPooler(self.vpr_encoder_dim)
                
            if self.cross_modal == 3:
                self.vpr_proj = nn.Linear(self.vpr_encoder_dim, embeds_dim)
                self.text_proj = nn.Linear(text_encoder_dim, embeds_dim)                                
            elif self.fusion_type == 'mlp':
                input_dim = self.vpr_encoder_dim + text_encoder_dim
                self.fusion = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, embeds_dim))                
            elif self.fusion_type == 'add':
                self.vpr_proj = nn.Linear(self.vpr_encoder_dim, embeds_dim)
                self.text_proj = nn.Linear(text_encoder_dim, embeds_dim)
            elif self.fusion_type == 'dynamic_weighting':
                input_dim = self.vpr_encoder_dim + text_encoder_dim
                self.fusion = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, 2), nn.Softmax(dim=1))                
            elif self.fusion_type == 'fixed_weighting':
                #learn fixed parameter for weighting image and text
                self.w_alpha = Parameter(torch.tensor([0.5]), requires_grad=True)                
                
        # init weight of linear layers but not the pretrained backbones
        self.apply(self._init_weights)
        
        # initialize the vpr encoder and text encoder
        if is_encode_image and ('blip' not in vpr_model_name and 'clip' not in vpr_model_name and 'siglip' not in vpr_model_name and 'eva' not in vpr_model_name):                  
            self.vpr_encoder = vpr_models.get_model(vpr_model_name, vpr_model_backbone, vpr_encoder_dim)                      
            if is_freeze_vpr:
                # Freeze vpr encoder parameters
                for param in self.vpr_encoder.parameters():
                    param.requires_grad = False
            self.vpr_encoder.eval()                    
        
        if is_encode_text:
            if 'blip' in vpr_model_name:
                self.text_encoder = BlipForImageTextRetrievalWrapper.from_pretrained(vpr_model_name)
                self.processor = BlipProcessor.from_pretrained(vpr_model_name)
            elif 'clip' in vpr_model_name or 'siglip' in vpr_model_name:
                self.max_text_length = 77
                if 'siglip' in vpr_model_name:
                    self.max_text_length = 64
                self.text_encoder = AutoModel.from_pretrained(vpr_model_name)
                self.processor = AutoProcessor.from_pretrained(vpr_model_name)
            elif 'eva' in vpr_model_name:
                self.text_encoder, _, self.processor = open_clip.create_model_and_transforms(vpr_model_name.upper(), pretrained='merged2b_s8b_b131k')#'EVA02-B-16'
                self.tokenizer = open_clip.get_tokenizer(vpr_model_name)                
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)  
                self.text_encoder = AutoModel.from_pretrained(text_model_name, attn_implementation="sdpa")        
                            
            if is_freeze_text:
                # Freeze text encoder parameters
                for param in self.text_encoder.parameters():
                    param.requires_grad = False                      
            
            # Define LoRA configuration
            # TaskType.FEATURE_EXTRACTION is appropriate for sentence embedding tasks            
            if is_trainable_text_encoder:                
                lora_targets = lora_target_modules
                if lora_all_linear:
                    lora_targets = "all-linear"                    
                
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_r*2,
                    lora_dropout=0.1,
                    target_modules=lora_targets,
                    task_type=TaskType.SEQ_CLS,
                    use_rslora=True,                    
                    bias="none",
                )
                # Get the PEFT model with LoRA adapters
                self.text_encoder = get_peft_model(self.text_encoder, lora_config)
            elif is_freeze_text:
                self.text_encoder.eval()        

                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # For linear layers, use Kaiming uniform initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            # For biases, it's common to initialize them to zero
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)        
    
    
    # the forward pass of the lightning model
    def forward(self, img, text):
        w = None
        text_embeds = None
        #img = transforms.Resize([320, 320], antialias=True)(img)

        if self.is_encode_image:
            if 'blip' in self.vpr_model_name:
                img_embeds_all = self.text_encoder.encode_image(img)
                img_embeds = img_embeds_all[:,0]
            elif 'clip' in self.vpr_model_name or 'siglip' in self.vpr_model_name:
                img_embeds = self.text_encoder.get_image_features(pixel_values=img)
            elif 'eva' in self.vpr_model_name:            
                img_embeds = self.text_encoder.encode_image(img)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            else:
                with torch.no_grad():                      
                    if 'dinov2' in self.vpr_model_name:
                        vpr_ret = self.vpr_encoder(img, is_training=True)    
                        img_embeds_all = vpr_ret['x_norm_patchtokens']
                        img_embeds = vpr_ret['x_norm_clstoken']                             
                    else:
                        img_embeds = self.vpr_encoder(img)             

            embeds = img_embeds
            embeds_orig = img_embeds
        if self.is_encode_text:                    
            if 'blip' in self.vpr_model_name:
                text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
                text_tokens = text_inputs.input_ids.to(img.device)
                attention_mask = text_inputs['attention_mask'].to(img.device)                
                text_embeds_all = self.text_encoder.encode_text(input_ids=text_tokens, attention_mask=attention_mask)
                text_embeds = text_embeds_all[:,0]
            elif 'clip' in self.vpr_model_name or 'siglip' in self.vpr_model_name:
                text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
                text_tokens = text_inputs.input_ids.to(img.device)
                attention_mask = None
                if 'attention_mask' in text_inputs:
                    attention_mask = text_inputs['attention_mask'].to(img.device)                
                text_embeds = self.text_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask)
            elif 'eva' in self.vpr_model_name:
                text_tokens = self.tokenizer(text).to(self.device)            
                text_embeds = self.text_encoder.encode_text(text_tokens)    
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            else:                
                text_tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(img.device)
                if self.is_trainable_text_encoder:
                    model_output = self.text_encoder(**text_tokens, output_hidden_states=True, return_dict=True)                                     
                    text_embeds_not_normilized = model_output[0][:, 0]
                    text_embeds = torch.nn.functional.normalize(text_embeds_not_normilized, p=2, dim=1)
                else:
                    with torch.no_grad():      
                        model_output = self.text_encoder(**text_tokens, output_hidden_states=True, return_dict=True)                                     
                        text_embeds_not_normilized = model_output[0][:, 0]                    
                        text_embeds = torch.nn.functional.normalize(text_embeds_not_normilized, p=2, dim=1)    
                text_embeds_all = model_output.last_hidden_state       
                attention_mask = text_tokens['attention_mask']      
            text_embeds_orig = text_embeds
                        
            # attention pooling on text
            if self.is_text_pooling:
                text_features = self.text_pooling(text_embeds_all, mask=attention_mask)
                text_embeds = torch.nn.functional.normalize(text_features, p=2, dim=1)
                
            # attention pooling on image
            if self.is_image_pooling:
                image_features = self.image_pooling(img_embeds_all)
                img_embeds = torch.nn.functional.normalize(image_features, p=2, dim=1)
        
        batch_size = img.shape[0]   
        
        if self.is_encode_image and self.is_encode_text:        
            if self.cross_modal == 3:
                img_embeds = self.vpr_proj(img_embeds)
                text_embeds = self.text_proj(text_embeds)
                img_embeds = torch.nn.functional.normalize(img_embeds, p=2, dim=1)
                text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
                embeds = img_embeds                

            elif self.fusion_type == 'mlp':
                embeds_input = torch.cat([img_embeds, text_embeds], dim=1)
                embeds = self.fusion(embeds_input) 
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            elif self.fusion_type == 'add':
                embeds = self.vpr_proj(img_embeds) + self.text_proj(text_embeds)
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            elif self.fusion_type == 'dynamic_weighting':                
                # calc dynamic weighting            
                embeds_input = torch.cat([img_embeds, text_embeds], dim=1)
                w = self.fusion(embeds_input)                
                embeds = img_embeds
            elif self.fusion_type == 'fixed_weighting':
                # use fixed weighting
                w = self.w_alpha
                w = torch.clamp(w, min=0, max=1)
                embeds = img_embeds            
                           
        elif self.is_encode_text:
            embeds = text_embeds
            embeds_orig = text_embeds_orig
        elif self.is_encode_image:
            embeds = img_embeds

        return embeds, text_embeds, w, embeds_orig, text_embeds_orig
    
    def encoder_image(self, img):
        img_embeds = self.vpr_encoder.backbone(img)
        img_embeds = self.vpr_encoder.aggregator(img_embeds)
        return img_embeds
    
    def encode_text(self, text):
        text_tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.my_device)
        model_output = self.text_encoder(**text_tokens)            
        text_embeds = model_output[0][:, 0]
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        return text_embeds
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        # max grad norm clipping
        # max_grad_norm = 5.0                
        # clip_grad_norm_(self.parameters(), max_norm=max_grad_norm)

        optimizer.step(closure=optimizer_closure)

            
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels, text_embeds, w, orig_descriptors, orig_text_embeds):
        
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:            
            if self.cross_modal:
                ref_labels = labels.clone()
                miner_outputs = self.miner(descriptors, labels, ref_emb=text_embeds, ref_labels=ref_labels)     
                loss = self.loss_fn(descriptors, labels, indices_tuple=miner_outputs, ref_emb=text_embeds, ref_labels=ref_labels)
            else:
                miner_outputs = self.miner(descriptors, labels)     
                loss = self.loss_fn(descriptors, labels, miner_outputs, embeds2=text_embeds, w=w)            
            
            if w is not None:
                if len(w.shape) > 1:
                    w_i = w[:,0].mean()
                else:
                    w_i = w.mean()
                self.log('w_i', w_i.item(), logger=True)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            if self.cross_modal == 4: # contrastive loss
                # contrastive loss cross modal
                logit_scale = self.contrastive_logit_scale
                loss = self.contrastive_loss(descriptors, text_embeds, logit_scale)                            
            else:
                loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels, texts = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        flat_texts = []
        for i in range(BS):
            for j in range(N):
                flat_texts.append(texts[j][i])

        # Feed forward the batch to the model
        descriptors, text_embeds, w, descriptors_orig, text_embeds_orig = self(images, flat_texts) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels, text_embeds, w, descriptors_orig, text_embeds_orig) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        
        # if batch_idx == 1:   # 0, 1 → two batches
        #     self.trainer.should_stop = True
        
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        # calculate descriptors
        descriptors, text_embeds, w, _, _ = self(places, texts)
        #return descriptors.detach().cpu()
        descriptors = descriptors.detach().cpu()
        text_embeds_cpu = None
        if text_embeds[0] is not None:
            text_embeds_cpu = text_embeds.detach().cpu()
        w_cpu = None
        if w is not None and w[0] is not None:
            w_cpu = w.detach().cpu()
        ret_dict = {'descriptors': descriptors, 'text_embeds': text_embeds_cpu, 'w': w_cpu}
        return ret_dict
    
    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            # stack all descriptors
            descriptors = []
            text_embeds = []
            w = []
            for d in val_step_outputs[i]:
                for key, value in d.items():
                    if key == 'descriptors':
                        descriptors.append(value)
                    elif key == 'text_embeds' and value is not None:
                        text_embeds.append(value)
                    elif key == 'w' and value is not None:
                        w.append(value)                        
            
            feats = torch.cat(descriptors, dim=0)
            text_feats = None
            if text_embeds != []:
                text_feats = torch.cat(text_embeds, dim=0)
            w_feats = None
            if w != []:
                w_feats = torch.cat(w, dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                # num_references = val_dataset.dbStruct.numDb
                num_references = val_dataset.num_db
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            
            if self.cross_modal:
                r_text_list = text_feats[ : num_references]
                q_text_list = text_feats[num_references : ]
                
                pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                                    q_list=q_text_list,
                                                    k_values=[1, 5, 10, 15, 20, 50, 100],
                                                    gt=positives,
                                                    print_results=True,
                                                    dataset_name=val_set_name,
                                                    faiss_gpu=self.faiss_gpu
                                                )
                
            
            elif self.fusion_type == 'dynamic_weighting' or self.fusion_type == 'fixed_weighting' or self.fusion_type == 'transformer':
                r_text_list = text_feats[ : num_references]
                q_text_list = text_feats[num_references : ]
                r_w_list = w_feats[ : num_references]
                q_w_list = w_feats[num_references : ]
                
                pitts_dict = utils.get_validation_recalls_dynamic_fusion(r_list=r_list, 
                                                    q_list=q_list,
                                                    r_text_list=r_text_list,
                                                    q_text_list=q_text_list,
                                                    w_r=r_w_list,
                                                    w_q=q_w_list,
                                                    k_values=[1, 5, 10, 15, 20, 50, 100],
                                                    gt=positives,
                                                    print_results=True,
                                                    dataset_name=val_set_name,
                                                    faiss_gpu=self.faiss_gpu
                                                )
                
            else:

                pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                                    q_list=q_list,
                                                    k_values=[1, 5, 10, 15, 20, 50, 100],
                                                    gt=positives,
                                                    print_results=True,
                                                    dataset_name=val_set_name,
                                                    faiss_gpu=self.faiss_gpu
                                                )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
        
    def on_save_checkpoint(self, checkpoint):
        if self.is_trainable_text_encoder:
            # Lightning gives you where THIS checkpoint is being written            
            ckpt_cb = next(
                (cb for cb in self.trainer.checkpoint_callbacks 
                if isinstance(cb, pl.callbacks.ModelCheckpoint)),
                None
            )                      

            # Directory containing the checkpoint file
            ckpt_dir = os.path.dirname(ckpt_cb.dirpath)

            self.text_encoder.save_pretrained(ckpt_dir)
            print("Saved PEFT adapter to:", ckpt_dir)
    

    
class CLSReweightingPooler(nn.Module):
    """
    Combines the CLS token with attention-pooled tokens.
    Output: a single pooled vector per sequence.
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention for token-level importance
        self.attention = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)

        # Learnable mixing of CLS and attention-pooled vector
        self.mix = nn.Linear(hidden_size * 2, hidden_size)

        # Optional nonlinearity
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None, return_scores=False):
        """
        hidden_states: [B, T, H]
        mask (optional): [B, T] (1 = keep token, 0 = ignore)
        """

        # ---- 1. CLS embedding ----
        cls = hidden_states[:, 0]  # [B, H]

        # ---- 2. Attention scores for each token ----
        scores = self.attention(hidden_states).squeeze(-1)  # [B, T]
        
        # Mask out CLS token
        scores[:, 0] = -1e4

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e4)

        weights = torch.softmax(scores, dim=-1)  # [B, T]

        # ---- 3. Attention-based pooled vector ----
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # [B, H]

        # # ---- 4. Concatenate CLS + attention-pooled ----
        combined = torch.cat([cls, pooled], dim=-1)  # [B, 2H]        
        combined = self.dropout(combined) 

        # ---- 5. Learnable mixing ----
        pooled = self.activation(self.mix(combined))  # [B, H]
        
        #pooled = cls + attn_pooled  # [B, H]

        if return_scores:
            return pooled, weights  # return per-token weights
        return pooled   
    

def mean_pooling(token_embeddings, attention_mask):
    # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # Sum of the attention mask
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
    # Mean Pooling
    return sum_embeddings / sum_mask
    
class MeanReweightingPooler(nn.Module):
    """
    Combines the Mean token with attention-pooled tokens.
    Output: a single pooled vector per sequence.
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention for token-level importance
        self.attention = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)

        # Learnable mixing of CLS and attention-pooled vector
        self.mix = nn.Linear(hidden_size * 2, hidden_size)

        # Optional nonlinearity
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None, return_scores=False):
        """
        hidden_states: [B, T, H]
        mask (optional): [B, T] (1 = keep token, 0 = ignore)
        """

        # ---- 1. CLS embedding ----
        cls = mean_pooling(hidden_states, mask)  # [B, H]        

        # ---- 2. Attention scores for each token ----
        scores = self.attention(hidden_states).squeeze(-1)  # [B, T]
        
        # Mask out CLS token
        scores[:, 0] = -1e4

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e4)

        weights = torch.softmax(scores, dim=-1)  # [B, T]

        # ---- 3. Attention-based pooled vector ----
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # [B, H]

        # # ---- 4. Concatenate CLS + attention-pooled ----
        combined = torch.cat([cls, pooled], dim=-1)  # [B, 2H]        
        combined = self.dropout(combined) 

        # ---- 5. Learnable mixing ----
        pooled = self.activation(self.mix(combined))  # [B, H]
        
        #pooled = cls + attn_pooled  # [B, H]

        if return_scores:
            return pooled, weights  # return per-token weights
        return pooled   
    

