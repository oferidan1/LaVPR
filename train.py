import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule, IMAGENET_MEAN_STD, BLIP_MEAN_STD, SIGLIP_MEAN_STD
from sentence_transformers import SentenceTransformer
import os
import argparse
from model.LaVPR import LaVPR


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Resume parameters
    parser.add_argument("--vpr_dim", type=int, default=512, help="dimension of the vpr embeddings")
    parser.add_argument("--vpr_model_name", type=str, default="mixvpr")
    parser.add_argument("--vpr_model_backbone", type=str, default="ResNet50")
    # Other parameters
    parser.add_argument("--gpu", type=str, default='0', help="gpu id(s) to use")    
    parser.add_argument("--epochs", type=int, default='10', help="number of epochs to train")    
    parser.add_argument("--train_csv", type=str, default="datasets/descriptions/gsv_cities_descriptions.csv")    
    parser.add_argument("--image_root", type=str, default="/mnt/d/data/gsv_cities/", help="root directory for images")
    #parser.add_argument("--val_csv", type=str, default="datasets/descriptions/pitts30k_val_descriptions.csv")    
    parser.add_argument("--val_csv", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")    
    parser.add_argument("--val_image_root", type=str, default="/mnt/d/data/pitts30k/images/val", help="root directory for images")
    parser.add_argument("--text_model_name", type=str, default="BAAI/bge-large-en-v1.5", help="text encoder model name")
    parser.add_argument("--is_freeze_text", type=int, default="1", help="freeze text encoder or not")
    parser.add_argument("--is_freeze_vpr", type=int, default="1", help="freeze vpr encoder or not")    
    parser.add_argument("--image_size", type=int, default="320", help="image size to vpr")
    parser.add_argument("--embeds_dim", type=int, default=512, help="dimension of the embeddings")
    parser.add_argument("--fusion_type", type=str, default='dynamic_weighting', help="type of fusion to use: mlp, add, dynamic_weighting, fixed_weighting")
    parser.add_argument("--is_encode_image", type=int, default="1", help="encode image or not")
    parser.add_argument("--is_encode_text", type=int, default="1", help="encode text or not")
    parser.add_argument("--is_text_pooling", type=int, default="1", help="pool text or not")
    parser.add_argument("--is_image_pooling", type=int, default="0", help="pool image or not")
    parser.add_argument("--is_pca", type=int, default="0", help="pool image or not")
    parser.add_argument("--is_trainable_text_encoder", type=int, default="0", help="train text encoder or not")
    parser.add_argument("--batch_size", type=int, default="120", help="batch size for training")
    parser.add_argument("--loss_name", type=str, default="MultiSimilarityLoss_Sij", help="name of the loss function to use")
    #parser.add_argument("--loss_name", type=str, default="MultiSimilarityLoss", help="name of the loss function to use")
    parser.add_argument("--cross_modal", type=int, default="0", help="cross modal 0=no/1=blip orig/2=our model/3=with projections/4=contrastive loss")    
    parser.add_argument("--is_val", type=int, default="1", help="run validation 0=no/1=yes")
    parser.add_argument("--lora_all_linear", type=int, default="0", help="lora all linear 0=no/1=yes")
    parser.add_argument("--lora_target_modules", nargs='+', default=["query", "value", "qkv"], help="when not lora_all_linear, lora target modules")    
    parser.add_argument("--lora_r", type=int, default="64", help="lora_all_linear 0=no/1=yes")     
    parser.add_argument("--text_dim", type=int, default=1024, help="dimension of the text embeddings")
    parser.add_argument("--img_per_place", type=int, default=4, help="number of images per place")
    args = parser.parse_args()
    
    return args            
            
if __name__ == '__main__':    
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    dataset_mean_std = IMAGENET_MEAN_STD
    image_size = args.image_size
    if 'blip' in args.vpr_model_name.lower() or 'clip' in args.vpr_model_name.lower() or 'eva' in args.vpr_model_name.lower():
        dataset_mean_std = BLIP_MEAN_STD
    elif 'siglip' in args.vpr_model_name.lower():
        dataset_mean_std = SIGLIP_MEAN_STD
    
    val_set_names = []
    if args.is_val:
        val_set_names = ['pitts30k_val']    
        
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(image_size, image_size),
        num_workers=4,#28,
        show_data_stats=True,
        mean_std=dataset_mean_std,
        #val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
        val_set_names=val_set_names,
        train_image_root=args.image_root,
        train_csv=args.train_csv,
        val_image_root=args.val_image_root,
        val_csv=args.val_csv,
    )

    if args.fusion_type == 'text_adapter':
        args.is_text_pooling = 1

    # examples of backbones
    # resnet18, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
    # efficientnet_b0, efficientnet_b1, efficientnet_b2
    # swinv2_base_window12to16_192to256_22kft1k
    model = LaVPR(
        #---- Encoder
        vpr_model_name=args.vpr_model_name.lower(),
        vpr_model_backbone=args.vpr_model_backbone,
        vpr_encoder_dim=args.vpr_dim,
        
        #---- Train hyperparameters
        lr=0.05, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
        optimizer='sgd', # sgd, adamw
        weight_decay=0.001, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=650,
        #milestones=[2],
        milestones=[2,4,6,8],
        lr_mult=0.3,

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=args.loss_name,
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
        text_model_name=args.text_model_name,
        embeds_dim=args.embeds_dim,
        is_freeze_vpr=args.is_freeze_vpr,
        is_freeze_text=args.is_freeze_text,
        fusion_type=args.fusion_type,
        is_encode_image=args.is_encode_image,
        is_encode_text=args.is_encode_text,
        is_trainable_text_encoder=args.is_trainable_text_encoder,
        is_text_pooling=args.is_text_pooling,
        is_image_pooling=args.is_image_pooling,
        cross_modal=args.cross_modal,
        lora_all_linear=args.lora_all_linear,
        lora_target_modules=args.lora_target_modules,
        lora_r=args.lora_r,
        text_encoder_dim=args.text_dim,
    )
    
    # if args.is_encode_image and  args.vpr_resume_model is not None:
    #     model_state_dict = torch.load(args.vpr_resume_model)
    #     model.vpr_encoder.load_state_dict(model_state_dict)
        
    model = model.to('cuda')
    
    if args.is_val:    
        # model params saving using Pytorch Lightning
        # we save the best 3 models accoring to Recall@1 on pittsburg val
        checkpoint_cb = ModelCheckpoint(
            monitor='pitts30k_val/R1',
            filename=f'{"resnet50"}' +
            '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=3,
            mode='max',)
    else:
        checkpoint_cb = ModelCheckpoint(        
            filename=f'{"resnet50"}' +
            '_epoch({epoch:02d})_step({step:04d})',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=-1,
            every_n_epochs=1,
            mode='max',)

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu', devices=[0],
        default_root_dir=f'./LOGS/{"resnet50"}', # Tensorflow can be used to viz

        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=16, # we use half precision to reduce  memory usage
        max_epochs=args.epochs,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )
    
    # # Manually call validation
    #trainer.validate(model=model, datamodule=datamodule)

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
