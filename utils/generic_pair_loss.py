import torch

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction
import torch.nn.functional as F

class GenericPairLoss(BaseMetricLossFunction):
    def __init__(self, mat_based_loss, **kwargs):
        super().__init__(**kwargs)
        self.loss_method = (
            self.mat_based_loss if mat_based_loss else self.pair_based_loss
        )

    def get_mu_std(self, embeddings):
                
        mu_text  = 0.65
        std_text = 0.07
        min_text = -6.07
        max_text = 4.92
        
        if embeddings.shape[1] == 256:
            #cricavpr
            mu_img   = 0.0094
            std_img  = 0.026
            min_img  = -5.67
            max_img  = 28.56    
        if embeddings.shape[1] == 512:
            #mixvpr 512
            mu_img   = 0.0111
            std_img  = 0.05
            min_img  = -5.24
            max_img  = 15.26
            
            #eigenplaces 512
            mu_img   = 0.043
            std_img  = 0.0596
            min_img  = -5.24
            max_img  = 15.08            
          
        elif embeddings.shape[1] == 4096: #mixvpr 4096
            mu_img   = 0.0048
            std_img  = 0.027
            min_img  = -5.55
            max_img  = 30.67
            
        elif embeddings.shape[1] == 10752: #cricavpr 10752
            mu_img   = 0.0094
            std_img  = 0.026
            min_img  = -5.67
            max_img  = 28.56
            
        return mu_img, std_img, min_img, max_img, mu_text, std_text, min_text, max_text
            
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels, embeds2, w):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        #mat = self.distance(embeddings, ref_emb)
        
        # cross modal case
        if w is None:
            s_ij = torch.matmul(embeddings, embeds2.T)
            
        # dual encoder or single encoder case
        else:        
            img_sim = torch.matmul(embeddings, embeddings.T)
        
            is_normalize = 0
            # normalize features
            if is_normalize:
                mu_img, std_img, min_img, max_img, mu_text, std_text, min_text, max_text = self.get_mu_std(embeddings)               
                img_sim  = torch.clamp(img_sim, min=-1.0, max=1.0)
                img_sim  = (img_sim - mu_img) / std_img        
                img_sim  = ((img_sim - min_img) / (max_img - min_img)) *2-1              
                #TBD: is_trainable_text_encoder
                # img_sim  = (img_sim - mu_text) / std_text
                # img_sim  = ((img_sim - min_text) / (max_text - min_text)) *2-1      
            
            s_ij = img_sim
            
            if embeds2 is not None:
                text_sim = torch.matmul(embeds2, embeds2.T)
                
                # normalize features
                if is_normalize:
                    text_sim = torch.clamp(text_sim, min=-1.0, max=1.0)        
                    text_sim = (text_sim - mu_text) / std_text
                    text_sim = ((text_sim - min_text) / (max_text - min_text)) *2-1          
                
                s_ij = text_sim      

                # calculate dynamic weights
                if len(w.shape) > 1:
                    w_i = w[:,0].unsqueeze(1)
                    w_t = w[:,1].unsqueeze(1)   
                    w_i_ij = ((w_i.unsqueeze(1) + w_i.unsqueeze(0)) / 2.0).squeeze(-1)
                    w_t_ij = ((w_t.unsqueeze(1) + w_t.unsqueeze(0)) / 2.0).squeeze(-1)
                    s_ij = w_i_ij * img_sim + w_t_ij * text_sim
                else:
                    # calculate fixed weights
                    s_ij = w*img_sim + (1-w)*text_sim

        return self.loss_method(s_ij, indices_tuple)

    def _compute_loss(self):
        raise NotImplementedError

    def mat_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        self._assert_either_pos_or_neg(pos_mask, neg_mask)
        return self._compute_loss(mat, pos_mask, neg_mask)

    def pair_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)

    @staticmethod
    def _assert_either_pos_or_neg(pos_mask, neg_mask):
        assert not torch.any(
            (pos_mask != 0) & (neg_mask != 0)
        ), "Each pair should be either be positive or negative"
