import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch

def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        r_list = r_list.to(torch.float32)
        q_list = q_list.to(torch.float32)

        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.isin(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performances on {dataset_name}"))
        
        return d


def get_validation_recalls_dynamic_fusion(r_list, q_list, r_text_list, q_text_list, w_r, w_q, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        embed_text_size = r_text_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index_i = faiss.GpuIndexFlatIP(res, embed_size, flat_config)
        # build index
        else:
            faiss_index_i = faiss.IndexFlatIP(embed_size)
            faiss_index_t = faiss.IndexFlatIP(embed_text_size)
        
        # add references
        r_list = r_list.to(torch.float32)
        q_list = q_list.to(torch.float32)
        r_text_list = r_text_list.to(torch.float32)
        q_text_list = q_text_list.to(torch.float32)

        faiss_index_i.add(r_list)
        faiss_index_t.add(r_text_list)

        # search for queries in the index
        max_k = min(10000, len(r_list))
        scores, predictions = faiss_index_i.search(q_list, max_k)
        scores_t, predictions_t = faiss_index_t.search(q_text_list, max_k)
        
        #loop over indexes and and re-rank predictions according to dynamic weights where index_i == index_t
        rerank_predictions(scores, predictions, scores_t, predictions_t, w_r, w_q, max_results=max(k_values))

        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.isin(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performances on {dataset_name}"))
        
        return d

@staticmethod
def rerank_predictions(vision_scores, vision_predictions, text_scores, text_predictions, w_r, w_q, max_results):
    # sum scores according the where vision and text predictions are the same
    combined_scores = []
    combined_predictions = []
    if len(w_r.shape) > 1:
        print("mean w_alpha vision:", w_r[:,0].mean(), w_r[:,0].std())
    else:
        print("mean w_alpha vision:", w_r.mean(), w_r.std())
        w_r = np.repeat(w_r, text_predictions.shape[0], axis=0)
        w_r = np.stack([w_r, 1-w_r], axis=1)     
        w_q = np.repeat(w_q, text_predictions.shape[0], axis=0)
        w_q = np.stack([w_q, 1-w_q], axis=1)     
    query_index = 0
    for v_scores, v_preds, t_scores, t_preds in zip(vision_scores, vision_predictions, text_scores, text_predictions):
        score_dict = {}
        w_query_v = w_q[query_index][0]
        for score, pred in zip(v_scores, v_preds):
            if pred not in score_dict:
                score_dict[pred] = 0
            #score_dict[pred] += w_alpha[pred][0] * score 
            score_dict[pred] += (w_r[pred][0]+w_query_v)/2 * score 
        w_query_t = w_q[query_index][1]
        for score, pred in zip(t_scores, t_preds):
            if pred not in score_dict:
                score_dict[pred] = 0
            #score_dict[pred] += w_r[pred][1] * score 
            score_dict[pred] += (w_r[pred][1]+w_query_t)/2 * score 
        # sort by score
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        preds_sorted = [item[0] for item in sorted_items][:max_results]
        scores_sorted = [item[1] for item in sorted_items][:max_results]
        combined_predictions.append(preds_sorted)
        combined_scores.append(scores_sorted)
        query_index += 1
        
    combined_predictions = np.array(combined_predictions)
    combined_scores = np.array(combined_scores)
    return combined_scores, combined_predictions

