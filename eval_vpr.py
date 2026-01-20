import argparse
import eval_parser
from argparse import Namespace
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from model.LaVPR_wrapper import LaVPR_wrapper
import os
from dataloaders.test_dataset import TestDataset
from dataloaders.MapillaryTestDataset import MSLSTest
import utils.visualizations as visualizations
from math import sqrt
from sklearn.decomposition import PCA
from scipy.stats import norm
from typing import Tuple, List
from scipy.interpolate import interp1d
from scipy.stats import ecdf
import pandas as pd
import cv2
import kornia as K
import kornia.feature as KF
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import re


def rerank_predictions_by_text_or_image(vision_scores, vision_predictions, text_scores, text_predictions, max_results):
    #get max_results of vision predictions and sort these ids by their text scores
    top_vision_ids = vision_predictions[:, :max_results]
    # filter  top_vision_ids from text predictions         
    top_text_scores = np.take_along_axis(text_scores, top_vision_ids, axis=1)
    top_texts_ids =  np.take_along_axis(text_predictions, top_vision_ids, axis=1)
    # #sort top_text_scores and return their indices from highest to lowest
    text_indices = np.flip(np.argsort(top_text_scores, axis=1), axis=1)
    # #get the final predictions
    final_predictions = np.take_along_axis(top_texts_ids, text_indices, axis=1)
    final_scores = np.take_along_axis(top_text_scores, text_indices, axis=1)
    return final_scores, final_predictions


def rerank_predictions_by_scores(test_ds, vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index):
    # sum scores according the where vision and text predictions are the same
    combined_scores = []
    combined_predictions = []    

    logger.info(f"mean w_alpha vision: {w_alpha[:,0].mean()}, {w_alpha[:,0].std()}")
    logger.info(f"mean w_alpha text: {w_alpha[:,1].mean()}, {w_alpha[:,1].std()}")

    for v_scores, v_preds, t_scores, t_preds in zip(vision_scores, vision_predictions, text_scores, text_predictions):
        score_dict = {}                              
        w_query_v = w_alpha[query_index][0]                 
        for score, pred in zip(v_scores, v_preds):                               
            alpha_vision = (w_alpha[pred][0]+w_query_v)/2                
            if pred not in score_dict:
                score_dict[pred] = 0
            score_dict[pred] += alpha_vision * score                 
        
        w_query_t = w_alpha[query_index][1]
        for score, pred in zip(t_scores, t_preds):            
            alpha_text = (w_alpha[pred][1]+w_query_t)/2                
            if pred not in score_dict:
                score_dict[pred] = 0            
            score_dict[pred] += alpha_text * score 
            
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


def normlize_features(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)    


def encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha):
    if args.bfloat16:
        images = images.bfloat16()
    if args.encode_mode == 'text':
        # single vector - text
        descriptors = model.encode_text(texts)
        descriptors = descriptors.to(torch.float32).cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors        
        text_descriptors[indices.numpy(), :] = descriptors 
    elif args.encode_mode == 'image':
        # single vector - image
        descriptors = model.encode_image(images.to(args.device))
        descriptors = descriptors.to(torch.float32).cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors    
        vision_descriptors[indices.numpy(), :] = descriptors
    elif args.cross_modal==1:
        image_features = model.encode_text(texts)
        image_features = image_features.cpu().numpy()
        vision_descriptors[indices.numpy(), :] = image_features     
        text_features = model.encode_image(images.to(args.device))
        text_features = text_features.cpu().numpy()
        text_descriptors[indices.numpy(), :] = text_features            
    elif args.is_dual_encoder:
        image_features, text_features = model.encode_dual(images.to(args.device), texts)
        # cat fusion: concat text and vision vectors
        if args.dual_encoder_fusion == 'cat':
            descriptors = torch.cat((image_features, text_features), dim=1)
            descriptors = descriptors.to(torch.float32).cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        # each fusion: save each modality
        image_features = image_features.to(torch.float32).cpu().numpy()
        vision_descriptors[indices.numpy(), :] = image_features        
        text_features = text_features.to(torch.float32).cpu().numpy()
        text_descriptors[indices.numpy(), :] = text_features                    
    else:
        # single vector of both image and text
        descriptors, text_features, w = model.encode_single(images.to(args.device), texts)
        descriptors = descriptors.cpu().numpy()        
        if args.cross_modal:
            vision_descriptors[indices.numpy(), :] = descriptors
            text_features = text_features.cpu().numpy()
            text_descriptors[indices.numpy(), :] = text_features
        elif args.fusion_type == 'dynamic_weighting' or args.fusion_type == 'fixed_weighting' or args.fusion_type == 'text_adapter' or args.fusion_type == 'transformer':
            vision_descriptors[indices.numpy(), :] = descriptors
            text_features = text_features.cpu().numpy()
            text_descriptors[indices.numpy(), :] = text_features  
            w = w.cpu().numpy()
            if args.fusion_type == 'fixed_weighting':
                #make w a 2D vector of [w, 1-w]. w in numpy
                w = np.repeat(w, indices.shape[0], axis=0)
                #make w a 2D vector of [w, 1-w]
                w_alpha[indices.numpy(), :] = np.stack([w, 1-w], axis=1)                
            else:
                w_alpha[indices.numpy(), :] = w
        else:
            all_descriptors[indices.numpy(), :] = descriptors        
            
def get_queries_predictions(encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results):
     # Use a kNN to find predictions
    #faiss_index = faiss.IndexFlatL2(encoder_dim)
    faiss_index = faiss.IndexFlatIP(encoder_dim)
    #normilize descriptors for cosine similarity
    database_descriptors = normlize_features(database_descriptors)      
    queries_descriptors = normlize_features(queries_descriptors)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    scores, predictions = faiss_index.search(queries_descriptors, max_results)
    return scores, predictions


def do_pca(descriptors, pca_dim):
    logger.debug("Fitting PCA on all descriptors")
    pca = PCA(n_components=pca_dim)
    pca.fit(descriptors)
    logger.debug("Transforming all descriptors using PCA")                        
    descriptors = pca.transform(descriptors)
    return descriptors


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"Testing with {args.vpr_model_name}")
    logger.info(f"The outputs are being saved in {log_dir}")

    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP_MEAN_STD = {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    dataset_mean_std = IMAGENET_MEAN_STD
    if 'blip' in args.vpr_model_name.lower() or 'clip' in args.vpr_model_name.lower() or 'eva' in args.vpr_model_name.lower():
        dataset_mean_std = BLIP_MEAN_STD
    elif 'siglip' in args.vpr_model_name.lower():
        dataset_mean_std = SIGLIP_MEAN_STD

    model = LaVPR_wrapper(args)
    logger.info(f"VLM encoder dim: {model.encoder_dim}")
    


    is_msls_challenge = False
    if 'msls_challenge' in args.image_root:        
        test_ds = MSLSTest(dataset_root=args.database_folder, image_root=args.image_root, csv_path=args.queries_csv, mean_std=dataset_mean_std, image_size=args.image_size)
        is_msls_challenge = True
    else:
        test_ds = TestDataset(
            args.database_folder,   
            args.queries_folder,
            args.queries_csv,
            args.image_root,        
            mean_std=dataset_mean_std,
            positive_dist_threshold=args.positive_dist_threshold,
            image_size=args.image_size,
            use_labels=args.use_labels,
        )
    logger.info(f"Testing on {test_ds}")
    all_descriptors = None
    vision_descriptors = None
    text_descriptors = None
    
    max_results = max(args.recall_values)
    query_index = 0

    logger.info(f"VPR dimension: {model.vpr_encoder_dim}, text dimension: {model.text_encoder_dim}, fusion type: {args.fusion_type}, is text pooling: {args.is_text_pooling}, is dual encoder: {args.is_dual_encoder}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )

        vision_descriptors = np.empty((len(test_ds), model.vpr_encoder_dim), dtype="float32")
        text_descriptors = np.empty((len(test_ds), model.text_encoder_dim), dtype="float32")            
        all_descriptors = np.empty((len(test_ds), model.encoder_dim), dtype="float32")
        w_alpha = np.empty((len(test_ds), 2), dtype="float32")
        w_alpha[:,0] = args.alpha_vision
        w_alpha[:,1] = 1.0-args.alpha_vision
            
        for images, indices, texts in tqdm(database_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)

        query_index = test_ds.num_database
        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size)#1)
        for images, indices, texts in tqdm(queries_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)
        
        if args.is_pca:            
            vision_descriptors = do_pca(vision_descriptors, args.embeds_dim)
            model.vpr_encoder_dim = args.embeds_dim
            logger.info(f"PCA reduced vision descriptors to dimension: {model.vpr_encoder_dim}")
            if args.encode_mode == 'image':
                all_descriptors = vision_descriptors
                model.encoder_dim = all_descriptors.shape[1]
            if args.fusion_type == 'cat' or (args.is_dual_encoder and args.dual_encoder_fusion == 'cat'):
                all_descriptors = np.concatenate((vision_descriptors, text_descriptors), axis=1)
                model.encoder_dim = all_descriptors.shape[1]
                logger.info(f"Concatenated descriptors dimension: {model.encoder_dim}")
   
    alpha = args.alpha_vision
    max_results_reranking = test_ds.num_database    
        
    if args.cross_modal:
        vision_database_descriptors = vision_descriptors[: test_ds.num_database]    
        text_queries_descriptors = text_descriptors[test_ds.num_database :]
        scores, predictions = get_queries_predictions(model.encoder_dim, vision_database_descriptors, all_descriptors, text_queries_descriptors, max_results)
        
    elif (args.is_dual_encoder and args.dual_encoder_fusion=='each') or args.fusion_type=='dynamic_weighting' or args.fusion_type=='fixed_weighting' or args.fusion_type=='text_adapter' or args.fusion_type=='transformer' or args.rerank_by_text_or_image:         
        
        # vision
        vision_queries_descriptors = vision_descriptors[test_ds.num_database :]
        vision_database_descriptors = vision_descriptors[: test_ds.num_database]                    
        vision_scores, vision_predictions = get_queries_predictions(model.vpr_encoder_dim, vision_database_descriptors, vision_descriptors, vision_queries_descriptors, max_results_reranking)
        # text
        text_queries_descriptors = text_descriptors[test_ds.num_database :]
        text_database_descriptors = text_descriptors[: test_ds.num_database]                
        text_scores, text_predictions = get_queries_predictions(model.text_encoder_dim, text_database_descriptors, text_descriptors, text_queries_descriptors, max_results_reranking)
        if args.rerank_by_text_or_image == 1: # rerank by text
            scores, predictions = rerank_predictions_by_text_or_image(vision_scores, vision_predictions, text_scores, text_predictions, args.max_rerank)
        elif args.rerank_by_text_or_image == 2: # rerank by image
            scores, predictions = rerank_predictions_by_text_or_image(text_scores, text_predictions, vision_scores, vision_predictions, args.max_rerank)
        # join vision and text predictions        
        elif args.rerank_by_scores:
            scores, predictions = rerank_predictions_by_scores(test_ds, vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index)            
    else:
        queries_descriptors = all_descriptors[test_ds.num_database :]
        database_descriptors = all_descriptors[: test_ds.num_database]    
        logger.info(f"dim database descriptors: {all_descriptors.shape[1]}")
        # get queries predictions
        scores, predictions = get_queries_predictions(model.encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results)        
        
    if is_msls_challenge:
        # save predictions to msls_challenge format
        test_ds.save_predictions(predictions, log_dir / "msls_challenge_predictions.txt", k=25)
    else:
        # For each query, check if the predictions are correct
        if args.use_labels:
            positives_per_query = test_ds.get_positives()
            recalls = np.zeros(len(args.recall_values))
            for query_index, preds in enumerate(predictions):
                for i, n in enumerate(args.recall_values):
                    if np.any(np.isin(preds[:n], positives_per_query[query_index])):
                        recalls[i:] += 1
                        break

            # Divide by num_queries and multiply by 100, so the recalls are in percentages
            recalls = recalls / test_ds.num_queries * 100
            recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
            logger.info(recalls_str)
            
            # open eval_vpr_results.csv in append mode and write the recalls
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{args.vpr_model_name},{w_alpha[0,0]},{args.fusion_type},{args.is_text_pooling},{args.vpr_dim},{args.is_pca},{args.encode_mode},{recalls_str}\n")
            
    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels, test_ds.images_paths_csv, texts=test_ds.descriptions
        )


if __name__ == "__main__":
    args = eval_parser.parse_arguments()
    main(args)
