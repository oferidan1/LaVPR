import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--positive_dist_threshold", type=int, default=25, help="distance (in meters) for a prediction to be considered a positive")
    
    parser.add_argument("--vpr_dim", type=int, default=512, help="_")
    parser.add_argument("--text_dim", type=int, default=1024, help="_")
    
    parser.add_argument("--database_folder", type=str, default="/mnt/d/data/amstertime/test/database")    
    parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/amstertime/test/queries")        
    parser.add_argument("--image_root", type=str, default="/mnt/d/data/amstertime/test")
    parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/amstertime_descriptions.csv")
    #parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/amstertime_descriptions_subset.csv")            
    
    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/pitts30k/images/test/database")    
    # parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/pitts30k/images/test/queries")    
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/pitts30k/images/test")    
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/pitts30k_test_descriptions.csv")

    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/pitts30k/images/val/database")    
    # parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/pitts30k/images/val/queries")    
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/pitts30k/images/val")    
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/pitts30k_val_descriptions.csv")    
    
    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/msls/val/database")    
    # parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/msls/val/query")   
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/msls/val/")    
    # # # # # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_val_descriptions.csv")
    # #parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_val_descriptions_blur.csv")
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_val_descriptions_weather.csv")

    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/msls_challenge")    
    # parser.add_argument("--queries_folder", type=str, default=None)       
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/msls_challenge/test")    
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_challenge_descriptions.csv")
    
    
    parser.add_argument("--num_workers", type=int, default=4, help="_")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="set to 1 if database images may have different resolution"
    )
    parser.add_argument(
        "--log_dir", type=str, default="default", help="experiment name, output logs will be saved under logs/log_dir"
    )
    parser.add_argument("--descriptor_dir", type=str, default="descriptors", help="_")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument(
        "--recall_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="values for recall (e.g. recall@1, recall@5)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="set to true if you have no labels and just want to "
        "do standard image retrieval given two folders of queries and DB",
    )
    parser.add_argument(
        "--num_preds_to_save", type=int, default=0, help="set != 0 if you want to save predictions for each query"
    )
    parser.add_argument(
        "--save_only_wrong_preds",
        action="store_true",
        help="set to true if you want to save predictions only for " "wrongly predicted queries",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=320,
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument(
        "--save_descriptors",
        action="store_true",
        help="set to True if you want to save the descriptors extracted by the model",
    )
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--vpr_model_name", type=str, default="mixvpr")
    parser.add_argument("--vpr_model_backbone", type=str, default="ResNet50")
    parser.add_argument("--text_model_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--lora_path", type=str, default=None)    
    parser.add_argument("--is_dual_encoder", type=int, default="0", help="is dual encoder")    
    parser.add_argument("--dual_encoder_fusion", type=str, default="cat", help="cat/each")    
    parser.add_argument("--encode_mode", type=str, default="both", help="both/image/text")   
    parser.add_argument("--fusion_type", type=str, default='none', help="type of fusion to use: mlp, add, transformer, dynamic_weighting, fixed_weighting, text_adapter")
    parser.add_argument("--is_normalize", type=int, default="0", help="is normalize features")    
    parser.add_argument("--max_results_reranking", type=int, default="25000", help="max results for reranking")    
    parser.add_argument("--alpha_vision", type=float, default=0.9, help="weight for vision scores in reranking")    
    parser.add_argument("--is_trainable_text_encoder", type=int, default="0", help="train text encoder or not")
    parser.add_argument("--is_encode_image", type=int, default="1", help="encode image or not")
    parser.add_argument("--is_encode_text", type=int, default="1", help="encode text or not")
    parser.add_argument("--rerank_by_scores", type=int, default="1", help="rerank_by_scores or rerank_by_rank")
    parser.add_argument("--is_pca", type=int, default="0", help="do pca on descriptors or not")
    parser.add_argument("--embeds_dim", type=int, default="512", help="embeds dimension")
    parser.add_argument("--is_text_pooling", type=int, default="0", help="pool text or not")
    parser.add_argument("--is_image_pooling", type=int, default="0", help="pool image or not")
    parser.add_argument("--cross_modal", type=int, default="0", help="cross modal 0=no/1=blip orig/2=our model")
    parser.add_argument("--rerank_by_text_or_image", type=int, default="0", help="rerank by text =1, by image=2")
    parser.add_argument("--max_rerank", type=int, default="100", help="max_rerank")    
    parser.add_argument("--bfloat16", type=int, default="0", help="bfloat16 or not")    


    args = parser.parse_args()
    
    args.use_labels = not args.no_labels
        
    return args
