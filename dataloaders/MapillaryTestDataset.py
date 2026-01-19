from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import posixpath

GT_ROOT = './datasets/' # BECAREFUL, this is the ground truth that comes with GSV-Cities

class MSLSTest(Dataset):
    def __init__(self, dataset_root, image_root, csv_path, mean_std, image_size=None):
        self.dataset_root = dataset_root
        self.image_root = image_root
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std['mean'], std=mean_std['std']),
        ]
        if image_size:
            image_size=(image_size,image_size)
            transformations.append(transforms.Resize(size=image_size, antialias=True))
        self.input_transform = transforms.Compose(transformations)

        self.dbImages = np.load(GT_ROOT+'msls_test/msls_test_dbImages.npy', allow_pickle=True)
        self.qImages = np.load(GT_ROOT+'msls_test/msls_test_qImages.npy', allow_pickle=True)
        
        self.images_paths_csv, self.descriptions = read_csv_file(csv_path, image_root)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_database = len(self.dbImages)
        self.num_queries = len(self.qImages)
        self.ground_truth = None
    
    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_root, self.images[index])
        img = Image.open(image_path).convert("RGB")
        
        description = ""      
        if image_path in self.images_paths_csv:   
            desc_index = self.images_paths_csv.index(image_path)
            max_length = 1024
            description = self.descriptions[desc_index][:max_length]   
            
            #remove image root from path
            city = image_path.replace(self.image_root, '').lstrip('/\\')
            #keep only city name
            city = city.split('/')[0].split('\\')[0]
            # add city to description
            description = 'city ' + city + ". " + description

        if self.input_transform:
            img = self.input_transform(img)

        return img, index, description

    def __len__(self):
        return len(self.images)
    
    def save_predictions(self, preds, path, k=5):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[i]).stem
                db = ' '.join([Path(self.dbImages[j]).stem for j in preds[i][:k]])
                f.write(f"{q} {db}\n")


@staticmethod
def read_csv_file(labels_file, image_root):    
    df = pd.read_csv(labels_file, 
        engine='python',  # Use python engine for better path handling
        encoding='utf-8',
        on_bad_lines='skip',
        quotechar='"',
        skipinitialspace=True)
    image_path = df['image_path'].values
    description = df['description'].values    
    image_path = [posixpath.join(image_root, p) for p in image_path]
    return image_path, description