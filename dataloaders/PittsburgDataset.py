from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat

import torchvision.transforms as T
import torch.utils.data as data
import os
import shutil
import pandas as pd
import numpy as np
from glob import glob
import posixpath
from PIL import Image
from sklearn.neighbors import NearestNeighbors

root_dir = '/mnt/d/data/pitts30k/orig'
images_root = '/mnt/d/data/pitts30k/Images/val/'
csv_path = 'datasets/descriptions/pitts30k_val_descriptions.csv'

# if not exists(root_dir):
#     raise FileNotFoundError(
#         'root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')


def input_transform(image_size=None):
    return T.Compose([
        T.Resize(image_size),# interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_whole_val_set_with_dir(input_transform, image_dir, csv_file):
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform, image_root=image_dir, csv_path=csv_file)

def get_whole_val_set(input_transform):
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_250k_val_set(input_transform):
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_whole_test_set(input_transform):
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_250k_test_set(input_transform):
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)

def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
                    utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr)
    

def copy_image_list(image_list):
    dst_path = 'out_dir'
    # makedir if not exists
    os.makedirs(dst_path, exist_ok=True)

    for image in image_list:
        image_dst = os.path.join(dst_path, os.path.basename(image))
        shutil.copy(image, image_dst)


def read_images_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over very large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing images

    Returns
    -------
    images_paths : list[str], paths of images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        print(f"Searching test images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*", recursive=True))
        images_paths = [p for p in images_paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any images")
    return images_paths

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False, image_root=None, csv_path=None):
        super().__init__()

        self.input_transform = input_transform

        #self.dbStruct = parse_dbStruct(structFile)
        # self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        # if not onlyDB:
        #     self.images += [join(queries_dir, qIm)
        #                     for qIm in self.dbStruct.qImage]

        # self.whichSet = self.dbStruct.whichSet
        # self.dataset = self.dbStruct.dataset
        
        #db = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        #q = [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
        #copy_image_list(db)
        #copy_image_list(q)
        
        self.image_path, self.description, self.image_path_database, self.image_path_queries = WholeDatasetFromStruct.read_csv_file(csv_path, image_root)        
        self.num_db = len(self.image_path_database)
        self.num_queries = len(self.image_path_queries)
        
        self.positives = None
        self.distances = None                
        

    def __getitem__(self, index):
        # img_path = self.images[index]
        img_path = self.image_path[index]
        img = Image.open(img_path).convert('RGB')

        if self.input_transform:
            img = self.input_transform(img)
        
        description = self.description[index]  
        max_len = 512
        description = description[:max_len]

        return img, index, description

    def __len__(self):
        # return len(self.images)
        return len(self.image_path)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            # Read UTM coordinates, which must be contained within the paths
            # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
            try:
                # This is just a sanity check
                image_path = self.image_path[0]
                utm_east = float(image_path.split("@")[1])
                utm_north = float(image_path.split("@")[2])
            except:
                raise ValueError(
                    "The path of images should be path/to/file/@utm_east@utm_north@...@.jpg "
                    f"but it is {image_path}, which does not contain the UTM coordinates."
                )

            # database_paths = self.image_path[self.index_database]
            # queries_paths = self.image_path[self.index_queries]
            self.database_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in self.image_path_database]
            ).astype(float)
            self.queries_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in self.image_path_queries]
            ).astype(float)

            # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            posDistThr = 25 #self.dbStruct.posDistThr
            self.distances, self.positives = knn.radius_neighbors(
                self.queries_utms, radius=posDistThr)
      
        return self.positives
    
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
        image_path_database = [item for item in image_path if 'database' in item]
        image_path_queries = [item for item in image_path if 'queries' in item]
        # index_database = np.array([i for i, text in enumerate(image_path) if 'database' in text])
        # index_queries = np.array([i for i, text in enumerate(image_path) if 'queries' in text])
        
        image_paths = image_path_database + image_path_queries

        return image_paths, description, image_path_database, image_path_queries
