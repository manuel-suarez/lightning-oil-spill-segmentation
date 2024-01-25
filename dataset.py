import os
import logging
import itertools
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class OilSpillTrainingDataset(Dataset):
    # List of classes (binary segmentation)
    CLASSES = ['oil', 'not oil']

    def __init__(self,
                 keys,              # List of file names
                 featuresPath,      # Path of feature files
                 labelsPath,        # Path of label files
                 featuresChannels,  # Feature channels specification
                 featureExt,        # File extension of feature files
                 labelExt,          # File extension of label files
                 dims
                 ):
        super().__init__()
        self.keys = keys
        self.dims = dims
        self.labelsPath = labelsPath
        self.featuresPath = featuresPath
        self.featuresChannels = featuresChannels
        self.featureExt = featureExt
        self.labelExt = labelExt

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # Key index (filename)
        key = self.keys[index]
        # Open feature channels and build multichannel image
        image = np.zeros(self.dims, dtype=np.float32)
        for j, feature in enumerate(self.featuresChannels):
            feature_path = os.path.join(self.featuresPath, feature, key + self.featureExt)
            #feature_image = imread(feature_path, as_gray=True).astype(np.float32)
            feature_image = np.array(Image.open(feature_path))

            image[...,j] = feature_image
        # Convert to pytorch format: HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        # Open label
        label_path = os.path.join(self.labelsPath, key + self.labelExt)
        label = np.array(Image.open(label_path))/255.0
        label = np.expand_dims(label, 0)
        #label = np.zeros((self.dims[0], self.dims[1], 1))
        #label[...,0] = imread(label_path, as_gray=True).astype(np.float32)/255.0
        #label = np.moveaxis(label, -1, 0)
        # Return data
        return image, label

class OilSpillPredictionDataset(Dataset):
    def __init__(self, image_dir, image_key, patch_size = 224):
        super().__init__()
        # Open and prepare multichannel image
        base_dir = os.path.join(image_dir, image_key)
        normfile = os.path.join(base_dir, f"{image_key}_norm.tif")
        #normimage = imread(normfile, as_gray=True).astype(np.float16)
        normimage = np.array(Image.open(normfile))
        print(f"Norm image shape: {normimage.shape}")
        varfile = os.path.join(base_dir, f"{image_key}_var.tif")
        #varimage = imread(varfile, as_gray=True).astype(np.float16)
        varimage = np.array(Image.open(varfile))
        print(f"Var image shape: {varimage.shape}")

        self.heigth = normimage.shape[0]
        self.width = normimage.shape[1]
        # Multichannel image
        self.src = np.zeros((self.heigth, self.width, 3))
        # origin-origin-var
        self.src[:, :, 0] = normimage
        self.src[:, :, 1] = normimage
        self.src[:, :, 2] = varimage
        print(f"Multichannel image shape: {self.src.shape}")

        self.patch_width = patch_size
        self.patch_height = patch_size

        self.nx = self.width // self.patch_width + 1
        self.ny = self.heigth // self.patch_height + 1
        print(f"Parches, nx: {self.nx}, ny: {self.ny}")

        self.ranges = list(itertools.product(range(0, self.ny), range(0, self.nx)))
        print(f"Num parches: {len(self.ranges)}, {self.nx * self.ny}")

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, index):
        (j,i) = self.ranges[index]
        logging.info(f"{index} patch, (j,i)=({j},{i})")

        x = self.patch_width * i
        y = self.patch_height * j
        if (x + self.patch_width > self.width):
            x = self.width - self.patch_width - 1
        if (y + self.patch_height > self.heigth):
            y = self.heigth - self.patch_height - 1

        patch = self.src[y:y + self.patch_height, x:x + self.patch_width, ...]
        patch = np.moveaxis(patch, -1, 0)

        return patch

def create_datasets(data_dir, train_dataset, cross_dataset, test_dataset):
    featuresPath = os.path.join(data_dir, 'features')
    labelsPath = os.path.join(data_dir, 'labels')
    featureExt = '.tiff'
    labelExt = '.pgm'
    dims = [224, 224, 3]
    featuresChannels = ['ORIGIN', 'ORIGIN', 'VAR']
    trainingSet = pd.read_csv(train_dataset)
    crossvalidSet = pd.read_csv(cross_dataset)
    testingSet = pd.read_csv(test_dataset)
    return (
        OilSpillTrainingDataset(trainingSet["key"], featuresPath, labelsPath, featuresChannels, featureExt, labelExt, dims),
        OilSpillTrainingDataset(crossvalidSet["key"], featuresPath, labelsPath, featuresChannels, featureExt, labelExt, dims),
        OilSpillTrainingDataset(testingSet['key'], featuresPath, labelsPath, featuresChannels, featureExt, labelExt, dims)
    )

def create_dataloaders(n_cpu, train_dataset, valid_dataset, test_dataset):
    return (
        DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=n_cpu//2),
        DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=n_cpu//2),
        DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=n_cpu//2)
    )
