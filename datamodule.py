import os
import pandas as pd

from dataset import OilSpillTrainingDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

class OilSpillDataModule(LightningDataModule):
    def __init__(self, data_dir, train_dataset, valid_dataset, test_dataset):
        super().__init__()
        self.features_path = os.path.join(data_dir, 'features')
        self.labels_path = os.path.join(data_dir, 'labels')
        self.feature_ext = '.tiff'
        self.label_ext = '.pgm'
        self.dims = [224, 224, 3]
        self.features_channels = ['ORIGIN', 'ORIGIN', 'VAR']

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def setup(self, stage):
        # Train/valid set
        if stage == "fit":
            trainSet = pd.read_csv(self.train_dataset)
            validSet = pd.read_csv(self.valid_dataset)
            self.train_set = OilSpillTrainingDataset(
                trainSet["key"],
                self.features_path,
                self.labels_path,
                self.features_channels,
                self.feature_ext,
                self.label_ext,
                self.dims)
            self.valid_set = OilSpillTrainingDataset(
                validSet["key"],
                self.features_path,
                self.labels_path,
                self.features_channels,
                self.feature_ext,
                self.label_ext,
                self.dims)
        elif stage == "test":
            testSet = pd.read_csv(self.test_dataset)
            self.test_set = OilSpillTrainingDataset(
                testSet['key'],
                self.features_path,
                self.labels_path,
                self.features_channels,
                self.feature_ext,
                self.label_ext,
                self.dims)
        elif stage == "predict":
            testSet = pd.read_csv(self.test_dataset)
            self.test_set = OilSpillTrainingDataset(
                testSet['key'],
                self.features_path,
                self.labels_path,
                self.features_channels,
                self.feature_ext,
                self.label_ext,
                self.dims)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count())

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count())