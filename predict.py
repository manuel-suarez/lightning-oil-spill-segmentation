import os
import argparse
import numpy as np
import torch
import lightning as L
from tqdm import tqdm
from PIL import Image
from module import OilSpillModule
from dataset import OilSpillPredictionDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Parse arguments
parser = argparse.ArgumentParser(
    prog='Oil spill cimat dataset segmentation',
    description='Segmentation on Cimat oil spill dataset',
    epilog='With a great power comes a great responsability'
)
parser.add_argument('checkpoint')
parser.add_argument('imagedir')
parser.add_argument('imagekey')
args = parser.parse_args()

# Load checkpoint
model = OilSpillModule.load_from_checkpoint(args.checkpoint)
model.eval()
model.float()

# Create image dataset
dataset = OilSpillPredictionDataset(args.imagedir, args.imagekey)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count()//2)
trainer = L.Trainer()
predictions = trainer.predict(model, dataloader)

print(len(predictions))
print(predictions)
print("Done!")