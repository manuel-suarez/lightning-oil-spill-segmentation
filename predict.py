import os
import argparse
import logging
import numpy as np
import lightning as L
from module import OilSpillModule
from dataset import OilSpillPredictionDataset
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
# logging
logging.basicConfig(filename=f"{args.imagekey}_predict.log", filemode='w',
                    format='%(message)s', level=logging.INFO)

# Load checkpoint
model = OilSpillModule.load_from_checkpoint(args.checkpoint)
model.eval()
model.float()

# Create image dataset
dataset = OilSpillPredictionDataset(args.imagedir, args.imagekey)
dataloader = DataLoader(dataset, batch_size=67, shuffle=False, num_workers=os.cpu_count()//2)
trainer = L.Trainer(devices=1)
predictions = trainer.predict(model, dataloader)
print(type(predictions))
#predictions = np.array(predictions)
#print(predictions.shape)

#print(predictions)
print("Done!")