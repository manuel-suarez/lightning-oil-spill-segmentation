import os
import argparse
import logging
import numpy as np
import lightning as L
from PIL import Image
from einops import rearrange
from module import OilSpillModule
from dataset import OilSpillPredictionDataset
from torch.utils.data import DataLoader
from skimage.io import imsave

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
print(type(predictions[0]))
predictions = np.array([p.numpy().astype(np.int32)*255 for p in predictions])
print(predictions.shape)

result = rearrange(predictions, 'i j c h w -> (i h) (j w) c')
print(result.shape, np.max(result), np.min(result))
result = np.squeeze(result, -1)

#imsave(f"{args.imagekey}_result.png", result)
Image.fromarray(result).save(f"{args.imagekey}_result.png")
print("Done!")