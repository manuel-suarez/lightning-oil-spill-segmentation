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

# Parse arguments
parser = argparse.ArgumentParser(
    prog='Oil spill cimat dataset segmentation',
    description='Segmentation on Cimat oil spill dataset',
    epilog='With a great power comes a great responsability'
)
parser.add_argument('imagedir')
parser.add_argument('imagekey')
parser.add_argument('num_epochs')
parser.add_argument('arch')
args = parser.parse_args()
# logging
base_dir = os.path.join(f"results_{args.num_epochs}_epochs", args.arch)
logging.basicConfig(filename=os.path.join(base_dir, f"{args.imagekey}_predict.log"), filemode='w',
                    format='%(message)s', level=logging.INFO)

# Load checkpoint
ckpt_fname = os.path.join(base_dir, 'resnet34.ckpt')
model = OilSpillModule.load_from_checkpoint(ckpt_fname)
model.eval()
model.float()

# Create image dataset
dataset = OilSpillPredictionDataset(args.imagedir, args.imagekey)
dataloader = DataLoader(dataset, batch_size=dataset.nx, shuffle=False, num_workers=os.cpu_count()//2)
trainer = L.Trainer(devices=1)
predictions = trainer.predict(model, dataloader)
print(type(predictions))
print(type(predictions[0]))
predictions = np.array([p.numpy().astype(np.int32)*255 for p in predictions])
print(predictions.shape)

result = rearrange(predictions, 'i j c h w -> (i h) (j w) c')
print(result.shape, np.max(result), np.min(result), np.count_nonzero(result == 255))
result = np.squeeze(result, -1)

#imsave(f"{args.imagekey}_result.png", result)
Image.fromarray(result).save(os.path.join(base_dir, 'results', f"{args.imagekey}_result.png"))
print("Done!")