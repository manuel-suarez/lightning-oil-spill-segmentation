import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from module import OilSpillModule

# Parse arguments
parser = argparse.ArgumentParser(
    prog='Oil spill cimat dataset segmentation',
    description='Segmentation on Cimat oil spill dataset',
    epilog='With a great power comes a great responsability'
)
parser.add_argument('checkpoint')
parser.add_argument('basedir')
parser.add_argument('keyfile')
args = parser.parse_args()
keyfile = args.keyfile
img_dir = os.path.join(args.basedir, keyfile)
print(img_dir)

# Load checkpoint
model = OilSpillModule.load_from_checkpoint(args.checkpoint)
model.eval()

# Open image file
normfile = os.path.join(img_dir, f"{keyfile}_norm.tif")
normimage = np.array(Image.open(normfile))

print(normfile, normimage.shape)
# Open variance file
varfile = os.path.join(img_dir, f"{keyfile}_var.tif")
varimage = np.array(Image.open(varfile))

print(varfile, varimage.shape)

# Compose multichannel image
src = np.zeros((normimage.shape[0], normimage.shape[1], 3))
# origin-origin-var
src[:, :, 0] = normimage
src[:, :, 1] = normimage
src[:, :, 2] = varimage
print(src.shape)

# Predict in patches
width = src.shape[1]
height = src.shape[0]

width_patch = 224
height_patch = 224

overlay = np.zeros((height, width), np.uint8)

nx = width // width_patch + 1
ny = height // height_patch + 1

for (j,i) in tqdm(zip(range(0, ny), range(0, nx))):
    y = height_patch * j
    x = width_patch * i

    if (x + width_patch > width):
        x = width - width_patch - 1
    if (y + height_patch > height):
        y = height - height_patch - 1

    z = x[y:y + height_patch, x:x + width_patch, ...]
    with torch.no_grad():
        overlay[y:y + height_patch, x:x + width_patch] = model(z)

print(overlay.shape)