import os
import argparse
import numpy as np
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
filename = os.path.join(img_dir, f"{keyfile}_norm.tif")
normimage = np.array(Image.open(filename))

print(filename, normimage.shape)
# Open variance file
filename = os.path.join(img_dir, f"{keyfile}_var.tif")
varimage = np.array(Image.open(args.varfilename))

print(filename, varimage.shape)
# Compose multichannel image

# Predict