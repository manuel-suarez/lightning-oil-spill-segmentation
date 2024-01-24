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
parser.add_argument('normfilename')
parser.add_argument('varfilename')
args = parser.parse_args()

# Load checkpoint
model = OilSpillModule.load_from_checkpoint(args.checkpoint)
model.eval()

# Open image file
normimage = np.array(Image.open(args.normfilename))
varimage = np.array(Image.open(args.varfilename))

print(normimage.shape, varimage.shape)
# Open variance file

# Compose multichannel image

# Predict