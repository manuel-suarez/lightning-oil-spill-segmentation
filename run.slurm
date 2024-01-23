#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=lightning-oil-segmentation
#SBATCH --mem=0
#SBATCH --time=0
#SBATCH --mail-user=manuel.suarez@cimat.mx
#SBATCH --mail-type=ALL

# Dataset Cimat paths
num_dataset=20
num_epochs=1
base=$HOME
dataset=$base/data/projects/consorcio-ia/data/oil_spills_17
input_path=$dataset/augmented_dataset
output_path=$dataset/training_results
traindataset=$input_path/learnigCSV_geo/trainingFiles/train$num_dataset'.csv'
crossdataset=$input_path/learnigCSV_geo/crossFiles/cross$num_dataset'.csv'
testdataset=$input_path/learnigCSV_geo/testingFiles/test$num_dataset'.csv'
# Training params
#arch='PSPNet'
for arch in Unet #UnetPlusPlus MAnet Linknet FPN PSPNet PAN DeepLabV3 DeepLabV3Plus
do
  ~/miniconda3/envs/py38/bin/python main.py $arch $input_path $output_path $traindataset $crossdataset $testdataset $num_epochs
done
echo 'Done!'