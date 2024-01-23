import os
import torch
import logging
import argparse
from datamodule import OilSpillDataModule
from dataset import OilSpillDataset
from module import OilSpillModule
from utils import save_figure

def process(base_dir, input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs):
    logging.info("Begin process")
    logging.info(f"\tArchitecture: {arch}")
    logging.info(f"\tEncoder: {encoder}")
    logging.info(f"\tInput dir: {input_dir}")
    logging.info(f"\tOutput dir: {output_dir}")
    classes = OilSpillDataset.CLASSES

    logging.info("1.- Datamodule configuration")
    datamodule = OilSpillModule(arch, encoder, in_channels=3, classes=len(classes))
    #logging.info(f"\tTrain dataset size: {len(datamodule.train_dataset)}")
    #logging.info(f"\tValid dataset size: {len(datamodule.valid_dataset)}")
    #logging.info(f"\tTest dataset size: {len(datamodule.test_dataset)}")

    figures_dir = os.path.join(base_dir, f"{arch}_figures")
    results_dir = os.path.join(base_dir, f"{arch}_results")
    logs_dir = os.path.join(base_dir, f"{arch}_logs")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # Samples
    # save_figure(datamodule.train_dataset, "Train", os.path.join(figures_dir, "figure_01.png"))
    # save_figure(datamodule.valid_dataset, "Valid", os.path.join(figures_dir, "figure_02.png"))
    # save_figure(datamodule.test_dataset, "Test", os.path.join(figures_dir, "figure_03.png"))

    logging.info("2.- Model instantiation")
    encoder = "resnet34"
    model = OilSpillModule(arch, encoder_name=encoder, in_channels=3, classes=1)

def main(arch, encoder, base_dir, input_dir, output_dir, train_dataset, cross_dataset, test_dataset, num_epochs):
    process(base_dir, input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Oil spill cimat dataset segmentation',
        description='Segmentation on Cimat oil spill dataset',
        epilog='With a great power comes a great responsability'
    )
    parser.add_argument('arch')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('train_dataset')
    parser.add_argument('cross_dataset')
    parser.add_argument('test_dataset')
    parser.add_argument('num_epochs')
    args = parser.parse_args()
    arch = args.arch
    base_dir = "results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join("results", f"{arch}_app.log"), filemode='w', format='%(asctime)s: %(name)s %(levelname)s - %(message)s', level=logging.INFO)
    # redirect lightning logging to file
    logger = logging.getLogger("lightning.pytorch")
    #logger.addHandler(logging.FileHandler("core.log"))

    logging.info("Start!")
    encoder = 'resnet34'
    main(arch, encoder, base_dir, args.input_dir, args.output_dir, args.train_dataset, args.cross_dataset, args.test_dataset, int(args.num_epochs))
    logging.info("Done!")
    print("Done!")