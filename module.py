from typing import Any

import torch
import logging
from lightning import LightningModule
from segmentation_models_pytorch import create_model
from segmentation_models_pytorch.metrics import get_stats, accuracy
from segmentation_models_pytorch.losses import DiceLoss, BINARY_MODE

class OilSpillModule(LightningModule):
    def __init__(self, arch, encoder_name, in_channels, classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )
        self.classes = classes

        # Segmentation loss (default binary_crossentropy)
        self.loss_fn = DiceLoss(BINARY_MODE, from_logits=True)

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, label = batch
        bs = image.shape[0]
        h, w = image.shape[2:]

        assert image.ndim == 4
        assert image.shape == (bs, 3, h, w) # Multichannel dataset

        #label = batch["label"]
        assert label.ndim == 4
        assert label.shape == (bs, self.classes, h, w)

        assert label.max() <= 1 and label.min() >= 0

        logits = self.forward(image)
        loss = self.loss_fn(logits, label)

        # Metrics
        probs = logits.sigmoid()
        preds = (probs > 0.5).float()

        # IoU
        tp, fp, fn, tn = get_stats(preds.long(), label.long(), mode="binary")
        # Accuracy
        #acc = accuracy(tp, fp, fn, tn)
        #self.log("acc", acc)
        self.log("train_loss", loss, sync_dist=True)

        return {
            "loss": loss,
            #"acc": acc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    def validation_step(self, batch, batch_idx):
        image, label = batch
        bs = image.shape[0]
        h, w = image.shape[2:]

        assert image.ndim == 4
        assert image.shape == (bs, 3, h, w) # Multichannel dataset

        #label = batch["label"]
        assert label.ndim == 4
        assert label.shape == (bs, self.classes, h, w)

        assert label.max() <= 1 and label.min() >= 0

        logits = self.forward(image)
        loss = self.loss_fn(logits, label)

        # Metrics
        probs = logits.sigmoid()
        preds = (probs > 0.5).float()

        # IoU
        tp, fp, fn, tn = get_stats(preds.long(), label.long(), mode="binary")
        # Accuracy
        # acc = accuracy(tp, fp, fn, tn)
        # self.log("acc", acc)
        self.log("valid_loss", loss, sync_dist=True)

        return {
            "loss": loss,
            # "acc": acc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    def test_step(self, batch, batch_idx):
        image, label = batch
        bs = image.shape[0]
        h, w = image.shape[2:]

        assert image.ndim == 4
        assert image.shape == (bs, 3, h, w) # Multichannel dataset

        #label = batch["label"]
        assert label.ndim == 4
        assert label.shape == (bs, self.classes, h, w)

        assert label.max() <= 1 and label.min() >= 0

        logits = self.forward(image)
        loss = self.loss_fn(logits, label)

        # Metrics
        probs = logits.sigmoid()
        preds = (probs > 0.5).float()

        # IoU
        tp, fp, fn, tn = get_stats(preds.long(), label.long(), mode="binary")
        # Accuracy
        # acc = accuracy(tp, fp, fn, tn)
        # self.log("acc", acc)
        self.log("test_loss", loss, sync_dist=True)

        return {
            "loss": loss,
            # "acc": acc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)