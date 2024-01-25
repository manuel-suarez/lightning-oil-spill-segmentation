from typing import Any

import torch
import logging
from lightning import LightningModule
from segmentation_models_pytorch import create_model
from segmentation_models_pytorch.metrics import get_stats, accuracy, iou_score
from segmentation_models_pytorch.losses import JaccardLoss, SoftBCEWithLogitsLoss, DiceLoss, BINARY_MODE

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
        #self.loss_fn = DiceLoss(BINARY_MODE, from_logits=True)
        self.loss_fn = SoftBCEWithLogitsLoss()
        #self.loss_fn = JaccardLoss(BINARY_MODE, from_logits=True)

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, label = batch
        bs = image.shape[0]
        h, w = image.shape[2:]

        assert image.ndim == 4
        assert image.shape == (bs, 3, h, w) # Multichannel dataset

        #label = batch["label"]
        logging.info(f"label ndim: {label.ndim}, shape: {label.shape}")
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
        acc = accuracy(tp, fp, fn, tn, reduction="micro")
        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        self.log_dict({
            "train_loss": loss,
            "train_acc": acc,
            "train_iou": iou
        },
            sync_dist=True,
            prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        bs = image.shape[0]
        h, w = image.shape[2:]

        assert image.ndim == 4
        assert image.shape == (bs, 3, h, w) # Multichannel dataset

        #label = batch["label"]
        logging.info(f"label ndim: {label.ndim}, shape: {label.shape}")
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
        acc = accuracy(tp, fp, fn, tn, reduction="micro")
        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        self.log_dict({
            "valid_loss": loss,
            "valid_acc": acc,
            "valid_iou": iou
        },
            sync_dist=True,
            prog_bar=True
        )

        return loss

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
        acc = accuracy(tp, fp, fn, tn, reduction="micro")
        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
            "test_iou": iou
        },
            sync_dist=True,
            prog_bar=True
        )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        probs = self(batch.float()).sigmoid()
        preds = (probs > 0.5).float()

        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)