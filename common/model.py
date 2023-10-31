"""
A ResNet-18
"""
from torchvision import models
import lightning.pytorch as pl
from torch import optim, nn
import torch
import wandb

class ResNet18(pl.LightningModule):
    def __init__(self, in_channels, num_classes, lr=1e-3, weight_decay=1e-05):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = models.resnet18()

        # replace first conv layer
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # replace last classificaiton layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), self.lr,
                           weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        labels = labels.squeeze()
        preds = self.model(imgs.float())
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.squeeze()
        logits = self.model(imgs.float())
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        self.log("val_loss", loss)

        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            preds=preds.cpu().detach().numpy(), y_true=labels.cpu().detach().numpy(), class_names = ["one","two","three","four","five", "six", "seven", "eight"])})

        return {'loss': loss, 'preds': preds, 'target': labels}

    #def on_validation_epoch_end(self, outputs):
    #    preds = torch.cat([tmp['preds'] for tmp in outputs])
    #    targets = torch.cat([tmp['target'] for tmp in outputs])
    #    confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=10)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs.float())
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("test_acc", acc)
        self.log("test_loss", loss)
