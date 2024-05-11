import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torcheval.metrics import AUC, MulticlassAccuracy, MulticlassConfusionMatrix
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                             EmbedPretrainingDataset,
                                             multimodal_collate_fn)
from mgca.datasets.rsna_mammo import RSNAMammo
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder, DinoEncoder
from torch import distributed as dist
from torch import nn

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ConVIRT(LightningModule):
    ''' PyTorch Lightning implementation of ConVIRT
        https://arxiv.org/pdf/2010.00747.pdf
    '''

    def __init__(self,
                 img_encoder: str = "resnet_50",
                 hidden_mlp: int = 2048,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 freeze_bert: bool = False,
                 momentum: float = 0.9,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 1e-4,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.img_encoder = img_encoder
        self.freeze_bert = freeze_bert

        self.zero_shot_text_feats = None
        self.all_scores = None
        self.all_labels = None
        if self.hparams.embed:
            num_classes = 4 if self.hparams.pred_density else 7
        elif self.hparams.rsna_mammo:
            num_classes = 2
        self.confmat = MulticlassConfusionMatrix(num_classes)

        self.init_encoder()

    def init_encoder(self):
        if "dino" in self.img_encoder:
            self.img_encoder = DinoEncoder(
                model_name=self.img_encoder, image_size=self.hparams.crop_size, 
                output_dim=self.hparams.emb_dim, vit_grad_ckpt=self.hparams.vit_grad_ckpt,
                )
        else:
            self.img_encoder = ImageEncoder(
                model_name=self.img_encoder, image_size=self.hparams.crop_size,
                output_dim=self.hparams.emb_dim, vit_grad_ckpt=self.hparams.vit_grad_ckpt,
                vit_ckpt_layer=self.hparams.vit_ckpt_layer)
        self.text_encoder = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=self.freeze_bert)

    def zero_shot_inference(self, batch, batch_idx):
        '''Inference with zero shot setting'''

        # Forward of query image encoder
        img_feat, _ = self.img_encoder(batch["imgs"])
        # Use classification token instead of averaged patch tokens
        img_emb = self.img_encoder.global_embed(img_feat)
        img_emb = F.normalize(img_emb, dim=-1)

        # Forward of query text encoder
        # N x CLS x S
        bsz = img_emb.size(0) # N x C
        batch_scores = []
        fixed_caption_ids = batch["caption_ids"][0] # 14 x S, get rid of batch dim
        fixed_attention_mask = batch["attention_mask"][0]
        fixed_token_type_ids = batch["token_type_ids"][0]
        for idx in range(bsz):
            if self.zero_shot_text_feats is None:
                sent_feat, _, _, _ = self.text_encoder(
                    fixed_caption_ids, fixed_attention_mask, fixed_token_type_ids)
                sent_emb = self.text_encoder.global_embed(sent_feat)
                sent_emb = F.normalize(sent_emb, dim=-1)
                self.zero_shot_text_feats = sent_emb
            scores = img_emb[idx:idx+1].mm(self.zero_shot_text_feats.t()) # 1 x CLS
            scores /= self.hparams.softmax_temperature
            batch_scores.append(scores.squeeze(0))
        scores = torch.stack(batch_scores, dim=0) # N x CLS

        ########### image-text zero-shot cls loss ################
        labels = torch.tensor(batch["path"]).type_as(self.zero_shot_text_feats) # N x CLS

        # Image to text classification loss
        loss0 = F.cross_entropy(scores, labels.argmax(dim=-1))

        all_scores = scores
        all_labels = labels
        self.confmat.update(
            torch.argmax(all_scores, dim=-1), all_labels.argmax(dim=-1))
        all_scores = all_scores.detach().to(torch.float32)
        all_scores = torch.softmax(all_scores, dim=-1).cpu().numpy()
        all_labels = all_labels.detach().to(torch.float32).cpu().numpy()
        if self.all_scores is None:
            self.all_scores = all_scores
        else:
            self.all_scores = np.concatenate([self.all_scores, all_scores], axis=0)
        if self.all_labels is None:
            self.all_labels = all_labels
        else:
            self.all_labels = np.concatenate([self.all_labels, all_labels], axis=0)

        # compute retrieval accuracy
        i2t_acc1 = self.precision_at_k(scores, labels.argmax(dim=-1), top_k=(1,))[0]

        return loss0, i2t_acc1, 0.


    def forward(self, batch):
        img_feat, _ = self.img_encoder(batch["imgs"])
        img_emb = self.img_encoder.global_embed(img_feat)
        img_emb = F.normalize(img_emb, dim=1)

        sent_feat, _, _, _ = self.text_encoder(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        sent_emb = self.text_encoder.global_embed(sent_feat)
        sent_emb = F.normalize(sent_emb, dim=1)

        return img_emb, sent_emb

    def info_nce_loss(self, out_1, out_2, temperature):
        bz = out_1.size(0)
        labels = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)

        return loss0 + loss1

    def shared_step(self, batch):
        img_emb, sent_emb = self(batch)
        loss = self.info_nce_loss(
            img_emb, sent_emb, self.hparams.softmax_temperature)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, prog_bar=True,
                 on_epoch=False, sync_dist=True, batch_size=self.hparams.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, prog_bar=True,
                 on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        loss_ita, acc1, acc5 = self.zero_shot_inference(
            batch, batch_idx)

        loss = self.hparams.lambda_1 * loss_ita

        log = {
            "val_loss": loss,
            "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_local": 0.,
            "val_loss_proto": 0.,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):

        # Calculate the confusion matrix using the accumulated predictions and targets
        conf_matrix = self.confmat.compute().cpu().numpy()
        print("### Confusion Matrix:\n", conf_matrix)
        if self.hparams.rsna_mammo:
            tn = conf_matrix[0, 0]
            tp = conf_matrix[1, 1]
            fn = conf_matrix[1, 0]
            fp = conf_matrix[0, 1]
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            print("\n### Sensitivity: {:.4f}".format(sensitivity))
            print("### Specificity: {:.4f}".format(specificity))
            print("### PPV: {:.4f}".format(ppv))
            print("### NPV: {:.4f}".format(npv))
        cls_cnt = np.sum(conf_matrix, axis=1)
        cls_hit = np.diag(conf_matrix)
        cls_acc = cls_hit / cls_cnt
        print("\n### Class Accuracy: ", [f"{100 * acc:.4f}" for acc in cls_acc])

        # Calculate the accuracy using the accumulated predictions and targets
        acc = 100 * accuracy_score(np.argmax(self.all_labels, -1), np.argmax(self.all_scores, -1))
        f1 = 100 * f1_score(np.argmax(self.all_labels, -1), np.argmax(self.all_scores, -1), average='macro')
        ba = 100 * balanced_accuracy_score(np.argmax(self.all_labels, -1), np.argmax(self.all_scores, -1))
        try:
            if len(np.unique(self.all_labels)) > 2:
                auc = 100 * roc_auc_score(np.argmax(self.all_labels, -1), 
                                          self.all_scores, multi_class="ovr")
            else:
                auc = 100 * roc_auc_score(self.all_labels, self.all_scores)
        except Exception as e:
            print("### Warning: AUC calculation failed with error:", e)
            auc = 0
        print("### Accuracy: {:.4f}".format(acc))
        print("### AUC: {:.4f}".format(auc))
        print("### F1: {:.4f}".format(f1))
        print("### Balanced Accuracy: {:.4f}".format(ba))

        # Reset metrics for the next test run
        self.confmat.reset()
        self.all_scores = None
        self.all_labels = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--embed", action="store_true")
        parser.add_argument("--rsna_mammo", action="store_true")
        parser.add_argument("--structural_cap", action="store_true")
        parser.add_argument("--simple_cap", action="store_true")
        parser.add_argument("--natural_cap", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--num_negatives", type=int, default=65536)
        parser.add_argument("--encoder_momentum", type=float, default=0.999)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_pct", type=float, default=1.)

        # Training args
        parser.add_argument("--max_epochs", type=int, default=50) # Unused
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--gpus", type=int, default=2)
        parser.add_argument("--strategy", type=str, default="ddp")
        parser.add_argument("--accelerator", type=str, default='gpu')
        parser.add_argument("--precision", type=str, default="32")
        parser.add_argument("--dev", action="store_true")
        parser.add_argument("--crop_size", type=int, default=224)
        parser.add_argument("--imsize", type=int, default=256)
        parser.add_argument("--vit_grad_ckpt", action="store_true")
        parser.add_argument("--vit_ckpt_layer", type=int, default=0)

        # Test args
        parser.add_argument("--pretrained_model", type=str, default=None)
        parser.add_argument("--eval", action="store_true", help="Run evaluation")
        parser.add_argument("--pred_density", action="store_true")
        parser.add_argument("--ten_pct", action="store_true")
        parser.add_argument("--instance_test_cap", action="store_true")
        parser.add_argument("--balanced_test", action="store_true")
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, DDPStrategy)
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


def cli_main():
    parser = ArgumentParser()
    # model args
    parser = ConVIRT.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dev:
        args.gpus = 1
    args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.embed:
        dataset_obj = EmbedPretrainingDataset
    elif args.rsna_mammo:
        dataset_obj = RSNAMammo
    else:
        dataset_obj = MultimodalPretrainingDataset
    # define datamodule
    datamodule = DataModule(dataset_obj, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers,
                            structural_cap = args.structural_cap,
                            simple_cap = args.simple_cap,
                            natural_cap = args.natural_cap,
                            pred_density=args.pred_density,
                            ten_pct=args.ten_pct, zero_shot=args.eval,
                            instance_test_cap=args.instance_test_cap,
                            balanced_test=args.balanced_test,
                            crop_size=args.crop_size,
                            imsize=args.imsize)

    # Add load from checkpoint
    if args.pretrained_model is None:
        model = ConVIRT(**args.__dict__)
    else:
        print(f"\n\n##### Loading pretrained model from {args.pretrained_model}\n\n")
        model = ConVIRT.load_from_checkpoint(args.pretrained_model, map_location="cpu", strict=False, **args.__dict__)

    if args.eval:
        args.gpus = 1
        model.eval()
        # Single GPU inference
        trainer = Trainer(
            accelerator=args.accelerator,
            precision=args.precision,
            devices=1,
            fast_dev_run=args.dev,
            max_epochs=1,
            deterministic=args.deterministic,
            inference_mode=True
        )
        trainer.test(model, datamodule=datamodule)
    else:
        torch.set_float32_matmul_precision('high')
        # get current time
        now = datetime.datetime.now(tz.tzlocal())
        extension = now.strftime("%Y_%m_%d_%H_%M_%S")
        ckpt_dir = os.path.join(
            BASE_DIR, f"../../../logs/ckpts/ConVIRT/{extension}")
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                            save_last=True, mode="min", save_top_k=1),
            EarlyStopping(monitor="val_loss", min_delta=0.,
                        patience=5, verbose=False, mode="min")
        ]
        logger_dir = os.path.join(
            BASE_DIR, f"../../../logs")
        os.makedirs(logger_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="ConVIRT", save_dir=logger_dir, name=extension)
        trainer = Trainer(
            accelerator=args.accelerator,
            strategy=args.strategy,
            devices=args.gpus,
            precision=args.precision,
            callbacks=callbacks,
            logger=wandb_logger,
            fast_dev_run=args.dev,
            max_epochs=args.max_epochs)

        model.training_steps = model.num_training_steps(trainer, datamodule)
        print(model.training_steps)
        trainer.fit(model, datamodule=datamodule)

        best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
        callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
