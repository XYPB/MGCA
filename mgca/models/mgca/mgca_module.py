import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torcheval.metrics import AUC, MulticlassAccuracy, MulticlassConfusionMatrix
from dateutil import tz
from einops import rearrange
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

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MGCA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        if "dino" in img_encoder:
            self.img_encoder_q = DinoEncoder(
                model_name=img_encoder, image_size=self.hparams.crop_size, 
                output_dim=self.hparams.emb_dim, vit_grad_ckpt=self.hparams.vit_grad_ckpt,
                )
        else:
            self.img_encoder_q = ImageEncoder(
                model_name=img_encoder, image_size=self.hparams.crop_size, 
                output_dim=self.hparams.emb_dim, vit_grad_ckpt=self.hparams.vit_grad_ckpt,
                vit_ckpt_layer=self.hparams.vit_ckpt_layer)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)

        self.zero_shot_text_feats = None
        if self.hparams.embed:
            num_classes = 4 if self.hparams.pred_density else 7
            if self.hparams.screen_only:
                num_classes = 4 if self.hparams.pred_density else 3
        elif self.hparams.rsna_mammo:
            num_classes = 2
        self.confmat = MulticlassConfusionMatrix(num_classes)
        self.all_scores = None
        self.all_labels = None
        self.all_paths = []

        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
        if self.hparams.gpus > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

    def zero_shot_inference(self, batch, batch_idx):
        '''Inference with zero shot setting'''

        # Forward of query image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(batch["imgs"])
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        # N x CLS x S
        bsz = img_emb_q.size(0) # N x C
        batch_scores = []
        fixed_caption_ids = batch["caption_ids"][0] # 14 x S, get rid of batch dim
        fixed_attention_mask = batch["attention_mask"][0]
        fixed_token_type_ids = batch["token_type_ids"][0]
        for idx in range(bsz):
            if self.zero_shot_text_feats is None:
                report_feat_q_full, word_feat_q, word_attn_q, sents = self.text_encoder_q(
                    fixed_caption_ids, fixed_attention_mask, fixed_token_type_ids)
                report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
                report_emb_q = F.normalize(report_emb_q, dim=-1)
                self.zero_shot_text_feats = report_emb_q
            scores = img_emb_q[idx:idx+1].mm(self.zero_shot_text_feats.t()) # 1 x CLS
            scores /= self.hparams.softmax_temperature
            batch_scores.append(scores.squeeze(0))
        scores = torch.stack(batch_scores, dim=0) # N x CLS

        ########### image-text zero-shot cls loss ################
        labels = torch.tensor(batch["label"]).type_as(self.zero_shot_text_feats) # N x CLS

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
        for path in batch["path"]:
            self.all_paths.append(path)

        # compute retrieval accuracy
        i2t_acc1 = self.precision_at_k(scores, labels.argmax(dim=-1), top_k=(1,))[0]

        return loss0, i2t_acc1, 0.

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(
            batch["imgs"])
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_ita = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        ########### Token-level alignment ################
        # cross attention patch to sentences
        mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
            batch["imgs"]).bool()

        if self.hparams.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                word_emb_q, patch_emb_q, patch_emb_q)
        else:
            atten_sim = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
            word_num = word_emb_q.size(1)
            # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
            atten_scores = F.softmax(
                atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
            word_atten_output = torch.bmm(atten_scores, patch_emb_q)

        word_atten_output = F.normalize(word_atten_output, dim=-1)

        word_sim = torch.bmm(
            word_emb_q, word_atten_output.permute(0, 2, 1)) / self.hparams.local_temperature

        with torch.no_grad():
            atten_weights = word_attn_q.detach()
            word_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)
            # TODO: maybe clip the tensor of 10 percentile and 90 percentile

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)

        word_sim = torch.bmm(word_emb_q, word_atten_output.permute(
            0, 2, 1)) / self.hparams.local_temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(
            word_emb_q).long().repeat(bz)
        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.

        if self.hparams.bidirectional:
            # Try not use atten layer
            if self.hparams.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)
                atten_sim[mask.unsqueeze(1).repeat(
                    1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(
                    atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
                patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            # patch_atten_output: bz, 196, 128
            if "vit" not in self.hparams.img_encoder or "dino" in self.hparams.img_encoder:
                patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(
                    bz, patch_num).type_as(batch["imgs"]) / patch_num

            else:
                with torch.no_grad():
                    img_attn_map = self.img_encoder_q.model.blocks[-1].attn.attention_map.detach(
                    )
                    atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
                    patch_atten_weights = []
                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        low = torch.quantile(atten_weight, 0.1)
                        high = torch.quantile(atten_weight, 0.9)
                        atten_weight = atten_weight.clip(low, high)
                        patch_atten_weights.append(atten_weight.clone())
                    patch_atten_weights = torch.stack(patch_atten_weights)

                patch_atten_weights /= patch_atten_weights.sum(
                    dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.hparams.local_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word

        else:

            loss_local = loss_word

        # normalize prototype layer
        with torch.no_grad():
            w = self.prototype_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototype_layer.weight.copy_(w)

        # Compute assign code of images
        img_proto_out = self.prototype_layer(img_emb_q)
        report_proto_out = self.prototype_layer(report_emb_q)

        # TODO: define this to hparams
        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.hparams.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.hparams.sinkhorn_iterations)         # bz, 500
            report_code = torch.exp(
                report_proto_out / self.hparams.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.hparams.sinkhorn_iterations)       # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.hparams.proto_temperature, dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.hparams.proto_temperature, dim=1)

        loss_i2t_proto = - \
            torch.mean(
                torch.sum(img_code * torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = - \
            torch.mean(torch.sum(report_code *
                       torch.log(img_proto_prob), dim=1))

        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.

        return loss_ita, loss_local, loss_proto, acc1, acc5

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.hparams.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.hparams.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def training_step(self, batch, batch_idx):
        loss_ita, loss_local, loss_proto, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto

        log = {
            "train_loss": loss,
            "train_loss_ita": self.hparams.lambda_1 * loss_ita,
            "train_loss_local": self.hparams.lambda_2 * loss_local,
            "train_loss_proto": self.hparams.lambda_3 * loss_proto,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    # freeze prototype layer
    def on_after_backward(self):
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            for param in self.prototype_layer.parameters():
                param.grad = None

    def validation_step(self, batch, batch_idx):
        loss_ita, loss_local, loss_proto, acc1, acc5 = self(
            batch, batch_idx, "valid")

        loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto

        log = {
            "val_loss": loss,
            "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_local": self.hparams.lambda_2 * loss_local,
            "val_loss_proto": self.hparams.lambda_3 * loss_proto,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
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
            print("\n### Sensitivity: {:.4f}".format(100*sensitivity))
            print("### Specificity: {:.4f}".format(100*specificity))
            print("### PPV: {:.4f}".format(100*ppv))
            print("### NPV: {:.4f}".format(100*npv))
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
        print("### Balanced Accuracy: {:.4f}".format(ba))
        print("### AUC: {:.4f}".format(auc))
        print("### F1: {:.4f}".format(f1))

        # Reset metrics for the next test run
        self.confmat.reset()
        self.all_scores = None
        self.all_labels = None

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

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
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--num_prototypes", type=int, default=500)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--screen_only", action="store_true")
        parser.add_argument("--aligned_mlo", action="store_true")

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
        parser.add_argument("--resume",  type=str, default=None)

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


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()
    # model args
    parser = MGCA.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dev:
        args.gpus = 1
    args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)
    if args.embed:
        dataset_obj = EmbedPretrainingDataset
    elif args.rsna_mammo:
        dataset_obj = RSNAMammo
    else:
        dataset_obj = MultimodalPretrainingDataset

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
                            imsize=args.imsize,
                            screen_only=args.screen_only,
                            aligned_mlo=args.aligned_mlo)

    # Add load from checkpoint
    if args.pretrained_model is None:
        model = MGCA(**args.__dict__)
    else:
        print(f"\n\n##### Loading pretrained model from {args.pretrained_model}\n\n")
        model = MGCA.load_from_checkpoint(args.pretrained_model, map_location="cpu", strict=False, **args.__dict__)

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
            BASE_DIR, f"../../../logs/ckpts/MGCA/{extension}")
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
            project="MGCA", save_dir=logger_dir, name=extension)
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
        print(f"\n### Resume from {args.resume}...\n")
        trainer.fit(model, datamodule=datamodule,
                    ckpt_path=args.resume,)

        best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
        callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
