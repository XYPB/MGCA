import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, 
                 num_workers, imsize=256, crop_size=224, structural_cap=False, 
                 simple_cap=False, natural_cap=False, pred_density=False,
                 ten_pct=False, instance_test_cap=False, zero_shot=False,
                 balanced_test=False, screen_only=False, aligned_mlo=False,
                 paired_test=False):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imsize = imsize
        self.crop_size = crop_size
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.pred_density = pred_density
        self.zero_shot = zero_shot
        self.ten_pct = ten_pct
        self.instance_test_cap = instance_test_cap
        self.balanced_test = balanced_test
        self.screen_only = screen_only
        self.aligned_mlo = aligned_mlo
        self.paired_test = paired_test

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        
        dataset = self.dataset(
            split="train", transform=transform, data_pct=self.data_pct,
            imsize=self.imsize,
            structural_cap = self.structural_cap,
            simple_cap = self.simple_cap,
            natural_cap = self.natural_cap,
            screen_only=self.screen_only,
            aligned_mlo=self.aligned_mlo)

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="valid", transform=transform, data_pct=self.data_pct,
            imsize=self.imsize,
            structural_cap = self.structural_cap,
            simple_cap = self.simple_cap,
            natural_cap = self.natural_cap,
            screen_only=self.screen_only,
            aligned_mlo=self.aligned_mlo,
            paired_test=self.paired_test)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="test", transform=transform, data_pct=self.data_pct,
            imsize=self.imsize,
            structural_cap = self.structural_cap,
            simple_cap = self.simple_cap,
            natural_cap = self.natural_cap,
            pred_density=self.pred_density,
            ten_pct=self.ten_pct,
            instance_test_cap=self.instance_test_cap,
            zero_shot=self.zero_shot,
            balanced_test=self.balanced_test,
            screen_only=self.screen_only,
            aligned_mlo=self.aligned_mlo,
            paired_test=self.paired_test)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )