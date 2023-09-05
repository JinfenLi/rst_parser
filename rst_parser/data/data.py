import os
from pathlib import Path
import pickle
from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ..utils.data import data_keys


class DataModule(LightningDataModule):

    def __init__(self,
                 dataset: str = None,
                 data_path: str = None, mode: str = None, num_classes: int = None,
                 train_batch_size: int = 1, eval_batch_size: int = 1, eff_train_batch_size: int = 1,
                 mini_batch_size: int = 1,
                 num_workers: int = 0,
                 num_train: int = None, num_dev: int = None, num_test: int = None,
                 num_train_seed: int = None, num_dev_seed: int = None, num_test_seed: int = None,
                 train_shuffle: bool = False

                 ):
        super().__init__()

        self.dataset = dataset
        self.data_path = data_path  # ${data_dir}/${.dataset}/${model.arch}/

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eff_train_batch_size = eff_train_batch_size
        self.mini_batch_size = mini_batch_size
        self.num_workers = num_workers

        self.num_samples = {'train': num_train, 'dev': num_dev, 'test': num_test}
        self.num_samples_seed = {'train': num_train_seed, 'dev': num_dev_seed, 'test': num_test_seed}

        self.train_shuffle = train_shuffle

    def load_dataset(self, split):
        dataset = {}
        data_path = os.path.join(self.data_path, split)
        assert Path(data_path).exists()

        for key in tqdm(data_keys, desc=f'Loading {split} set'):
            if self.num_samples[split] is not None:
                filename = f'{key}_{self.num_samples[split]}_{self.num_samples_seed[split]}.pkl'
            else:
                filename = f'{key}.pkl'

            with open(os.path.join(data_path, filename), 'rb') as f:
                dataset[key] = pickle.load(f)

        return dataset

    def setup(self, splits=['all'], stage=None, dataset=None):
        self.data = {}
        if dataset is None:
            splits = ['train', 'dev', 'test'] if splits == ['all'] else splits
            for split in splits:
                dataset = self.load_dataset(split)
                self.data[split] = TextClassificationDataset(dataset, split, self.mini_batch_size)
        else:
            self.data['pred'] = TextClassificationDataset(dataset, 'pred', self.mini_batch_size)



    def train_dataloader(self):

        return DataLoader(
            self.data['train'],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['train'].collater,
            shuffle=self.train_shuffle,
            pin_memory=True
        )

    def val_dataloader(self, test=False):
        if test:
            return DataLoader(
                self.data['dev'],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data['dev'].collater,
                pin_memory=True
            )
        return [
            DataLoader(
                self.data[eval_split],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data[eval_split].collater,
                pin_memory=True)

            for eval_split in ['dev', 'test']
        ]

    def test_dataloader(self):
        return DataLoader(
            self.data['test'],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['test'].collater,
            pin_memory=True
        )


    def predict_dataloader(self):
        return DataLoader(
            self.data['pred'],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['pred'].collater,
            pin_memory=True
        )


class EDUDataset(Dataset):
    def __init__(self, dataset, mini_batch_size):
        self.data = dataset
        self.mini_batch_size = mini_batch_size

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        # item_idx = torch.LongTensor([self.data['item_idx'][idx]])
        input_ids = self.data['input_ids'][idx]
        attention_mask = self.data['attention_mask'][idx]
        result_tuple = (input_ids, attention_mask)
        return result_tuple
    def collater(self, items):
        batch = {
            # 'item_idx': torch.cat([x[0] for x in items]),
            'input_ids': torch.stack([x[0] for x in items], dim=0),
            'attention_mask': torch.stack([x[1] for x in items], dim=0),
            # 'split': self.split,
            # 'mini_batch_size': self.mini_batch_size,
        }

        return batch

class TextClassificationDataset(Dataset):
    def __init__(self, dataset, split, mini_batch_size):
        self.data = dataset
        self.split = split
        self.mini_batch_size = mini_batch_size

    def __len__(self):
        return len(self.data['item_idx'])

    def __getitem__(self, idx):
        item_idx = torch.LongTensor([self.data['item_idx'][idx]])
        input_ids = torch.LongTensor(self.data['edu_input_ids'][idx])
        attention_mask = torch.LongTensor(self.data['edu_attention_masks'][idx])
        result_tuple = (item_idx, input_ids, attention_mask)

        if 'spans' in self.data:
            spans = torch.LongTensor([self.data['spans'][idx]])
            actions = torch.LongTensor([self.data['actions'][idx]])
            forms = torch.LongTensor([self.data['forms'][idx]])
            relations = torch.LongTensor([self.data['relations'][idx]])
            result_tuple += (spans, actions, forms, relations,)
        return result_tuple

    def collater(self, items):
        batch = {
            'item_idx': torch.cat([x[0] for x in items]),
            'input_ids': torch.stack([x[1] for x in items], dim=0),
            'attention_mask': torch.stack([x[2] for x in items], dim=0),
            'spans': torch.cat([x[3] for x in items]) if self.split != 'pred' else None,
            'actions': torch.cat([x[4] for x in items]) if self.split != 'pred' else None,
            'forms': torch.cat([x[5] for x in items]) if self.split != 'pred' else None,
            'relations': torch.cat([x[6] for x in items]) if self.split != 'pred' else None,
            'split': self.split,
            'mini_batch_size': self.mini_batch_size,
        }

        return batch
