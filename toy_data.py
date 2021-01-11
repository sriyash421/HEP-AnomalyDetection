import os
import torch
import uproot
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pytorch_lightning as pl
from torch.utils.data import random_split
from utils import print_dict
from sklearn.preprocessing import StandardScaler


class ToyDatasetModule(pl.LightningDataModule):
    def __init__(self, root_path,
                 norm_array,
                 sig_sum,
                 bkg_sum,
                 bkg_list,
                 sig_list,
                 test_ratio,
                 val_ratio,
                 batch_size):
        super().__init__()
        self.root_path = root_path
        self.norm_array = norm_array
        self.sig_sum = sig_sum
        self.bkg_sum = bkg_sum
        self.bkg_list = bkg_list
        self.sig_list = sig_list
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size

    def prepare_data(self):
        '''
        return:
        sig : gaussian ((0.5,0.5),0.1*I) , 10000
        bkg : gaussian ((0,0),I) , 10000

        '''
        print("Reading Dataset...(toy)")
        sig_df = pd.read_csv("toy_sig.csv")
        bkg_df = pd.read_csv("toy_bkg.csv")
        self.sig = sig_df.to_numpy()
        self.bkg = bkg_df.to_numpy()
        print(f"Signal samples: {self.sig.shape}")
        print(f"Background samples: {self.bkg.shape}")
        self.input_size = 2

    def setup(self, stage):
        '''
        function to create tensordatasets by splitting according to ratio and samplers
        '''
        if self.norm_array:
            self.sig[:, :-1], self.bkg[:, :-1] = normalise_features(self.sig[:, :-1], self.bkg[:, :-1])

        np.random.shuffle(self.sig)
        np.random.shuffle(self.bkg)
        bkg_train, bkg_val, bkg_test = self.split_sets(
            self.bkg, self.val_ratio, self.test_ratio)
        _, _, sig_test = self.split_sets(self.sig)
        self.train = bkg_train
        self.val = bkg_val
        self.test = ConcatDataset([bkg_test, sig_test])
        print(
            f"Final sizes: train:{len(self.train)} val:{len(self.val)} test_size:{len(self.test)}")

    def split_sets(self, data, val_ratio=0, test_ratio=1):
        data = np.array(data, dtype=np.float32)

        features_tensor = torch.from_numpy(data[:, :-1])
        target = torch.from_numpy(data[:, -1])
        total_size = features_tensor.shape[0]
        val_size = int(total_size * val_ratio)
        test_size = int(total_size * test_ratio)
        train_size = total_size - val_size - test_size
        dataset = TensorDataset(features_tensor, target)
        return random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        train = DataLoader(
            self.train, batch_size=self.batch_size, num_workers=8, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val, batch_size=self.batch_size,
                         num_workers=8, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=8, shuffle=False)
        return test


def norweight(weight_array, norm=1000):
    print(f"Normalising the arrays")
    new = weight_array.copy()
    total_weight = np.sum(new)
    frac = norm / total_weight
    new = frac * new
    return new

def normalise_features(sig, bkg):
    print(f"Normalising the arrays")
    data = np.concatenate([sig, bkg], axis=0)
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(sig), scaler.transform(bkg)
