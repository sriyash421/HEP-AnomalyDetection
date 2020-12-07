import os
import torch
import uproot
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pytorch_lightning as pl
from torch.utils.data import random_split
from utils import print_dict


class DatasetModule(pl.LightningDataModule):
    def __init__(self, root_path,
                 campaigns,
                 channel,
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
        self.campaigns = campaigns
        self.channel = channel
        self.norm_array = norm_array
        self.sig_sum = sig_sum
        self.bkg_sum = bkg_sum
        self.bkg_list = bkg_list
        self.sig_list = sig_list
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size

    # def read_data(self, class_list, campaign, df, target=None):
    #     labels = pd.DataFrame()
    #     for class_type in class_list:
    #         events = uproot.open(
    #             f"{self.root_path}/merged/{campaign}/{class_type}.root")
    #         tree = events[events.keys()[0]]
    #         features = tree.keys()
    #         tree_pd = tree.pandas.df(features)
    #         df = pd.concat([df, tree_pd], ignore_index=True)
    #         if target is not None:
    #             labels = pd.concat([labels, pd.DataFrame(
    #                 {"target": target*np.ones(len(tree_pd))})], ignore_index=True)
    #     return df, labels

    def prepare_data(self):
        '''
        return:
            sig_df: pandas of all mc signals [take all features]
            bkg_df: pandas of all mc backgrounds [take all features]
            sig_zeros: a pandas df with "target" columns, containing total number of sig of ones
            bkg_labels: a pandas df with "target" columns, containing labels of bkg
            E.g say our bkg_list = [diboson, wjets, top], then all diboson have target 1, wjets have 2 and top has 3
            as target

            Note: instead of 0/1 we have 0 for sig and 1,2... for bkg
        '''
        print("Reading Dataset...")
        sig_df = pd.DataFrame()
        bkg_df = pd.DataFrame()

        for campaign in self.campaigns:
            print(f"Reading campaign: {campaign}...")
            for i,bkg in enumerate(self.bkg_list):
                events = uproot.open(
                    f"{self.root_path}/merged/{campaign}/{bkg}.root"
                )
                tree = events[events.keys()[0]]
                features = tree.keys()
                tree_pd = tree.pandas.df(self.selected_features)
                tree_labels = pd.DataFrame({"target": np.ones(len(tree_pd) * (i+1))})
                tree_pd = pd.concat([tree_pd, tree_labels], axis=1)
                bkg_df = pd.concat([bkg_df, tree_pd], ignore_index=True)

            for j, sig in enumerate(self.sig_list):
                events = uproot.open(
                    f"{self.root_path}/merged/{campaign}/{sig}.root"
                )
                tree = events[events.keys()[0]]
                features = tree.keys()
                tree_pd = tree.pandas.df(self.selected_features)
                tree_labels = pd.DataFrame({"target": np.zeros(len(tree_pd))})
                tree_pd = pd.concat([tree_pd, tree_labels], axis=1)
                sig_df = pd.concat([sig_df, tree_pd], ignore_index=True)
            return sig_df, bkg_df, sig_df["target"].to_frame(), bkg_df["target"].to_frame()




    def setup(self, stage):
        '''
        function to create tensordatasets by splitting according to ratio and samplers
        '''
        if self.norm_array:
            self.sig[:, :-1] = norweight(self.sig[:, :-1], self.sig_sum)
            self.bkg[:, :-1] = norweight(self.bkg[:, :-1], self.bkg_sum)
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
