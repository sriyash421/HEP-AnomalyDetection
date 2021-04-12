import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from models import Classifier, AutoEncoder
import numpy as np
from utils import print_dict, get_distance_matrix
import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss, roc_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Model(pl.LightningModule):

    def __init__(self,
                 momentum,
                 nesterov,
                 learn_rate,
                 learn_rate_decay,
                 classifier_wt,
                 encoder_wt,
                 optimizer,
                 classifier_nodes,
                 encoder_nodes,
                 dropout,
                 activation,
                 input_size,
                 output_size,
                 save_tb_logs,
                 log_path,
                 K,
                 inf_batch_size
                 ):
        '''create a training class'''
        super(pl.LightningModule, self).__init__()
        self.classifier = Classifier(
            classifier_nodes, dropout, activation, input_size, output_size)
        self.encoder = AutoEncoder(
            encoder_nodes, dropout, activation, input_size+sum(classifier_nodes)+output_size)
        self.m = encoder_nodes[-1]
        self.example_input_array = torch.ones((1, input_size))
        self.momentum = momentum
        self.nesterov = nesterov
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.classifier_wt = classifier_wt
        self.encoder_wt = encoder_wt
        self.optimizer_ = optimizer
        self.encoder_loss_fn = torch.nn.MSELoss()
        self.classifier_loss_fn = torch.nn.CrossEntropyLoss()
        self.K = K
        self.inf_batch_size = inf_batch_size
        self.log_path = log_path
        self.num_bkg = output_size

    def forward(self, input):
        '''get output'''
        predictions, features = self.classifier(input)
        recon_features, latent_rep = self.encoder(features.detach())
        return predictions, features, recon_features, latent_rep

    def configure_optimizers(self):
        '''create optimizer and scheduler'''
        optimizer = None
        if self.optimizer_ == 'adam':
            optimizer = torch.optim.Adam(self.parameters(
            ), lr=self.learn_rate, betas=[self.momentum, 0.999])
        else:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learn_rate, momentum=self.momentum, nesterov=self.nesterov)

        def scheduler_fn(epoch): return 1./(1+epoch*self.learn_rate_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
        return [optimizer], [scheduler]

    def step_helper(self, batch):
        inputs, targets = batch
        predictions, features, recon_features, latent_rep = self(inputs)
        classifier_loss = self.classifier_loss_fn(
            predictions, (targets-1).long())
        recon_loss = self.encoder_loss_fn(features, recon_features)
        total_loss = self.classifier_wt*classifier_loss+self.encoder_wt*recon_loss
        accuracy = (torch.argmax(predictions, dim=1)
                    == (targets-1)).float().mean()
        return total_loss, classifier_loss, recon_loss, accuracy, latent_rep

    def training_step(self, batch, batch_idx):
        '''executed during training'''
        total_loss, classifier_loss, recon_loss, accuracy, _ = self.step_helper(
            batch)
        self.log('train_loss', total_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', accuracy, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_classifier_loss', classifier_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_recon_loss', recon_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': total_loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        '''executed during validation'''
        total_loss, classifier_loss, recon_loss, accuracy, _ = self.step_helper(
            batch)
        self.log('val_loss', total_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', accuracy, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_classifier_loss', classifier_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recon_loss', recon_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': total_loss, 'val_acc': accuracy}

    def on_test_epoch_start(self):
        self.eval()
        training_samples = []
        for _, batch in enumerate(self.train_dataloader()):
            _, _, _, features = self(batch[0].to(self.device))
            training_samples.append(features)
        for _, batch in enumerate(self.val_dataloader()):
            _, _, _, features = self(batch[0].to(self.device))
            training_samples.append(features)
        self.training_features = torch.cat(training_samples, axis=0)
        self.test_features = []
        self.test_target = []

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions, features, recon_features, latent_rep = self(inputs)
        self.test_features.append(latent_rep)
        self.test_target.append(batch[1])

    def test_epoch_end(self, outputs):
        self.test_features = torch.cat(self.test_features, axis=0)
        self.test_target = torch.cat(self.test_target, axis=0)
        self.analysis()
        self.anomaly_detection()
    
    def analysis(self):
        features = TSNE(n_components=2).fit_transform(self.test_features.cpu().numpy())
        target = self.test_target.cpu().numpy()
        mc_sig_index = np.where(target == 0)
        mc_bkg_index = [np.where(target == i+1) for i in range(self.num_bkg)]
        plt.figure()
        plt.scatter(features[mc_sig_index,0],features[mc_sig_index,1], label="Signal")
        for i in range(self.num_bkg):
            plt.scatter(features[mc_bkg_index[i],0],features[mc_bkg_index[i],1], label=f"Background_{i+1}")
        plt.title("Hidden features")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.log_path}/report_features.png")
        plt.clf()
        

    def get_mean_dist(self, x, samples):
        x = samples-x
        x = torch.norm(x, dim=1)
        return torch.sort(x)[0][-self.K:].mean()

    def plot(self, scores, target, name):
        mc_sig_index = np.where(target == 1)
        mc_bkg_index = np.where(target == 0)
        false_pos_rate, true_pos_rate, _ = roc_curve(target, scores)
        plt.subplot(2, 1, 1)
        plt.title("score distribution")
        plt.hist(scores[mc_sig_index], bins=40, label="Signal",
                 range=[0, 1], histtype=u"step")
        plt.hist(scores[mc_bkg_index], bins=40,
                 label="Background", range=[0, 1], histtype=u"step")
        plt.yscale("log")
        plt.ylabel("#events")
        plt.xlabel("prediction score")
        plt.legend(loc="best")
        # plotting ROC curve
        plt.subplot(2, 1, 2)
        plt.title("ROC curve")
        plt.plot(false_pos_rate, true_pos_rate)
        plt.xlabel("False Positive Rate"), plt.ylabel("True Positive Rate")
        plt.text(0.8, 0.2, f"AUC = {roc_auc_score(target, scores)}", bbox=dict(
            facecolor="none", edgecolor="black", boxstyle="square"))
        plt.tight_layout()
        plt.savefig(f"{self.log_path}/report_{name}.png")
        plt.clf()

    def anomaly_detection(self):
        training_dists = get_distance_matrix( self.inf_batch_size,
            self.training_features, self.training_features, self.K, self.device)
        train_mean, train_std = torch.mean(
            training_dists), torch.std(training_dists)
        d_train_samples = get_distance_matrix( self.inf_batch_size,
            self.test_features, self.training_features, self.K, self.device)
        d_test_samples = get_distance_matrix( self.inf_batch_size,
            self.test_features, self.test_features, self.K, self.device)
        delta_trad = torch.tensor(
            [(d_train-train_mean)/(train_std+1e-8) for d_train in d_train_samples])
        delta_new = torch.tensor([(d_test**(-self.m)-d_train**(-self.m))/(
            d_train**(int(-self.m/2))+1e-8) for d_test, d_train in zip(d_test_samples, d_train_samples)])
        rms_trad, rms_new = (delta_trad**2).mean(), (delta_new**2).mean()
        scores_trad = 0.5*(1+torch.erf(delta_trad*(1.0/(rms_trad*(2**0.5)))))
        scores_new = 0.5*(1+torch.erf(delta_new*(1.0/(rms_new*(2**0.5)))))
        scores_comb = torch.mul(scores_trad, scores_new)**0.5

        targets = (self.test_target == 0).float().cpu()

        anomaly_comb = ((scores_comb.cpu() >= 0.5) ==
                       targets).float().mean()
        anomaly_trad = ((scores_trad.cpu() >= 0.5) ==
                        targets).float().mean()

        stats = {"Anomaly_new Acc": anomaly_trad, "Anomaly_trad Acc": anomaly_trad}
        self.plot(scores_trad.numpy(), targets.numpy(), "Trad")
        self.plot(scores_new.numpy(), targets.numpy(), "New")
        self.plot(scores_comb.numpy(), targets.numpy(), "Comb")
        temp_df = pd.DataFrame()
        temp_df["targets"] = targets
        temp_df["new_scores"] = scores_new
        temp_df["trad_scores"] = scores_trad
        temp_df["comb_scores"] = scores_comb
        temp_df.to_csv(f"{self.log_path}/score_table.csv")
        print_dict(stats)
