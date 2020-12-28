import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from models import Classifier, AutoEncoder
import numpy as np
from utils import print_dict

class Model(pl.LightningModule):

    def __init__(self,
                 momentum,
                 nesterov,
                 learn_rate,
                 learn_rate_decay,
                 sig_class_weight,
                 bkg_class_weight,
                 optimizer,
                 classifier_nodes,
                 encoder_nodes,
                 dropout,
                 activation,
                 input_size,
                 output_size,
                 save_tb_logs,
                 log_path,
                 K
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
        self.sig_class_weight = sig_class_weight
        self.bkg_class_weight = bkg_class_weight
        self.optimizer_ = optimizer
        self.encoder_loss_fn = torch.nn.MSELoss()
        self.classifier_loss_fn = torch.nn.NLLLoss()
        self.K = K
        self.metrics = {
            "train_history_acc": [],
            "train_history_loss": [],
            "val_history_acc": [],
            "val_history_loss": [],
        }
        self.log_path = log_path
        # self.save_tb_logs = save_tb_logs

    def forward(self, input):
        '''get output'''
        print("input",input.shape)
        predictions, features = self.classifier(input)
        print("predictions", predictions.shape)
        print("features", features.shape)
        recon_features, latent_rep = self.encoder(features.detach())
        print("recon_features",recon_features.shape)
        print("latent_rep",latent_rep.shape)
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
        classifier_loss = self.classifier_loss_fn(preds, targets-1)
        recon_loss = self.encoder_loss_fn(features, recon_features)
        total_loss = classifier_loss+recon_loss
        accuracy = (torch.argmax(predictions, dim=1)
                    == (targets-1)).float.mean()
        return total_loss, classifier_loss, recon_loss, accuracy

    def training_step(self, batch, batch_idx):
        '''executed during training'''
        total_loss, classifier_loss, recon_loss, accuracy = self.step_helper(
            batch)
        log = dict(
            train_total_loss=total_loss,
            train_classifier_loss=classifier_loss,
            train_recon_loss=recon_loss,
            train_accuracy=accuracy
        )
        return {'loss': total_loss, 'acc': accuracy, 'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        '''executed during validation'''
        total_loss, classifier_loss, recon_loss, accuracy = self.step_helper(
            batch)
        log = dict(
            val_total_loss=total_loss,
            val_classifier_loss=classifier_loss,
            val_recon_loss=recon_loss,
            val_accuracy=accuracy
        )
        return {'val_loss': val_loss, 'val_acc': val_acc, 'log': log, 'progress_bar': log}

    def training_epoch_end(self, outputs):
        '''log metrics across train epoch'''
        avg_train_loss = torch.stack([output['loss']
                                      for output in outputs]).mean()
        avg_train_acc = torch.stack([output['acc']
                                     for output in outputs]).mean()
        train_metrics = {'average_train_loss': avg_train_loss,
                         'average_train_acc': avg_train_acc}
        return {
            'log': train_metrics
        }

    def validation_epoch_end(self, outputs):
        '''log metrics across val epoch'''
        avg_val_loss = torch.stack([output['val_loss']
                                    for output in outputs]).mean()
        avg_val_acc = torch.stack([output['val_acc']
                                   for output in outputs]).mean()
        val_metrics = {'average_val_loss': avg_val_loss,
                       'average_val_acc': avg_val_acc}
        return {
            'log': val_metrics
        }

    def on_test_epoch_start():
        self.eval()
        self.training_samples = []
        for _, batch in enumerate(self.train_dataloader):
            _, _, _, features = self(batch[0])
            self.training_samples.append(features)
        for _, batch in enumerate(self.val_dataloader):
            _, _, _, features = self(batch[0])
            self.training_samples.append(features)
        self.training_features = torch.cat(self.training_samples, axis=0)
        self.test_features = []
        self.test_target = []

    def testing_step(self, batch, batch_idx):
        _, _, _, features = self(batch[0])
        self.test_features.append(features)
        self.test_target.append(features)
        total_loss, classifier_loss, recon_loss, accuracy = self.step_helper(
            batch)
        log = dict(
            test_total_loss=total_loss,
            test_classifier_loss=classifier_loss,
            test_recon_loss=recon_loss,
            test_accuracy=accuracy
        )
        return {'test_loss': test_loss,
                'test_acc': val_acc,
                'test_class_loss': classifier_loss,
                'test_recon_loss': recon_loss,
                'progress_bar': log}

    def on_test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([output['test_loss']
                                     for output in outputs]).mean()
        avg_test_acc = torch.stack([output['test_acc']
                                    for output in outputs]).mean()
        avg_test_closs = torch.stack([output['test_class_loss']
                                    for output in outputs]).mean()
        avg_test_rloss = torch.stack([output['test_reco_loss']
                                    for output in outputs]).mean()
        test_metrics = {'average_test_loss': avg_test_loss,
                        'average_test_acc': avg_test_acc,
                        'avg_test_closs': avg_test_closs,
                        'avg_test_rloss': avg_test_rloss}
        self.test_features = torch.cat(self.test_features, axis=0)
        self.test_target = torch.cat(self.test_target, axis=0)
        self.anomaly_detection(test_metrics)
        return {
            'log': test_metrics
        }

    def get_mean_dist(self, x, samples):
        x = samples-x
        x = torch.linalg.norm(x, dim=1)
        return torch.sort(x)[-self.K:].mean()

    def plot(self, scores, target, name):
        mc_sig_index = target == 1
        mc_bkg_index = target == 0
        false_pos_rate, true_pos_rate, _ = roc_curve(target, scores)
        plt.subplot(2,1,1)
        plt.title("score distribution")
        plt.hist(scores[mc_sig_index], bins=40, label="Signal", range=[0,1], histtype=u"step")
        plt.hist(scores[mc_bkg_index], bins=40, label="Background", range=[0,1], histtype=u"step")
        plt.yscale("log")
        plt.ylabel("#events")
        plt.xlabel("prediction score")
        plt.legend(loc="best")
        #plotting ROC curve
        plt.subplot(2,1,2)
        plt.title("ROC curve")
        plt.plot(false_pos_rate, true_pos_rate)
        plt.xlabel("False Positive Rate"), plt.ylabel("True Positive Rate")
        plt.text(0.8, 0.2, f"AUC = {roc_auc_score(target, scores)}", bbox=dict(facecolor="none", edgecolor="black", boxstyle="square"))
        plt.tight_layout()
        plt.savefig(f"{self.log_path}/report_{name}.png")

    def anomaly_detection(self, stats):
        training_dists = torch.tensor(
            [self.get_mean_dist(x, self.training_samples) for x in self.training_samples])
        train_mean, train_std = torch.mean(
            training_dists), torch.std(training_dists)
        d_train_samples = torch.tensor(
            [self.get_mean_dist(x, self.test_features) for x in self.test_features])
        d_test_samples = torch.tensor(
            [self.get_mean_dist(x, self.test_features) for x in self.test_features])
        delta_trad = torch.tensor(
            [(d_train-train_mean)/(train_std+1e-8) for d_train in d_train_samples])
        delta_new = torch.tensor([(d_test**(-self.m)-d_train**(-self.m))/(
            d_train**(int(-self.m/2))+1e-8) for d_train in d_train_samples])
        rms_trad, rms_new = (delta_trad**2).mean(), (delta_new**2).mean()
<<<<<<< Updated upstream
        score_trad = 0.5*(1+torch.erf(delta_trad*(1.0/(rms_trad*(2**0.5)))))
        score_new = 0.5*(1+torch.erf(delta_new*(1.0/(rms_new*(2**0.5)))))
        score_new = torch.mul(score_trad, score_new)**0.5

        targets = (self.test_target == 0).float()
        
        anomaly_new = ((score_new>=0.5)==targets).float().mean()
        anomaly_trad = ((score_trad>=0.5)==targets).float().mean()
        
        stats["Anomaly_new Acc"] = anomaly_new
        stats["Anomaly_trad Acc"] = anomaly_trad
        self.plot(scores_trad, target, "Trad")
        self.plot(scores_new, target, "New")
=======
        scores_trad = 0.5*(1+torch.erf(delta_trad*(1.0/(rms_trad*(2**0.5)))))
        scores_new = 0.5*(1+torch.erf(delta_new*(1.0/(rms_new*(2**0.5)))))
        scores_new = torch.mul(scores_trad, scores_new)**0.5

        targets = (self.test_target == 0).float().cpu()
        print("\n\n")
        print(targets)

        anomaly_new = ((scores_new.cpu() >= 0.5) ==
                       targets[:1001]).float().mean()
        anomaly_trad = ((scores_trad.cpu() >= 0.5) ==
                        targets[:1001]).float().mean()

        stats["Anomaly_new Acc"] = anomaly_new
        stats["Anomaly_trad Acc"] = anomaly_trad
        targets[0] = 1
        targets[10] = 1
        scores_trad = np.random.rand(scores_trad.shape[0])
        scores_new = np.random.rand(scores_new.shape[0])
        print(f"scores_trad, scores_new: {scores_trad.shape}, {scores_new.shape}")
        self.plot(scores_trad, targets[:1001], "Trad")
        self.plot(scores_new, targets[:1001], "New")
>>>>>>> Stashed changes
