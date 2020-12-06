import os
import torch
import argparse
import pytorch_lightning as pl
from utils import read_config, get_early_stopper, get_checkpoint_callback, final_logs, print_dict
from train import Model
from dataset import DatasetModule
import numpy as np
from models.model import pDNN

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.ini")
parser.add_argument("--num_gpus", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.config
    gpus = args.num_gpus if args.num_gpus is not None else 0
    params = read_config(filename)
    use_gpu = (args.num_gpus > 0)

    if params["JOB_TYPE"] == "train":
        if not os.path.exists(params["SAVE_DIR"]):
            os.makedirs(params["SAVE_DIR"])
        if not os.path.exists(params["LOG_DIR"]):
            os.makedirs(params["LOG_DIR"])
        if not os.path.exists(params["CHECKPOINTS_DIR"]):
            os.makedirs(params["CHECKPOINTS_DIR"])
        
        dataset = DatasetModule(root_path=params["ROOT_PATH"],
                                campaigns=params["CAMPAIGN"],
                                channel=params["CHANNEL"],
                                norm_array=params["NORM_ARRAY"],
                                sig_sum=params["SIG_SUM"],
                                bkg_sum=params["BKG_SUM"],
                                bkg_list=params["BKG_LIST"],
                                sig_list=params["SIG_LIST"],
                                test_ratio=params["TEST_SPLIT"],
                                val_ratio=params["VAL_SPLIT"],
                                batch_size=params["BATCH_SIZE"])

        early_stopping, logger, model_checkpoint = None, None, None
        if params["EARLY_STOP"]:
            early_stopping = get_early_stopper(
                monitor=params["ES_MONITOR"], min_delta=params["ES_DELTA"], patience=params["ES_PATIENCE"], mode=params["ES_MODE"])

        if params["SAVE_TB_LOGS"]:
            logger = pl.loggers.TensorBoardLogger(
                save_dir=params["LOG_DIR"], log_graph=False)

        if params["SAVE_MODEL"]:
            model_checkpoint = get_checkpoint_callback(
                PATH=params["CHECKPOINTS_DIR"], monitor='val_loss', save_last=params["CHECK_EPOCH"])  #

        model = Model(
            momentum=params["MOMENTUM"],
            nesterov=params["NESTEROV"],
            learn_rate=params["LEARN_RATE"],
            learn_rate_decay=params["LEARN_RATE_DECAY"],
            sig_class_weight=params["SIG_CLASS_WEIGHT"],
            bkg_class_weight=params["BKG_CLASS_WEIGHT"],
            optimizer=params["OPTIMIZER"],
            classifier_nodes=params["CLASSIFIER_NODES"],
            encoder_nodes=params["ENCODER_NODES"],
            dropout=params["DROPOUT"],
            activation=params["ACTIVATION"],
            input_size=dataset.input_size,
            output_size=len(params["BKG_LIST"]),
            save_tb_logs=params["SAVE_TB_LOGS"],
            log_path=params["LOG_DIR"],
            K=params["K"],
        )

        trainer = pl.Trainer(early_stop_callback=early_stopping,
                             checkpoint_callback=model_checkpoint,
                             logger=logger,
                             max_epochs=params["EPOCHS"],
                             gpus=gpus)
        '''training the model'''
        trainer.fit(model, dataset)