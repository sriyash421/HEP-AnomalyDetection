import os
import json
from configparser import ConfigParser
import pytorch_lightning as pl
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss, roc_curve
import matplotlib.pyplot as plt


def read_config(filename="config.ini"):
    '''read config'''
    if not os.path.exists(filename):
        raise Exception("Config file not found")
    print(f"Parsing the config file")
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(filename)

    temp = {}
    for section in parser.sections():
        params = parser.items(section)
        for param in params:
            temp[param[0]] = param[1]
    config = dict()

    config["JOB_NAME"] = str(temp["job_name"])
    config["JOB_TYPE"] = str(temp["job_type"])
    # if config["JOB_TYPE"] == "train":
    count = 0
    if os.path.exists(temp["save_dir"]):
        for i in os.listdir(temp["save_dir"]):
            if config["JOB_NAME"] in i:
                count += 1
    config["SAVE_DIR"] = os.path.join(
        str(temp["save_dir"]), config["JOB_NAME"]+f"_{count}")
    config["LOG_DIR"] = os.path.join(config["SAVE_DIR"], "logs")
    config["CHECKPOINTS_DIR"] = os.path.join(
        config["SAVE_DIR"], "checkpoints")
    # elif config["JOB_TYPE"] == "test":
    #     config["LOAD_DIR"] = str(temp["load_dir"])
    #     config["RESULTS_DIR"] = str(temp["results_dir"])
    #     config["LOG_DIR"] = os.path.join(config["RESULTS_DIR"], "logs")

    config["ROOT_PATH"] = str(temp["root_path"])
    config["NUM_FEATURES"] = int(temp["num_features"])
    config["CAMPAIGN"] = list(json.loads(str(temp["campaigns"])))
    config["CHANNEL"] = str(temp["channel"])
    config["NORM_ARRAY"] = True if temp["norm_array"] == "true" else False
    config["SIG_SUM"] = float(temp["sig_sumofweight"])
    config["BKG_SUM"] = float(temp["bkg_sumofweight"])
    config["BKG_LIST"] = list(json.loads(temp["bkg_list"]))
    config["SIG_LIST"] = list(json.loads(temp["sig_list"]))
    config["TEST_SPLIT"] = float(temp["test_rate"])
    config["VAL_SPLIT"] = float(temp["val_split"])
    config["BATCH_SIZE"] = int(temp["batch_size"])

    config["EARLY_STOP"] = True if temp["use_early_stop"] == "true" else False
    config["ES_MONITOR"] = str(temp["early_stop_monitor"])
    config["ES_DELTA"] = float(temp["early_stop_min_delta"])
    config["ES_PATIENCE"] = int(temp["early_stop_patience"])
    config["ES_MODE"] = str(temp["early_stop_mode"])
    config["ES_RESTORE"] = True if temp["early_stop_restore_best_weights"] == "true" else False

    config["SAVE_TB_LOGS"] = True if temp["save_tb_logs"] == "true" else False

    config["SAVE_MODEL"] = True if temp["save_model"] == "true" else False
    config["CHECK_EPOCH"] = True if temp["check_model_epoch"] == "true" else False

    config["EPOCHS"] = int(temp["epochs"])
    config["MOMENTUM"] = float(temp["momentum"])
    config["NESTEROV"] = True if temp["nesterov"] == "true" else False
    config["LEARN_RATE"] = float(temp["learn_rate"])
    config["LEARN_RATE_DECAY"] = float(temp["learn_rate_decay"])
    config["SIG_CLASS_WEIGHT"] = float(temp["sig_class_weight"])
    config["BKG_CLASS_WEIGHT"] = float(temp["bkg_class_weight"])
    config["OPTIMIZER"] = str(temp["optimizer"])
    config["CLASSIFIER_NODES"] = json.loads(temp["classifier_nodes"])
    config["ENCODER_NODES"] = json.loads(temp["encoder_nodes"])
    config["DROPOUT"] = float(temp["dropout_rate"])
    config["ACTIVATION"] = str(temp["activation_fn"])
    config["K"] = int(temp["k_value"])

    print_dict(config, "Config")
    return config


def print_dict(dict, name):
    '''print dictionaries'''
    print("-"*40+f"{name}"+"-"*40)
    for k, v in dict.items():
        print(f"{k:<50} {v}")


def get_early_stopper(monitor, min_delta, patience, mode):
    print(f"Getting the early stopper")
    early_stopper = pl.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode
    )
    return early_stopper


def get_checkpoint_callback(PATH, monitor, save_last):
    print(f"Getting the checkpoint callback")
    checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=PATH,
        monitor=monitor,
        save_last=save_last
    )
    return checkpoint