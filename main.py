# -*- coding: utf-8 -*-
# """
# Created on Mon Oct 11 18:18:52 2021

# @author: claus
# """
import pickle
import torch
import numpy as np
import os
import pandas as pd
import glob
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from time import time

import training
import testing
from dataloaders import *
# from model_file import *
import config
import file_management


# %%


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    # config.WVAD_model = load_model_initial()
    # file_management.load_model()
    print(count_parameters(config.WVAD_model))

    torch.autograd.set_detect_anomaly(False)

    # training_results2 = load_results()
    epochs = 1
    start_time = time()
    # file_management.save_model_initial(config.WVAD_model)
    # config.learning_rate_AN *= 0.0000000000000000001
    for t in range(epochs):
        # t = 6
        config.training_results_AUC["alpha"].append(config.AN_weight)
        print(f"Epoch {t+1}\n--------------TRAIN-----------------")
        config.WVAD_model.train()
        # dataset_train = TIMIT_train()
        # train_data_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
        epoch_time = time()
        config.training_results_big["time_passed"].append(
            epoch_time-start_time)
        config.padded = 0
        # training.train_loop_AN(train_data_loader, 0, t, config.padded)

        config.learning_rate *= config.LR_factor
        config.learning_rate_AN *= config.LR_factor

        print(f"Epoch {t+1}\n--------------VALIDATION-----------------")
        noises = ["N1", "N2", "N3", "N4"]
        SNRs = ["-5", "0", "5", "10", "15", "20", "CLEA"]
        config.validation = 1
        config.padded = 0
        config.WVAD_model.eval()
        for j in noises:
            for k in SNRs:
                print(f"{j} {k}")
                config.SNR_level_AURORA = k
                config.noise_type_AURORA = j
                dataset_test = AURORA2_test()
                test_loader = DataLoader(
                    dataset_test, batch_size=1, shuffle=False)
                testing.validation_loop(test_loader, 0, t, config.padded)
                # test_loop_ROC(test_loader, loss_best,t, padded)
    #     file_management.save_results(config.training_results_big, t)
        file_management.save_model(config.WVAD_model, t)


    noises = ["N1", "N2", "N3", "N4"]
    SNRs = ["-5", "0", "5", "10", "15", "20", "CLEA"]
    config.validation = 1
    config.padded = 0
    config.WVAD_model.eval()
    tholds = np.linspace(-1, 1, 101)
    for t in range(1):
        config.VAD_threshold = t
        for j in noises:
            for k in SNRs:
                config.SNR_level_AURORA = k
                config.noise_type_AURORA = j
                dataset_test = AURORA2_test()
                test_loader = DataLoader(
                    dataset_test, batch_size=1, shuffle=False)
                testing.testing_loop(test_loader, 0, t, config.padded)
    file_management.save_results_AUC(config.training_results_AUC)
    print("Done!")
# 