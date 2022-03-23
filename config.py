import torch
import numpy as np
from torch import nn

from model_file import combined_networks

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))
# %%
training_results = {
    "val_-5" : [],
    "val_0" : [],
    "val_5" : [],
    "val_10" : [],
    "val_15" : [],
    "val_20" : [],
    "val_CLEAN" : [],
    "test_-5" : [],
    "test_0" : [],
    "test_5" : [],
    "test_10" : [],
    "test_15" : [],
    "test_20" : [],
    "test_CLEAN" : [],
    "epochs" : [],
    "learning_rate" : [],
    "training_loss" : [],
    "training_acc" : [],
    "time_passed" : []
    }


training_results_big = {
    "training" : [],
    "val_caf_-5" : [],
    "val_caf_0" : [],
    "val_caf_5" : [],
    "val_caf_10" : [],
    "val_caf_15" : [],
    "val_caf_20" : [],
    "val_caf_CLEAN" : [],
    "test_caf_-5" : [],
    "test_caf_0" : [],
    "test_caf_5" : [],
    "test_caf_10" : [],
    "test_caf_15" : [],
    "test_caf_20" : [],
    "test_caf_CLEAN" : [],
    "val_bbl_-5" : [],
    "val_bbl_0" : [],
    "val_bbl_5" : [],
    "val_bbl_10" : [],
    "val_bbl_15" : [],
    "val_bbl_20" : [],
    "val_bbl_CLEAN" : [],
    "test_bbl_-5" : [],
    "test_bbl_0" : [],
    "test_bbl_5" : [],
    "test_bbl_10" : [],
    "test_bbl_15" : [],
    "test_bbl_20" : [],
    "test_bbl_CLEAN" : [],
    "val_bus_-5" : [],
    "val_bus_0" : [],
    "val_bus_5" : [],
    "val_bus_10" : [],
    "val_bus_15" : [],
    "val_bus_20" : [],
    "val_bus_CLEAN" : [],
    "test_bus_-5" : [],
    "test_bus_0" : [],
    "test_bus_5" : [],
    "test_bus_10" : [],
    "test_bus_15" : [],
    "test_bus_20" : [],
    "test_bus_CLEAN" : [],
    "val_ped_-5" : [],
    "val_ped_0" : [],
    "val_ped_5" : [],
    "val_ped_10" : [],
    "val_ped_15" : [],
    "val_ped_20" : [],
    "val_ped_CLEAN" : [],
    "test_ped_-5" : [],
    "test_ped_0" : [],
    "test_ped_5" : [],
    "test_ped_10" : [],
    "test_ped_15" : [],
    "test_ped_20" : [],
    "test_ped_CLEAN" : [],
    "val_ssn_-5" : [],
    "val_ssn_0" : [],
    "val_ssn_5" : [],
    "val_ssn_10" : [],
    "val_ssn_15" : [],
    "val_ssn_20" : [],
    "val_ssn_CLEAN" : [],
    "test_ssn_-5" : [],
    "test_ssn_0" : [],
    "test_ssn_5" : [],
    "test_ssn_10" : [],
    "test_ssn_15" : [],
    "test_ssn_20" : [],
    "test_ssn_CLEAN" : [],
    "val_str_-5" : [],
    "val_str_0" : [],
    "val_str_5" : [],
    "val_str_10" : [],
    "val_str_15" : [],
    "val_str_20" : [],
    "val_str_CLEAN" : [],
    "test_str_-5" : [],
    "test_str_0" : [],
    "test_str_5" : [],
    "test_str_10" : [],
    "test_str_15" : [],
    "test_str_20" : [],
    "test_str_CLEAN" : [],
    "epochs" : [],
    "learning_rate" : [],
    "loss_DB" : [],
    "loss_AN" : [],
    "test_TP" : [],
    "test_FP" : [],
    "test_TN" : [],
    "test_FN" : [],
    "time_passed" : []
    }

training_results_AUC = {
   
    "N1_-5_ACC" : [],
    "N1_0_ACC" : [],
    "N1_5_ACC" : [],
    "N1_10_ACC" : [],
    "N1_15_ACC" : [],
    "N1_20_ACC" : [],
    "N1_CLEA_ACC" : [],
    
    "N1_-5_TP" : [],
    "N1_-5_FP" : [],
    "N1_-5_TN" : [],
    "N1_-5_FN" : [],
    "N1_0_TP" : [],
    "N1_0_FP" : [],
    "N1_0_TN" : [],
    "N1_0_FN" : [],
    "N1_5_TP" : [],
    "N1_5_FP" : [],
    "N1_5_TN" : [],
    "N1_5_FN" : [],
    "N1_10_TP" : [],
    "N1_10_FP" : [],
    "N1_10_TN" : [],
    "N1_10_FN" : [],
    "N1_15_TP" : [],
    "N1_15_FP" : [],
    "N1_15_TN" : [],
    "N1_15_FN" : [],
    "N1_20_TP" : [],
    "N1_20_FP" : [],
    "N1_20_TN" : [],
    "N1_20_FN" : [],
    "N1_CLEA_TP" : [],
    "N1_CLEA_FP" : [],
    "N1_CLEA_TN" : [],
    "N1_CLEA_FN" : [],
    
    "N2_-5_ACC" : [],
    "N2_0_ACC" : [],
    "N2_5_ACC" : [],
    "N2_10_ACC" : [],
    "N2_15_ACC" : [],
    "N2_20_ACC" : [],
    "N2_CLEA_ACC" : [],
    
    "N2_-5_TP" : [],
    "N2_-5_FP" : [],
    "N2_-5_TN" : [],
    "N2_-5_FN" : [],
    "N2_0_TP" : [],
    "N2_0_FP" : [],
    "N2_0_TN" : [],
    "N2_0_FN" : [],
    "N2_5_TP" : [],
    "N2_5_FP" : [],
    "N2_5_TN" : [],
    "N2_5_FN" : [],
    "N2_10_TP" : [],
    "N2_10_FP" : [],
    "N2_10_TN" : [],
    "N2_10_FN" : [],
    "N2_15_TP" : [],
    "N2_15_FP" : [],
    "N2_15_TN" : [],
    "N2_15_FN" : [],
    "N2_20_TP" : [],
    "N2_20_FP" : [],
    "N2_20_TN" : [],
    "N2_20_FN" : [],
    "N2_CLEA_TP" : [],
    "N2_CLEA_FP" : [],
    "N2_CLEA_TN" : [],
    "N2_CLEA_FN" : [],
    
    "N3_-5_ACC" : [],
    "N3_0_ACC" : [],
    "N3_5_ACC" : [],
    "N3_10_ACC" : [],
    "N3_15_ACC" : [],
    "N3_20_ACC" : [],
    "N3_CLEA_ACC" : [],
    
    "N3_-5_TP" : [],
    "N3_-5_FP" : [],
    "N3_-5_TN" : [],
    "N3_-5_FN" : [],
    "N3_0_TP" : [],
    "N3_0_FP" : [],
    "N3_0_TN" : [],
    "N3_0_FN" : [],
    "N3_5_TP" : [],
    "N3_5_FP" : [],
    "N3_5_TN" : [],
    "N3_5_FN" : [],
    "N3_10_TP" : [],
    "N3_10_FP" : [],
    "N3_10_TN" : [],
    "N3_10_FN" : [],
    "N3_15_TP" : [],
    "N3_15_FP" : [],
    "N3_15_TN" : [],
    "N3_15_FN" : [],
    "N3_20_TP" : [],
    "N3_20_FP" : [],
    "N3_20_TN" : [],
    "N3_20_FN" : [],
    "N3_CLEA_TP" : [],
    "N3_CLEA_FP" : [],
    "N3_CLEA_TN" : [],
    "N3_CLEA_FN" : [],
    
    "N4_-5_ACC" : [],
    "N4_0_ACC" : [],
    "N4_5_ACC" : [],
    "N4_10_ACC" : [],
    "N4_15_ACC" : [],
    "N4_20_ACC" : [],
    "N4_CLEA_ACC" : [],
    
    "N4_-5_TP" : [],
    "N4_-5_FP" : [],
    "N4_-5_TN" : [],
    "N4_-5_FN" : [],
    "N4_0_TP" : [],
    "N4_0_FP" : [],
    "N4_0_TN" : [],
    "N4_0_FN" : [],
    "N4_5_TP" : [],
    "N4_5_FP" : [],
    "N4_5_TN" : [],
    "N4_5_FN" : [],
    "N4_10_TP" : [],
    "N4_10_FP" : [],
    "N4_10_TN" : [],
    "N4_10_FN" : [],
    "N4_15_TP" : [],
    "N4_15_FP" : [],
    "N4_15_TN" : [],
    "N4_15_FN" : [],
    "N4_20_TP" : [],
    "N4_20_FP" : [],
    "N4_20_TN" : [],
    "N4_20_FN" : [],
    "N4_CLEA_TP" : [],
    "N4_CLEA_FP" : [],
    "N4_CLEA_TN" : [],
    "N4_CLEA_FN" : [],
    
    "time_passed" : [],
    "threshold" : [],
    "alpha" : []
    }
learning_rate = 1e-2# 1e-3
learning_rate_AN = 1e-3
VAD_threshold = 0
LR_factor = 0.7
# Initialize the loss function
loss_primary = nn.BCELoss()
loss_secondary = nn.CrossEntropyLoss()


WVAD_model = combined_networks().to(device)

# Initialize the optimiser


optimizer_conv1R = torch.optim.RMSprop(WVAD_model.conv1.parameters(), lr=learning_rate)
optimizer_conv2R = torch.optim.RMSprop(WVAD_model.conv2.parameters(), lr=learning_rate)
optimizer_conv3R = torch.optim.RMSprop(WVAD_model.conv3.parameters(), lr=learning_rate)
optimizer_conv4R = torch.optim.RMSprop(WVAD_model.conv4.parameters(), lr=learning_rate)

optimizer_FB = torch.optim.RMSprop(WVAD_model.FB.parameters(), lr=learning_rate)

optimizer_AN1 = torch.optim.RMSprop(WVAD_model.AN1.parameters(), lr=learning_rate_AN)
optimizer_AN2 = torch.optim.RMSprop(WVAD_model.AN2.parameters(), lr=learning_rate_AN)
optimizer_AN3 = torch.optim.RMSprop(WVAD_model.AN3.parameters(), lr=learning_rate_AN)

optimizer_DB1 = torch.optim.RMSprop(WVAD_model.DB1.parameters(), lr=learning_rate)
optimizer_DB2 = torch.optim.RMSprop(WVAD_model.DB2.parameters(), lr=learning_rate)
optimizer_DB3 = torch.optim.RMSprop(WVAD_model.DB3.parameters(), lr=learning_rate)

lambda1 = lambda LR_factor: learning_rate*LR_factor
lambda2 = lambda LR_factor: learning_rate_AN*LR_factor*-1


scheduler_FB = torch.optim.lr_scheduler.LambdaLR(optimizer_FB, lr_lambda=lambda1)

scheduler_DB1 = torch.optim.lr_scheduler.LambdaLR(optimizer_DB1, lr_lambda=lambda1)
scheduler_DB2 = torch.optim.lr_scheduler.LambdaLR(optimizer_DB2, lr_lambda=lambda1)
scheduler_DB3 = torch.optim.lr_scheduler.LambdaLR(optimizer_DB3, lr_lambda=lambda1)

scheduler_AN1 = torch.optim.lr_scheduler.LambdaLR(optimizer_AN1, lr_lambda=lambda2)
scheduler_AN2 = torch.optim.lr_scheduler.LambdaLR(optimizer_AN2, lr_lambda=lambda2)
scheduler_AN3 = torch.optim.lr_scheduler.LambdaLR(optimizer_AN3, lr_lambda=lambda2)



noise_type_AURORA = "caf"
SNR_level_AURORA = 0
training_batch_size = 1
testing_batch_size = 160
validation = 0
padded = 0
training = 0
AN_weight = 0.1

output_folder = "Enter your folder name here"
