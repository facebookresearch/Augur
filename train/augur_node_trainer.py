#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# coding: utf-8

# In[1]:



import os, sys
import time
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import lib
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import random
import pandas as pd
from itertools import chain
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
experiment_name = 'augur_node_shallow'
experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(experiment_name, *time.gmtime()[:5])
print("experiment:", experiment_name)


# In[2]:


data = lib.Dataset("AUGUR", random_state=round(time.time()), quantile_transform=False,scaling='None', log_transform=False,quantile_noise=1e-3)
in_features = data.X_train.shape[1]

random_state=1337
output_distribution='normal'

data.y_train = np.log10(data.y_train)
data.y_valid = np.log10(data.y_valid)
data.y_test = np.log10(data.y_test)

print("Dataset reading Successful!")

# Plots the y-distribution
plt.hist(data.y_train, density=False, bins=30)
plt.xlabel('Total performance')
plt.ylabel('count')
plt.savefig("y_train_dist.png")
plt.close()


plt.hist(data.y_test.reshape(-1), density=False, bins=30)
plt.xlabel('Total performance')
plt.ylabel('count')
plt.savefig("y_test_dist.png")
plt.close()

plt.hist(data.y_valid.reshape(-1), density=False, bins=30)
plt.xlabel('Total performance')
plt.ylabel('count')
plt.savefig("y_valid_dist.png")
plt.close()


model = nn.Sequential(
    lib.DenseBlock(in_features, 128, num_layers=6, tree_dim=3, depth=8, flatten_output=False,
                   choice_function=lib.entmax15, bin_function=lib.entmoid15),
    lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree

).to(device)

with torch.no_grad():
    res = model(torch.as_tensor(np.float32(data.X_train[:1000]), device=device))
    # trigger data-aware init


#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)


# In[31]:
from qhoptim.pyt import QHAdam
optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }
print("qhoptim import successful!")

# In[33]:
if(True):
    experiment_name = "augur_energy_6_layers_128_depth8_log_transformed__rel_error"
    #experiment_name = "dummy_test"
    trainer = lib.Trainer(
        model=model, loss_function=F.mse_loss,
        experiment_name=experiment_name,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        verbose=True,
        n_last_checkpoints=5
    )


# Training parameters to control

loss_history, mse_history = [], []
best_mse = float('inf')
best_step_mse = 0
early_stopping_rounds = 5000
report_frequency = 100


# Train and plot the training loss and validation loss

if (True):
    for batch in lib.iterate_minibatches(np.float32(data.X_train), np.float32(data.y_train), batch_size=512,
                                                    shuffle=True, epochs=float('inf')):
        metrics = trainer.train_on_batch(*batch, device=device)

        loss_history.append(metrics['loss'])

        if trainer.step % report_frequency == 0:
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')
            mse = trainer.evaluate_mse(
                np.float32(data.X_valid), np.float32(data.y_valid), device=device, batch_size=512)

            if mse < best_mse:
                best_mse = mse
                best_step_mse = trainer.step
                trainer.save_checkpoint(tag='best_mse')
            mse_history.append(mse)

            trainer.load_checkpoint()  # last
            trainer.remove_old_temp_checkpoints()

            plt.figure(figsize=[18, 6])
            plt.subplot(1, 2, 1)
            plt.plot(loss_history)
            plt.title('Loss')
            plt.grid()
            plt.subplot(1, 2, 2)
            plt.plot(mse_history)
            plt.title('MSE')
            plt.grid()
            #plt.show()
            filename = experiment_name + ".png"
            plt.savefig(filename)
            plt.close()
            print("Loss %.5f" % (metrics['loss']))
            print("Val MSE: %0.5f" % (mse))

        if trainer.step > best_step_mse + early_stopping_rounds:
            print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
            print("Best step: ", best_step_mse)
            print("Best Val MSE: %0.5f" % (best_mse))
            break

# In case you want to test a particular checkpoint, uncomment the following
# and comment line 173-177
'''
trainer_test = lib.Trainer(model=model, loss_function=F.mse_loss)
ckpt_path = "/workspace/node/node/notebooks/augur_energy_6k_dataset_6_layers_128_depth8_log_transformed__rel_error/checkpoint_best_mse.pth"
trainer_test.load_checkpoint(path=ckpt_path)
mse, pred, ground, error = trainer_test.evaluate_mse_test(np.float32(data.X_test), np.float32(data.y_test), device=device)
print('Best step: ', trainer_test.step)
print('Mean Error', np.mean(error))

'''
# Evaluation on the test dataset
trainer.load_checkpoint(tag='best_mse')
mse, pred, ground, error = trainer.evaluate_mse_test(np.float32(data.X_test), np.float32(data.y_test), device=device)
print('Best step: ', trainer.step)
print("Test MSE: %0.5f" % (mse))

# Plot the correlation on the test set
plt.scatter(ground, ground, color='green', alpha=0.1)
plt.scatter(ground, pred, color='gray')
test_filename = experiment_name + "_test.png"
plt.savefig(test_filename)
plt.close()
