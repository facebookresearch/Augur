## Augur Pytorch Models

This folder includes the trained pytorch models for predicting the latency and energy given an NN layer input. The architecture is based [Neural Oblivious Decision Trees (NODE)](https://arxiv.org/pdf/1909.06312.pdf).

The directory structure is as follows:

```
+-- energy
¦   +-- checkpoint_best_mse.pth
¦   +-- good_models
¦       +-- good_model_1.pth
¦       +-- good_model_2.pth
+-- latency
    +-- checkpoint_best_mse.pth
    +-- good_models
        +-- latency_model_jan_02.pth
```

The pytorch model inside the `energy folder` is the model used by Augur to predict the energy. Likewise, the pytorch model inside the `Latency` folder is
the model used by Augur to predict the latency. Each of the model is a six layer Oblivious decision tree model:
```
Trainer(
  (model): Sequential(
    (0): DenseBlock(
      (0): ODST(in_features=8, num_trees=128, depth=8, tree_dim=3, flatten_output=True)
      (1): ODST(in_features=392, num_trees=128, depth=8, tree_dim=3, flatten_output=True)
      (2): ODST(in_features=776, num_trees=128, depth=8, tree_dim=3, flatten_output=True)
      (3): ODST(in_features=1160, num_trees=128, depth=8, tree_dim=3, flatten_output=True)
      (4): ODST(in_features=1544, num_trees=128, depth=8, tree_dim=3, flatten_output=True)
      (5): ODST(in_features=1928, num_trees=128, depth=8, tree_dim=3, flatten_output=True)
    )
    (1): Lambda()
  )
)
```

The input (X-vector) to the NODE model is the layer specification (Hyperparameters) in the following format:

| Kx          | IFMx        | IFMy         | Zin              | Zout              | Stride | Padding | Groups      | 
|-------------|-------------|--------------|------------------|-------------------|--------|---------|-------------|
| 3           | 224         | 224          | 3                | 64                | 1      | 1       | 1           |


If we use the energy model, the output is Energy predicted in pJ

| Energy                              |
|-------------------------------------|
| xxxxxx (pJ)                         |
| Energy estimated by Profiler |

Similarly, if we use the latency model, the output is Latency predicted in cycles. Please use appropriate frequency to convert into sec.

| Latency                              |
|--------------------------------------|
| xxxxxx (cycles)                      |
| Latency estimated by profiler |

## How to use the Augur Models

For information about how to use the model for predicting the latency and energy, please refer to the tutorial [here](https://github.com/facebookresearch/Augur/blob/main/tutorials/tutorial_augur_test_networks.ipynb)

## Information about the models

The best model is denoted by `xxxbest_mse.pth` suffix. Please use that first.
