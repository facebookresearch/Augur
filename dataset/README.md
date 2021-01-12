## Datasets

This folder contains the profiled popular neural network topologies used both inside as well as outside of Facebook. 

The datasets are divided into four categories:

* *fb_convnets:* These are the neural network topologies used within Facebook for various applications
* *torchvision models:* These are the popular neural network topologies that are available in [torch model zoo](https://pytorch.org/docs/stable/torchvision/models.html)
* *neural power:* These includes the neural network topologies used in [NeuralPower paper](https://arxiv.org/pdf/1710.05420.pdf). NeuralPower is a related work where they provide layery
by layer profiling and uses a 2nd order polynomial fit as the model.
* *SuperNet NAS:* These includes 250 randomly generated neural network during Neural Architecture Search.

All the datasets have the following format:

| Kx          | IFMx        | IFMy         | Zin              | Zout              | Stride | Padding | Groups      | Latency                              | Energy                              |
|-------------|-------------|--------------|------------------|-------------------|--------|---------|-------------|--------------------------------------|-------------------------------------|
| 3           | 224         | 224          | 3                | 64                | 1      | 1       | 1           | xxxxxx (cycles)                      | xxxxxx (pJ)                         |

Where:
* **Kx:** Kernel Size 
* **IFMx:** Input Width 
* **IFMy:** Input Height
*  **Zin:** # Input Channels
* **Zout:** # Output Channels
* **Stride:** Stride
* **Padding:** Padding  
* **Groups:** # of groups
* **Latency:** Latency estimated by profiler. The unit is in cycles
* **Energy:** Energy estimated by Profiler. The unit is in pJ


**Note:** These are not the training dataset. Please use these to evaluate the accuracy of the Augur Models against output of Profiler.
