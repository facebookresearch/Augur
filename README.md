# Augur
Augur is a Neural Network based profiler that provides per-layer inference latency and 
energy cost of running a NN layer on a consumer grade commercial NN accelerator.

## Motivation
Explosion in the use of neural networks for a wide range of applications has exposed stark contrast in the achieved efficiency between the model architectures used in benchmarks (e.g., ResNets) versus the model used in real-world applications. One of the primary contributors for the efficiency gap is the lack of easily accessible tools/models for accurately predicting the performance of a given neural network model on a target hardware. Existing tools/simulators to predict performance rely on analytical models and are very architecture specific. Though analytical models are fast, they trade-off speed with accuracy. Similarly, the tools used in industry are often architecture specific and hence making them public will compromise the intellectual property of the hardware accelerator. In order to mitigate this issue while also equipping researchers with tools for benchmarking their candidate neural network models, we introduce “Augur”. In Augur , we use neural network to abstract the internal neural network accelerator. We believe, the methodology can also applied to other hardware accelerators.


## Results

The Augur models (Latency and Energy) are validated with the following neural network topologies:

### Augur Latency Model

#### Torch Vision Models
source: [Here](https://pytorch.org/docs/stable/torchvision/models.html)

| Name 	| Mean Error 	| Median Error 	|
|-	|-	|-	|
| mnasnet_conv_layers 	| 7.16 	| 6.581838369 	|
| mobilenetv2_conv_layers 	| 12.65218776 	| 13.82807732 	|
| resnet18_conv_layers 	| 10.37722175 	| 8.54156208 	|
| ResNet50_conv_layers 	| 5.496340983 	| 6.219698191 	|
| resnext_conv_layers 	| 12.92247698 	| 10.45484352 	|
| squeezenet_conv_layers 	| 30.01804018 	| 30.90766144 	|
| wide_resnet_50_conv_layers 	| 6.541868343 	| 8.048145294 	|

The per-layer error distribution for the torch vision models are shown below:
![Augur Torch Vision Error Distribution](https://github.com/facebookresearch/Augur/tree/main/media/torch_vision_error_dist_latency.png) 

#### NN topologies used within Facebook

| Name 	| Mean Error 	| Median Error 	|
|-	|-	|-	|
| fb_convnet_0 	| 5.446041399 	| 6.167070866 	|
| fb_convnet_1 	| 17.68401167 	| 19.73781013 	|
| fb_convnet_2 	| 8.940167127 	| 6.181158066 	|
| fb_convnet_3 	| 5.246578295 	| 1.712571144 	|
| fb_convnet_4 	| 8.442601095 	| 8.828035355 	|
| fb_convnet_5 	| 8.374918502 	| 0.373719104 	|
| fb_convnet_6 	| 12.01322312 	| 10.48944187 	|
| fb_convnet_7 	| 9.479609925 	| 6.007945538 	|
| fb_convnet_8 	| 0.057147175 	| 0.057147175 	|


The per-layer error distribution for the NN models used within Facebook are shown below:

![Augur FB NN Networks Error Distribution](https://github.com/facebookresearch/Augur/tree/main/media/fb_nets_latency_error.png) 

### Augur Energy Model

#### Torch Vision Models
source: [Here](https://pytorch.org/docs/stable/torchvision/models.html)

| Name 	| Mean 	| Error 	|
|-	|-	|-	|
| mnasnet_conv_layers 	| 5.435133171 	| 4.46666503 	|
| mobilenetv2_conv_layers 	| 11.60152614 	| 12.13379955 	|
| resnet18_conv_layers 	| 7.646391082 	| 4.701381683 	|
| ResNet50_conv_layers 	| 2.029456476 	| 1.972967923 	|
| resnext_conv_layers 	| 7.848110957 	| 6.782331467 	|
| squeezenet_conv_layers 	| 21.96295261 	| 23.18850613 	|
| wide_resnet_50_conv_layers 	| 4.996872148 	| 4.782130241 	|

The per-layer error distribution for the torch vision models are shown below:
![Augur Torch Vision Error Distribution](https://github.com/facebookresearch/Augur/tree/main/media/torch_vision_error_dist_energy.png) 


#### NN topologies used within Facebook

| Name 	| Mean 	| Error 	|
|-	|-	|-	|
| fb_convnet_0 	| 3.934220818 	| 4.201302528 	|
| fb_convnet_1 	| 13.34231735 	| 14.56498528 	|
| fb_convnet_2 	| 5.898854964 	| 4.68245554 	|
| fb_convnet_3 	| 3.095317179 	| 0.631572843 	|
| fb_convnet_4 	| 4.392609096 	| 2.520818233 	|
| fb_convnet_5 	| 6.307614606 	| 0.647551119 	|
| fb_convnet_6 	| 3.434638685 	| 2.690585852 	|
| fb_convnet_7 	| 6.479155425 	| 4.923496723 	|
| fb_convnet_8 	| 0.019582175 	| 0.019582175 	|

The per-layer error distribution for the NN models used within Facebook are shown below:

![Augur FB NN Networks Error Distribution](https://github.com/facebookresearch/Augur/tree/main/media/fb_nets_energy_error.png) 

## Using Augur for NAS

Here are the results for Augur on 250 randomly generated SuperNet neural networks:

![Augur FB NN Networks Error Distribution](https://github.com/facebookresearch/Augur/tree/main/media/pareto_plots.png) 

## Ogranization

The repository is organized as follows:

```
.
+---- dataset
+---- lib
+---- models
+---- qhoptim
+---- README.md
+---- train
+---- tutorials
```
Each of the sub-folders also have Readme which has more information about the utility.

* `datasets` folder contains popular neural network topologies widely used inside as well as outside of facebook. These are also profiled with internal profiler which will be used a ground truth data to evaluate the accuracy of the Augur models.

* `lib` folder contains the NODE specific training files used in training for the Augur models.

* `models` folder contains the Augur models for predicting the latency and energy given a input layer specification. These models are trained in PyTorch.

* `train` folder contains the script for training the Augur models with a new dataset.

* `tutorials` folder contains a Jupyter notebooks. Currently there are two notebook which demonstrates how to use the Augur models. The NN topologies inside the `dataset` folder is used to demonstrate the functionality.


## License

Our code is released under CC BY-NC 4.0. However, our code depends on other libraries,
including Numpy, which each have their own respective licenses that must also be
followed.
