Cifar10 dataset Challenge
![Chat Application](
    https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png
)
The Cifar10 dataset, with 60,000 images, is one of the popular datasets in the field of Computer Vision, allowing the evaluation of model performance in classifying very small images (32×32 pixels) and experimentation.

The purpose of creating this repository is to examine the performance of various models, including simple neural networks and Convolutional Neural Networks (CNNs), on this dataset.

One of the most successful models is ResNet, which has been used in many architectures and newer models. Here, a ResNet model with 270K parameters is implemented, and we will attempt to assess its performance, displaying relevant graphs at the end.

Update: Information related to the MobileNet model has also been added. It's worth noting that this model has not been optimized for Cifar10, but soon, information about smaller models with performance similar to the original model will be available.

All the models were created layer by layer by myself, and I did not use ready-to-use models from TensorFlow. I want to improve myself by developing CNN architectures from what is written in articles.


# Refrences

 Papers

[VGG Paper](https://arxiv.org/pdf/1409.1556.pdf)

[Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)

[Resnet](https://arxiv.org/pdf/1512.03385.pdf)

[Mobilenet]()

# Enviroment setup

For ease of installation and execution of the relevant files, you can easily set up the environment for running the code by simply running the install.sh file.

Note that to execute the install.sh file, you need to grant permission to execute it, which you can do using this command.

``` sudo chmod -R 777 .```
 next 
``` sudo ./install.sh ```

# How to use

### data
 
To access this dataset, the code utilizes tensorflow.keras.datasets, eliminating the need for separate data downloads. Upon code execution, this dataset will automatically be loaded using the load function.


### train

By simply running the main.py file, the training process will automatically commence. The training logs will be stored in a directory named "logs," which already exists in this repository, eliminating the need to train for accessing the training progress.

### Evaluation
To evaluate this dataset on the cifar10 test section, a file named eval_cifar10.py has been created. It requires only one argument during execution, which is the path to the .keras or .h5 file.

Example:
- ``` python3 eval_cifar10.py './model.keras' ```

 convert

Conversion functionality has not been implemented yet. However, once this section is completed, an update will be made, and additional explanations will be provided.

In this section, you can see visualization and tables of these models during testing Eval.

# Evaluation Reports

|  Model      | params  | accuracy | validation loss  |
| :---:       |  :---:  |  :---:   |      :---:       |
| ResNet      |   270 K |    90    |      0.4744      |
| MobileNetV1 | 3.509 M |   90.38  |      0.4625      |



# Model keras vs onnx vs Quantized onnx formats for MobileNetV1 Benchmarking (Speed , size)


It's important to note that the speed test is a numerical average derived from detecting an image a hundred times.

|  Model         | size (MB)| speed (ms)    |     accuracy     |
| :---:          |  :---:   |  :---:        |      :---:       |
| keras          |   28.3   |    100        |      90.38       |
| onnx           |   13.9   |     9         |      90.38       |
| Quantized onnx |   3.6    |     24        |      82.92       |


The conclusion I've drawn is that using ONNX can greatly increase the speed of the model on CPU. This is because it significantly reduces the model's floating points sizes and ... . Therefore, ONNX can be extensively utilized for deploying artificial intelligence models in systems with memory and hardware limitations, allowing for broader usage in such constrained environments.

#### Formats speed Plots (NEW)
[evaluation_loss_vs_iteration](./MobileNetSpeedTest/plots/speed_models.png)



ResNet Plots
[evaluation_loss_vs_iteration](./ResNet/Plots/)

MobileNet Plots
[MobileNet](./MobileNet/plots/)


# Download Models
[DOWNLOAD ResNet Model](https://drive.google.com/file/d/1TAVxMqBrmFTmV4KSTwp52cFEL6f75HKs/view?usp=sharing)
[DOWNLOAD MobileNet Model](https://drive.google.com/file/d/110rINRfM_LCZPNCIpWafLWO7YawMYgeA/view?usp=sharing)"# cifar10_challenge" 
