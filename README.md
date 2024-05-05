# Knowledge Distillation from VGG16 to Mobilenet

Teacher model - VGG16
Student model - Mobilenet

## Requirements

- python 3.10
- pytorch 2.2.1+cu121

## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. Dataset
We have used the CIFAR-100 dataset. 
- It contains 60,000 color images.
- Images are of size 32x32 pixels.
- The dataset is organized into 100 classes, each containing 600 images.
- There are 50,000 training images and 10,000 testing images.
- Each image is labeled with one of the 100 fine-grained classes.

### 4. Train the model
To train the VGG16 model
```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```

To train the Mobilenet model
```bash
# use gpu to train mobilenet
$ python train.py -net mobilenet -gpu
```

The weights file with the best accuracy would be written to the disk with name suffix 'best' (default in checkpoint folder).


### 5. test the model
Test the VGG16 model 
```bash
$ python test.py -net vgg16 -weights path_to_best_vgg16_weights_file
```

Test the mobilenet model 
```bash
$ python test.py -net mobilenet -weights path_to_best_mobilenet_weights_file
```

## Implementation Details and References

- VGG [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- Mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- Hyperparameter settings [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.1 divided by 5 at 60th, 120th, 160th epochs, trained for 200 epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9. 

## Results

|dataset|network|params|top1 err|top5 err|time(ms) per inference step|FLOPs|
|:-----:|:-----:|:----:|:------:|:------:|:-------------:|:--------------:|
|cifar100|vgg16|34.0M|27.77|10.12|269.19|334.14|
|cifar100|mobilenet|3.3M|33.06|10.15|49.25|48.32|
