# Knowledge Distillation from VGG16 to Mobilenet

- Teacher model - VGG16
- Student model - Mobilenet

## Requirements

- python 3.10
- pytorch 2.2.1+cu121

## Usage
### 1. Models
#### VGG16
VGG16, a renowned CNN architecture from the University of Oxford, excels in image classification with its 16 layers and simple structure. It comprises 13 convolutional and 3 fully connected layers, employing 3x3 filters and 2x2 max-pooling. Despite its effectiveness, its size and depth can be computationally intensive.

#### MobileNet
MobileNet, developed by Google, is tailored for mobile and embedded devices, featuring 28 layers and innovative depthwise separable convolutions. This reduces parameters and computational complexity while maintaining performance. MobileNet adapts to various input sizes and is widely used for transfer learning, albeit potentially sacrificing some accuracy for efficiency.

### 2. Dataset
We have used the CIFAR-100 dataset. 
- It contains 60,000 color images.
- Images are of size 32x32 pixels.
- The dataset is organized into 100 classes, each containing 600 images.
- There are 50,000 training images and 10,000 testing images.
- Each image is labeled with one of the 100 fine-grained classes.

### 3. Train the model
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

To perform knowledge distillation from the trained VGG16 to the Mobilenet model
```bash
# use gpu to train mobilenet
$ python knowledge_distillation_train.py -gpu -teacher path_to_best_vgg16_weights_file -student path_to_best_mobilenet_weights_file
```

The weights file with the best accuracy would be written to the disk with name suffix 'best' (default in checkpoint folder).


### 4. test the model
Test the VGG16 model 
```bash
$ python test.py -net vgg16 -weights path_to_best_vgg16_weights_file
```

Test the mobilenet model 
```bash
$ python test.py -net mobilenet -weights path_to_best_mobilenet_weights_file
```

Test the knowledge distilled mobilenet model 
```bash
$ python knowledge_distillation_test.py -gpu -weights path_to_best_knowledge_distilled_mobilenet_weights_file
```

## Implementation Details and References

- VGG [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- Mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- Hyperparameter settings [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.1 divided by 5 at 60th, 120th, 160th epochs, trained for 200 epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9.
- Code reference [GitHub : weiaicunzai](https://github.com/weiaicunzai/pytorch-cifar100)

## Results

|dataset|network|learning rate|batch size|size (MB)|params|top1 err|top5 err|time(ms) per inference step (CPU)|time(ms) per inference step (CPU)|FLOPs|
|:-----:|:-----:|:----:|:----:|:------:|:----:|:------:|:------:|:--------------:|:--------------:|:---------:|
|cifar100|vgg16|0.1|128|136.52|34.0M|27.77|10.12|164.3091|11.0140|334.14|
|cifar100|mobilenet|0.1|128|24.03|3.32M|33.06|10.15|62.9727|9.4442|48.32|
|cifar100|knowledge distilled mobilenet|0.1|128|24.03|3.32M|32.61|10.26|63.8224|9.7993|48.32|
|cifar100|knowledge distilled mobilenet|0.001|64|24.03|3.32M|32.16|10.83|65.0266|8.7168|48.32|
