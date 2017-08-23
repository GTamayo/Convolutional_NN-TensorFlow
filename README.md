# Convolutional_NN-TensorFlow
Convolutional Neural Network using tensorflow over CIFAR10 dataset

### CIFAR10 dataset Description
CIFAR10 is a dataset with a long history in computer vision and machine learning. Like MNIST, it
is a common benchmark that various methods are tested against. CIFAR10 is a set of 60,000 color images
of size 32×32 pixels, each belonging to one of ten categories: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, and truck.


### Requirements
- [Download dataset CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and extract in CIFAR-10_data
- install requirements

### Train Model

| Parameter        | Description           |  Default|
| ------------- |:-------------|:-------------|
| steps         | step or epoch in training process  | 100 |
| batch_size    | train batch size |100|


```sh
python3 train_model.py -steps 100 -batch_size 100 
```
