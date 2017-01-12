---
layout: post
title:  "SDCND Traffic Sign Classifier"
date:   2016-12-30 16:00:00 +0000
categories: sdcnd tensorflow
---

Project 2 of [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/drive) by Udacity is about building a deep neural network for recognizing traffic signs.
The dataset for this task is a labeled collection of [German Traffic Signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
Each sign is represented as 32x32 RGB image and a label number which corresponds to one of 43 classes like 'Speed limit (20 km/h)' or 'Priority road'.
Dataset consists of training set with 39209 examples and a test set of 12630.
Project's task is to build traffic sign classifier which implementation is described in this post.

<!--excerpt-->

## Problem

Build a deep learning classifier which accepts 32x32x3 input image and outputs one of 43 class labels.
As the part of the project, tensorflow is the tool which has to be used.

## About the data

Figures below illustrate examples for several signs:
![Speed limit (30km/h) examples](/assets/sdcnd-traffic-signs/signs_examples1.png)
![End of speed limit (80km/h) examples](/assets/sdcnd-traffic-signs/signs_examples2.png)
![Yield examples](/assets/sdcnd-traffic-signs/signs_examples3.png)
![Dangerous curve to the right examples](/assets/sdcnd-traffic-signs/signs_examples4.png)
![Beware of ice/snow examples](/assets/sdcnd-traffic-signs/signs_examples5.png)
![Roundabout mandatory examples](/assets/sdcnd-traffic-signs/signs_examples6.png)

Both train and test data is equaly unbalanced.
Here is histogram of examples by classes in training set:
![Train set classes distribution](/assets/sdcnd-traffic-signs/classes_dist_train.png)

Test data histogram:

![Test set classes distribution](/assets/sdcnd-traffic-signs/classes_dist_test.png)

Many of the images in the dataset are very dark and its hard sometimes for a human to recognize what sign is it.
Inbalance and brightness unnormalization is what we need to handle before training the classifier.

## Preprocessing

Inspired by [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) I used normalization by converting an image to YUV space and normalizing Y-channel.
OpenCV offers 2 methods [histogram equalization](http://docs.opencv.org/trunk/d5/daf/tutorial_py_histogram_equalization.html): one is considering global constrast of the image and adaptive equalization.
Figure below shows manually picked dark images and result of both normalizations.

![Image normalization](/assets/sdcnd-traffic-signs/normalization.png)

By visual comparison I choosed to use adaptive normalization.
Better effect is most notable for 30, 60, 70 and 120 speed limits signs.
Next figure shows how this normalizations look for usual colored images

![Image normalization](/assets/sdcnd-traffic-signs/normalization_color.png)

This is actually all steps I made for preprocessing: convert image to YUV and normalize Y-channel.
Neural network input will accept YUV color channels aswell.

## Augmentation

To handle class distribution inbalance in training set, I used image position shifting in range of [-2, 2] pixels, scaling in [.9, 1.1] ratio and rotation in [-15, 15] degrees.
These parameters I also used from [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
Figure below shows how augmented images look like (most left image is a source):

![Image augmentation examples](/assets/sdcnd-traffic-signs/augmented_examples.png)

My next step was to split training set into two: one for actual training and another for validation.
I splitted it in 80%/20% ratio.
Then for training set I used augmentation to ensure 3000 examples for every single class.
Here is an example of the balancing procedure output:
`
Class 0: exist 168, need to generate 2832
Class 1: exist 1776, need to generate 1224
Class 2: exist 1800, need to generate 1200
...
Class 40: exist 288, need to generate 2712
Class 41: exist 192, need to generate 2808
Class 42: exist 192, need to generate 2808
`

Here is the resulting histogram for training set.

![Train set with balanced classes](/assets/sdcnd-traffic-signs/classes_train_augmented.png)

## Model training

I used 2 models for this: LeNet-like model and VGG-like model.

LeNet-like architecture draft:
`INPUT -> CONV1 -> CONV2 -> FC1 -> FC2 -> SOFTMAX`

VGG-like architecture draft:
`INPUT -> CONV1 -> RELU -> CONV2 -> RELU -> POOL1 -> CONV3 -> RELU -> CONV4 -> RELU -> POOL2 -> FC1 -> FC2 -> SOFTMAX`

In total I trained 8 models with next dataset parameters:

| # | Model | Is balanced | Is Normalized | Color space | Epoch 10 Val acc | Overall Test acc |
| - | ----- | ----------- | ------------- | ----------- | ---------------- | ---------------- |
| 1 | LeNet | No          | No            | YUV         | 0.975            | 0.882            |
| 2 | LeNet | No          | Yes           | RGB         | 0.982            | 0.933            |
| 3 | LeNet | No          | Yes           | YUV         | 0.980            | 0.913            |
| 4 | LeNet | Yes         | No            | YUV         | 0.990            | 0.923            |
| 5 | LeNet | Yes         | Yes           | RGB         | 0.975            | 0.957            |
| 6 | LeNet | Yes         | Yes           | YUV         | 0.989            | 0.950            |
| 7 | VGG-like | Yes      | Yes           | RGB         | 0.997            | 0.980            |
| 8 | VGG-like | Yes      | Yes           | YUV         | 0.996            | 0.975            |

Visualization of model performance on test set:

![Different model performance](/assets/sdcnd-traffic-signs/performance.png)

Because this trainings and evaluations were made just with 10 epochs, I took best model and parameters to train it for longer.
Resulting performance of VGG-like model with balanced and normalized images in RGB space is **98.3%** which is not that bad :)

## Outcome

Since this project was not about findinf the most powerful model for this dataset but to show the ability of build and use from scracth tensorflow models, this work is far from being finished.
Key notes from this projects which I took are next.
First, data augmentation is a **very** powerful tool which can boost deep neural network performance significantly.
Second, brightness normalization improved the performance of the model very well, so this and other appliable computer vision techniques definetely should be used for preprocessing.

Promising ways to improve 98.3% even higher:
* use transfer learning (more likely t finetune since some intermediate layer)
* use more powerful models like Inception and ResNet
* use other computer vision approaches for better image preprocessing and normalization
