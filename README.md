# CocoaNet
> A Convolutional Neural Network for classifying cocoa infections

<div align=center><img src="./data/cocoa.jpg"/></div>

## Table of Contents

* [General info](#general-info)
* [Data](#data)
* [Technologies](#technologies)
* [Future work](#future-work)

## General info

70% of the world’s cocoa beans come from four West African countries: Ivory Coast, Ghana, Nigeria and Cameroon. Cocoa is the chief agricultural export of Ghana and Ghana's main cash crop. However there are numerous infections that are crippling cocoa production. 

The purpose of this project is to develop an image classification system that would be able to analyse cocoa images and detect the presence of infections.

## Data

A dataset of 3,618 images has been curated which contains images in seven classes, which are:
* Black Pod
* Swollen Shoot
* Healthy Leaves
* Healthy Pod
* Pest Infested Pod
* pest Infested Stem
* Healthy Stem

## Technologies

The project is created with:
* Python 3.6+
* Pytorch 1.0
* Fastai 

## Future work

The model has been on the data we've gathered and deployed as a mobile application which you can find [here](https://play.google.com/store/apps/details?id=com.cocoanet). However, we intend to curate another dataset which would have bounding boxes around the infections in an image. This is to allow the image to identify multiple infections in a single image.
