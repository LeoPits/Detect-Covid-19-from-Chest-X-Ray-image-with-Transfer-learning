# Detect-Covid-19-from-Chest-X-Ray-image-with-Transfer-learning
## Introduction
This repository contains the implementation of a deep learning model to detect COVID-19 from chest X-ray images using transfer learning in PyTorch.
## Dataset
The model will be trained using datasets from two sources
* Cohen’s COVID Chest X-ray Dataset: This is an open database of COVID-19 cases with chest X-ray or CT images, collected from various public sources and through indirect collection from hospitals and physicians1. (https://github.com/ieee8023/covid-chestxray-dataset.git)
* Paul Mooney’s Chest X-ray Dataset (Pneumonia): This dataset contains X-ray images of patients suffering from Pneumonia, which can be used to compare against normal chest X-rays. It’s available on Kaggle and includes images for training, testing, and validation2.(https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
![healty](https://github.com/LeoPits/Detect-Covid-19-from-Chest-X-Ray-image-with-Transfer-learning/assets/19689590/9e4c0388-a04a-4586-9b34-7a23be5b1a57)
![positive](https://github.com/LeoPits/Detect-Covid-19-from-Chest-X-Ray-image-with-Transfer-learning/assets/19689590/aeb65c9d-6e6f-4100-9d11-78eed088c304)

## Model Overview
We use a pre-trained MobileNet V2 model and fine-tune it on a dataset of chest X-ray images to classify them as COVID-19 positive or negative.


## Requirements
- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
