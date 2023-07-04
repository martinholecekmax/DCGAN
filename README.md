# DCGAN

This is a PyTorch implementation of DCGAN which is a generative adversarial network (GAN) using deep convolutional neural networks (CNN). The DCGAN architecture was first explored in 2015 and has seen impressive results in generating new images; you can read the paper [here](https://arxiv.org/pdf/1511.06434.pdf).

## Training on MNIST

The `train.py` script will download the MNIST dataset and train the model on the dataset. You can run the script using the following command:

```
python train.py
```

## Training on CelebA

The `train_celeba.py` script will train the model on the CelebA dataset downloaded from [here](https://www.kaggle.com/dataset/504743cb487a5aed565ce14238c6343b7d650ffd28c071f03f2fd9b25819e6c9). After downloading the dataset, you need to extract the images into a folder named `celeba_dataset`. Then you can run the script using the following command:

```
python train_celeba.py
```

## Folder Structure

The folder structure is as follows:

```
.
├── celeba_dataset
│   └── images
│       ├── 000001.jpg
│       ├── ...
│       └── 202599.jpg
├── dataset
│   └── MNIST
│       └── raw (downloaded dataset)
├── weights
│   └── (saved weights)
├── discriminator.py
├── generator.py
├── test.py
├── train_celeba.py
└── train.py
```
