# MNIST Training

This directory contains all files related to training a model on the MNIST dataset and further optimizing and packaging it for deployment for using it with SageMaker.

## Files

* `requirements.txt` - The development dependencies file for the training process.
* `model.py` - The model definition used for training a neural network on the mnist dataset. 
* `train_mnist.py` - The main training script. This script is responsible for downloading the dataset, training the model, and saving it as `/results/model.pth`.
* `results` - The directory where model artifacts are saved to.
* `to_torchscript.py` - A script which converts the saved model to TorchScript as `/results/model.pt`.
* `inference.py` - The inference script. This module is responsible for loading the model and using it to make predictions in a SageMaker environment.
* `wrap_script.sh` - A script that wraps the necessary files of the model into a zip file (`/results/zip/model.tar.gz`) which can be used as a SageMaker model.
