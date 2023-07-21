# Autoencoder-classifier test task
This is a test task for a Middle Reseacher (СV) position.



## Project Description
### Problem Statement

    Написать и обучить модель-автокодировщик на датасете на выбор: CIFAR10, CIFAR100.
    Обучить модель-классификатор на латентных представлениях обученного автокодировщика.


### Data

![CIFAR10](/reports/CIFAR10.png)
To see full data statistics you can see `run/data_exploration.ipynb`.
https://www.cs.toronto.edu/~kriz/cifar.html


### Solution

In this project a few small models was trained on CPU to check the parametrs like dimentions of latent space classifier input.

For classifier training is on the "bottleneck" autoencoder latent space representations: 
![AE_bottleneck](/reports/AE_bottleneck.png)

The scheme of solution pipeline:
![pipeline](/reports/pipeline.png)

#### Encoder architectures

##### Convolutional Autoencoder

Pros:

- Can handle spatial information effectively due to the convolutional layers.
- Can potentially provide better performance when trained on image data like CIFAR10 or CIFAR100.
- Easy to understand and implement.

Cons:

- May fail to generate new samples that are as diverse as the original dataset because it lacks a mechanism for ensuring that the distribution of latent variables has good coverage of the space.
- The structure and size of the network can greatly influence the model's ability to learn.

##### Variational Autoencoder:

Pros:

- Incorporates probabilistic encoders and decoders, and thus handles uncertainties better.
- Can generate new samples by drawing from the latent space.

Cons:

- More complex to implement due to the reparameterization trick.
- Sometimes can generate blurrier images compared to other generative models.
- The choice of the latent space dimension is critical and can significantly influence the model's performance.

#### Classifier

There is all-pairs testing for 2  autoencoder architectures and 1 classifier for different latent space dimention on both CIFAR10 and CIFAR100 normalised datasets.


### Results
To see the final results of training process for all model pairs on different latent space dimentions you can see `run/master_notebook.ipynb`.

Example:

AE_loss for CIFAR10 dataset:
![AE_CIFAR10_loss](/reports/AE_CIFAR10_loss.png)

AE_loss for CIFAR100 dataset:
![AE_CIFAR100_loss](/reports/AE_CIFAR100_loss.png)

Classifier performance:
![Classifier](/reports/Classifier.png)

## Structure
    .                              
    ├── data                                # CIFAR10/CIFAR100 datasets
    ├── run 
    │   ├── data_exploration.ipynb          # Explore dataset classes and structure
    │   └── master_notebook.ipynb           # Full pipeline
    ├── reports                             # Solution plots
    ├── models                              # Trained models and checkpoints
    └── src
        ├── data_loader.py                  # Load and preprocess the CIFAR10/CIFAR100 dataset
        ├── autoencoder.py                  # Autoencoder architectures
        ├── train.py                        # Autoencoder training loop
        ├── classifier.py                   # Classifier architectures and training procedure
        ├── baseline.py                     # Baseline solution
        └── utils.py                        # Useful functions