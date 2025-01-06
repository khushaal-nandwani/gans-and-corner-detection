# GANs, WGANs, and Corner Detection

This project showcases implementations of advanced techniques in generative adversarial networks (GANs), Wasserstein GANs (WGANs), and computer vision tasks like corner detection. Each task demonstrates unique methodologies: GANs and WGANs highlight the creation of realistic images with contrasting architectures and optimization strategies, while the corner detection task employs the Harris Corner Detector to identify key features in real-world images. This comprehensive exploration bridges generative models and computer vision, offering valuable insights into their practical applications.

# Table of Contents
- [Usage](#usage)
- [Task 1](#task-1-gan)
- [Task 2](#task-2-wgan)
- [Task 3](#task-3-corner-detection)


# Usage
All the tasks are in their respective files. Basically just run with python with the necessary libraries. For example for task 1, you would run `python task1.py`.

For Task 3, you will have to set the following variables as True or False, depending on what subsection of task you want to test. Please only make one of them true at a time. 
show_corners = False
scatterplot = True
print_evs = False

# [Task 1](task1.py): GAN 
A GAN is implemented where the generator uses transposed convolutional layers and the discriminator uses convolutional layers. 
Each has 5 layers. 

## Generator's Architecture
- Transpose Convolution Layer 1: 64 -> 512, kernel_size=4, stride=1, padding=0
- Transpose Convolution Layer 2: 512 -> 256, kernel_size=3, stride=2, padding=1
- Transpose Convolution Layer 3: 256 -> 128, kernel_size=4, stride=2, padding=1
- Transpose Convolution Layer 4: 128 -> 64, kernel_size=4, stride=2, padding=1
- Transpose Convolution Layer 5: 64 -> 1, kernel_size=3, stride=1, padding=1
- Sigmoid Activation

Each layer has a batch normalization layer and a ReLU activation function between them. 

## Discriminator's Architecture
- Convolution Layer 1: 1 -> 16, kernel_size=4, stride=2, padding=0
- Convolution Layer 2: 16 -> 64, kernel_size=4, stride=2, padding=1
- Convolution Layer 3: 64 -> 128, kernel_size=4, stride=2, padding=1
- Convolution Layer 4: 128 -> 64, kernel_size=4, stride=2, padding=1
- Convolution Layer 5: 64 -> 1, kernel_size=4, stride=2, padding=1

Each layer has a LeakyReLU activation function.

# [Task 2](task2.py): WGAN

Note: ChatGPT was used to create this code. To understand the intuition the original [WGAN](https://arxiv.org/abs/1701.07875) implementation was used.

 Compared to simple GAN, my WGAN converged way faster and producd better results. My
WGAN converged around 357th epoch, while the simple GAN converged around 1000th epoch.
Here is the image generated from 357th epoch.

Training WGAN was different from conv GAN in serveral ways

- Loss Function: While Conv GAN usd Binary Cross Entropy Loss with logits, in WGAN we used Wasserstein Loss or Earth
Mover’s Distance.
- Weight Clipping: In WGAN, we had to clip the weights of the discriminator to a small
range, to ensure that the discriminator is a K-Lipschitz function. This is not required in
Conv GAN.
- Discriminator vs Critic: In Conv GAN, we used the term discriminator, while in
WGAN, we used the term critic. This is because, in WGAN, the critic is not a binary
classifier, but a function that estimates the Wasserstein distance between the real and
generated distributions.
- Critic Updation: In Conv GAN, discriminator and generator are updatd equally. In
WGAN, the critic is updated more times than the generator, in our cas . This is because,
the critic is more important in WGAN, as it is used to estimate the Wasserstein distance.
- Optimizer: In Conv GAN, we used Adam optimizer for both generator and discrimi-
nator. In WGAN, we used RMSProp optimizer for the critic and the generator.
- Mode Collapse: As mentioned before, I ran into a lot of mode collapse in the simple
GAN. I had to tweak various parameters to finally make the model converge. In WGAN, I
did not face any mode collapse, and the model converged smoothly. As expected, WGAN
is more stable than simple GAN.

# [Task 3](task3.py): Corner Detection
Two images (I1 and I2) of the Sandford Fleming Building taken under two different
viewing directions:
• https://commons.wikimedia.org/wiki/File:University College, University of Toronto.jpg
• https://commons.wikimedia.org/wiki/File:University College Lawn, University of Toronto, Canada.jpg

available in [images folder](/images).

The task is to detect corners in these images using the Harris corner detector. The steps are as follows:
- The eigenvalues of the Second Moment Matrix (M) was calculated for each pixel of I1 and
I2.
- The scatter plot of λ1 and λ2 (where λ1 >λ2) for all the pixels in I1 and the same
scatter plot for I2 was plotted as shown [here](images/scatterplot.png). Each point shown at location (x,y) in the scatter plot,
corresponds to a pixel with eigenvalues: λ1 = x and λ2 = y.
- Based on the scatter plots, a threshold for min(λ1,λ2) was picked to detect corners. The corners detected in I1 and I2 were shown as red dots in the images [here](images/corners.png).

