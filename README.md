# Cats-GAN

This repo is intended to be used as both a general framework for producing simple, synthetic images using WGAN and an example application of generating cat faces (WGAN).

## Models

The GAN in this repo has two separate models, a discriminator and a generator.

The generator $G$ will take a random vector $Z$ and produce a fake image $x^*$. If the generator is successful, the fake image $x^*$ will approximate the distribution of real images $x$. If the discriminator $D$ is successful, then it will be able to distinguish between real images and fake images.

$$
\begin{aligned}
z_i&\sim U(0, 1)\\
G(Z)&=x',~x'\sim x \\
D(x')&=0 \\
D(x)&=1
\end{aligned}
$$


## Cat Data

The cat face data used in this project are obtained from 5 different publid datasets, organized in fferlito's [Cat-faces-dataset github repo](https://github.com/fferlito/Cat-faces-dataset). 

![](data/cats/cat_0.png)
![](data/cats/cat_1.png)
![](data/cats/cat_2.png)
![](data/cats/cat_3.png)
![](data/cats/cat_4.png)
![](data/cats/cat_5.png)

