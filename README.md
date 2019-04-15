# CS6140 Project GAN for generating dogs
This project is for course 6140 Machine Learning. It uses Deep Convolutional Generative Adversarial Network (DCGAN) to generate dog imasges.
The dataset chosen 3 breeds of dogs from Stanford Dog Dataset. Different preprocessing is using to generate more training images and have differnt
dataset to compare the results. The quality of the generated  image is measured by Inception Score.

# Preporcessing
I cannot just use center_crop to get my images for the position of dog in Stanford Dog Dataset is not static and just crop will get very bad result.

# Prerequisite
* Anaconda Python 3.7
* tensorflow environment
```sh
conda create -n tensorflow_gpuenv tensorflow-gpu
conda activate tensorflow_gpuenv
```
It should contains most necessary packages like SciPy, NumPy and Pillow
* image_slicer
* (Optional) moviepy

# Usage
```sh
# train original_images (it how many iteration you want in main.py)
python main.py --dataset original_images --input_height=128 --output_height=128 --train
# get samples
python main.py --dataset original_images --input_height=128 --output_height=128

```

# Reference
The code is derived from

# Inception Score
Inception Score(IS) is used to evaluate the quality of images that network generates. From the paper I read, usually it will use 50k images to feed into Inception network.
However, using a laptop is just hard to obey this requirement. For split =8 and images from 64 to 640, the result below:

It shows that after n>200, the Inception Score does not change a lot, so I will use 640 images to compute IS to evaluate my model.
