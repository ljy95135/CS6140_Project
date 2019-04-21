# CS6140 Project GAN for generating dogs
code zip file only contains source code without dataset and checkpoint and other results.
Please go to https://github.com/ljy95135/CS6140_Project to download it.

You can only download some dataset in data folder and train GAN or download the latest checkpoint to generate samples.

This project is for course 6140 Machine Learning. It uses Deep Convolutional Generative Adversarial Network (DCGAN) to generate dog images.
The dataset chosen 3 breeds of dogs from Stanford Dog Dataset. Different pre-processing is using to generate more training images and have different
dataset to compare the results. The quality of the generated  image is measured by Inception Score.

# Prerequisite
* Anaconda Python 3.7
* tensorflow environment
```sh
conda create -n tensorflow_gpuenv tensorflow-gpu
conda activate tensorflow_gpuenv
```
It should contains most necessary packages like SciPy, NumPy and Pillow
* image_slicer
* dlib(to do preprocessing, result is in data folder so it is optional)
It is hard to install dlib on Windows so I create a new environment for dlib
since it will downgrade my python and tensorflow and it is only use for face extracting, it is better to put it in
another environment.
```
conda create --name dlib_face
conda install -c menpo dlib
conda activate dlib_face
```
* (Optional) moviepy

# Pre-porcessing
Stanford Dod Dataset provides bound box to crop and I can also extract face by myself. face extracting uses dlib which
is used for human face extraction so this way is not able to crop dog faces for every image. On the other hand, data augmentation,
which is implemented in preprocessing.py is able to flip and rotate images to generate more training data.

# Usage
```sh
# train original_images (model is stored per 500 iteration, different data size decides how many epoch is needed)
python main.py --dataset original_images --input_height=128 --output_height=128 --epoch=56 --train
python main.py --dataset faces_and_augmented --input_height=128 --output_height=128 --epoch=84 --train
python main.py --dataset cropped_images --input_height=128 --output_height=128 --epoch=56 --train
python main.py --dataset cropped_augmented_images --input_height=128 --output_height=128 --epoch=7 --train

# get samples (change to different trained model by dataset name)
python main.py --dataset faces_and_augmented --input_height=128 --output_height=128
```

# Inception Score
Inception Score(IS) is used to evaluate the quality of images that network generates. From the paper I read, usually it will use 50k images to feed into Inception network.
However, using a laptop is just hard to obey this requirement. For split =8 and images from 64 to 640, the result below:

It shows that after n>300, the Inception Score does not change a lot, so I will use 640 images to compute IS to evaluate my model.

It needs Inception net v3 which will be automatically downloaded if program does not find it in imagenet folder.

```python
    # first part
    # # edit and run this command when need to split the 8*8 image to 64 images
    # filenames = glob.glob(os.path.join('./inception_score/original_images_inception_score', '*.*'))
    # split_images(filenames)

    # second part
    # get inception score
    if softmax is None:
        _init_inception()

    # change name to directory of result images
    filenames = glob.glob(os.path.join('./inception_score/original_images_inception_score', '*.*'))
    main(filenames)
```
In main method, first we want to split generated images because generator generates an 8*8 image and we want to split them.
You can uncomment first part and comment second part. Delete original combined image after split images are created. Then
comment first part and run second part.

# Result
Here is some generated image examples.

Here is graph of gnerator loss
![g loss](https://raw.githubusercontent.com/ljy95135/CS6140_Project/master/result/g_loss.png)

Here is graph of discriminator loss
![d loss](https://raw.githubusercontent.com/ljy95135/CS6140_Project/master/result/d_loss.png)

You can go to result folder to see all other assets.
![examples](https://raw.githubusercontent.com/ljy95135/CS6140_Project/master/result/examples.png)

# Reference
The code is derived from [@carpedm20](https://github.com/carpedm20/DCGAN-tensorflow).
