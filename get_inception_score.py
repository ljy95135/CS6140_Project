"""
It will use the image GAN generates (8*8)
first it will separate the images to 64 and then get Inception Score

it may need time and internet to download inception neural net
"""

import glob
import os
import sys
import tarfile

import math
import numpy as np
import scipy.misc
import tensorflow as tf
from six.moves import urllib

MODEL_DIR = './imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None


def split_images(filenames):
    """
    split image into 64 images to evaluate their Inception score
    each splited image is 128*128

    need image_slicer
    """
    import image_slicer
    for file in filenames:
        image_slicer.slice(file, 64)


def get_inception_score(images, splits=10):
    """
    Call this function with list of images. Each of elements should be a
    numpy array with values ranging from 0 to 255.

    ref: https://github.com/openai/improved-gan
    """
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 100
    with tf.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'InputTensor:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


def _init_inception():
    """
    automatically called when softmax is none

    ref: https://github.com/openai/improved-gan
    """
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Import model with a modification in the input tensor to accept arbitrary
        # batch size.
        input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                      name='InputTensor')
        _ = tf.import_graph_def(graph_def, name='',
                                input_map={'ExpandDims:0': input_tensor})
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)


def main(files):
    def get_images(filename):
        return scipy.misc.imread(filename)

    images = [get_images(filename) for filename in filenames]
    print(len(images))
    print(get_inception_score(images))


if __name__ == '__main__':
    # # edit and run this command when need to split the 8*8 image to 64 images
    # filenames = glob.glob(os.path.join('./inception_score/original_images_inception_score', '*.*'))
    # split_images(filenames)

    # get inception score
    if softmax is None:
        _init_inception()

    # change name to directory of result images
    filenames = glob.glob(os.path.join('./inception_score/original_images_inception_score', '*.*'))
    main(filenames)
