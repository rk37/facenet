"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face

import matplotlib.pyplot as plt


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 80 # minimum size of face
    threshold = [ 0.7, 0.8, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            # img_list.append(prewhitened)
            img_list.append(aligned)
    images = np.stack(img_list)
    return images

# def main(args):
#     images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
#     pass



if __name__ == '__main__':

    image_size = 160
    margin = 44
    gpu_memory_fraction = 0.5

    image_files = ["../data/images/BKEY7264.jpg",
                   "../data/images/IMG_3524.JPG",
                   "../data/images/IMG_3890.JPG",
                   "../data/images/IMG_3894.JPG",
                   "../data/images/IMG_20150723_135442.jpg",
                   "../data/images/IMG_20160502_135317.jpg",
                   "../data/images/IMG_20160517_112108.jpg",
                   "../data/images/IMG_20160517_112303.jpg"]

    images = load_and_align_data(image_files, image_size, margin, gpu_memory_fraction)

    lw = len(images)
    for idx,i in enumerate(images):
        plt.subplot((lw/8)+1,8,(idx)+1)
        plt.imshow(i)
        pass


    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model("../data/pre_trainde_models/20180402-114759/20180402-114759.pb")

            images = [facenet.prewhiten(img) for img in images]

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(images)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, images[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')

        pass
    pass

    plt.show()

