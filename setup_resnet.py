
## Modified by Jinghui Chen to adopt ResNet_V2 model for attack code.
## Original copyright license follows.


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import random
import tarfile
import scipy.misc
import PIL

import numpy as np
from six.moves import urllib
import tensorflow as tf

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os


DATA_URL = 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz'
_RESNET_CHECKPOINT_NAME = 'resnet_v2_50.ckpt'
DIR_NAME = 'resnet'
RESNET_DIR = os.path.join(
    os.path.dirname(__file__),
    DIR_NAME
)
RESNET_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    DIR_NAME,
    _RESNET_CHECKPOINT_NAME
)


def optimistic_restore(session, save_file):
#     print (save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file) 
    
#     variables_to_restore = slim.get_variables_to_restore(include=["resnet_v2_50"])
#     saver = tf.train.Saver(variables_to_restore)
#     saver.restore(session, save_file) 


    
def _get_model(reuse):
    arg_scope = nets.resnet_v2.resnet_arg_scope()
    func = nets.resnet_v2.resnet_v2_50
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse = reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image



class resnet_model:
    def __init__(self, sess):
        global _resnet_initialized
        
        self.sess = sess
        self.image_size = 299
        self.num_channels = 3
        self.num_labels = 1001
        
        _resnet_initialized = False
    
    def predict(self, image):
        global _resnet_initialized
        
        network_fn = _get_model(reuse=_resnet_initialized)
        size = self.image_size
        preprocessed = _preprocess(image, size, size)
        logits, _ = network_fn(preprocessed)
#         logits = logits[:,1:] # ignore background class
#         print (logits)
        logit = tf.squeeze(logits, [1, 2]) 
        predictions = tf.argmax(logit, 1)
        
#         print (logits.shape, logit.shape, predictions.shape)

        if not _resnet_initialized:
            optimistic_restore(self.sess, RESNET_CHECKPOINT_PATH)
            _resnet_initialized = True
            
        return logit, predictions 
    

def download_and_extract():
    """Download and extract model tar file."""
    dest_directory = RESNET_DIR
    if not os.path.exists(RESNET_DIR):
        os.makedirs(RESNET_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main():
    download_and_extract()
    

def load_image(ff):
    path = "../imagenetdata/imgs/"+ff
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0 - 0.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    return [img, int(ff.split(".")[0])]


class ImageNet:
    def __init__(self):
        from multiprocessing import Pool
        pool = Pool(8)
        file_list = sorted(os.listdir("../imagenetdata/imgs/"))
        random.seed(12345)

        random.shuffle(file_list)
        r = pool.map(load_image, file_list[0:900])
#         print(file_list[0:500])
        r = [x for x in r if x != None]
        test_data, test_labels = zip(*r)
#         print (test_labels)
        self.test_data = np.array(test_data)
        self.test_labels = np.zeros((len(test_labels), 1001))
        self.test_labels[np.arange(len(test_labels)), test_labels] = 1



if __name__ == '__main__':
    main()
