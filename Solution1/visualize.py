from os.path import isdir
import tf_cnnvis
import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg19
from keras.preprocessing.image import load_img, img_to_array
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Script visualizes CNN features'
    )
    parser.add_argument(
        '-i', '--image', type=str, help='File path of input image', required=True
    )
    parser.add_argument(
        '-l', '--log', type=str, help='Output directory for log', required=True
    )
    parser.add_argument(
        '-o', '--output', type=str, help='Output directory for output images', required=True
    )
    args = parser.parse_args()
    file_path = args.image
    path_logdir = args.log
    path_outdir = args.output
    return file_path, path_logdir, path_outdir

file_path, path_logdir, path_outdir = get_args()

vgg_dir = '/home/ivcl/Desktop/mammogist2/tensorflow_vgg'  # modify directory
# Make sure vgg exists
if not isdir(vgg_dir):
    raise Exception("VGG directory doesn't exist!")

sess = tf.Session()
print("Session start")
vgg = vgg19.Vgg19()
input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
with tf.name_scope("content_vgg"):
    vgg.build(input_)

# Activation visualization
print(file_path)
im = load_img(file_path, target_size=(224, 224))
im_arr = img_to_array(im)
im_arr = np.expand_dims(im_arr, axis=0)
print(np.shape(im_arr))

layers = ['r', 'p', 'c']

is_success = tf_cnnvis.activation_visualization(
    graph_or_path=tf.get_default_graph(),
    value_feed_dict={input_: im_arr},
    layers=layers,
    path_logdir=path_logdir,
    path_outdir=path_outdir
)

sess.close()