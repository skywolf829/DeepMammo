import sys
sys.path.insert(0, '/models/research/')
import os
import tensorflow as tf
slim = tf.contrib.slim
import tf_slim.modle.slim.nets as net
import tf_slim
import inception_v4
import cv2

checkpoint = "./inception_v4.ckpt"

sess = tf.Session()
arg_scope = net.inception_v4_arg_scope()
input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
with slim.arg_scope(arg_scope):
    logits, end_points = net.inception_v4(inputs=input_tensor)


