import os
import cv2
import glob
import pickle
#import scipy.misc
import math
#import lmdb
import struct
import random
import itertools
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import scipy.io as sio
#from captcha.image import ImageCaptcha
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

import colorsys
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
#from shapely.geometry import Polygon

from .bbox import *
from Mybase.comp_utils import tensor_update


def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


def distort_color(image=None, color_order=0):
    
    if color_order == 0:
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    elif color_order == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    elif color_order == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    elif color_order == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.)
    else:
        raise ValueError("color_order must be in [0, 3]")

    return tf.clip_by_value(image, 0.0, 255.0)



def distort_bbox(boxs=None, order=0):
    
    img_shp = tf.cast(boxs[-1, :-1], dtype=tf.int32)
    boxs = boxs[0:-1]
    if order == 0:
        box_beg = tf.zeros(shape=[3], dtype=tf.int32)
        box_siz = tf.stack([img_shp[0], img_shp[1], -1], axis=-1)
    elif order == 1:
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(img_shp, bounding_boxes=tf.expand_dims(boxs, 0), \
                                                   min_object_covered=0.1, aspect_ratio_range=[0.5, 2.0], \
                                                   area_range=[0.3, 1.0], max_attempts=50, \
                                                   use_image_if_no_bounding_boxes=True)
    elif order == 2:
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(img_shp, bounding_boxes=tf.expand_dims(boxs, 0), \
                                                   min_object_covered=0.3, aspect_ratio_range=[0.5, 2.0], \
                                                   area_range=[0.3, 1.0], max_attempts=50, \
                                                   use_image_if_no_bounding_boxes=True)
    elif order == 3:
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(img_shp, bounding_boxes=tf.expand_dims(boxs, 0), \
                                                   min_object_covered=0.5, aspect_ratio_range=[0.5, 2.0], \
                                                   area_range=[0.3, 1.0], max_attempts=50, \
                                                   use_image_if_no_bounding_boxes=True)
    elif order == 4:
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(img_shp, bounding_boxes=tf.expand_dims(boxs, 0), \
                                                   min_object_covered=0.7, aspect_ratio_range=[0.5, 2.0], \
                                                   area_range=[0.3, 1.0], max_attempts=50, \
                                                   use_image_if_no_bounding_boxes=True)
    elif order == 5:
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(img_shp, bounding_boxes=tf.expand_dims(boxs, 0), \
                                                   min_object_covered=0.9, aspect_ratio_range=[0.5, 2.0], \
                                                   area_range=[0.3, 1.0], max_attempts=50, \
                                                   use_image_if_no_bounding_boxes=True)
    elif order == 6:
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(img_shp, bounding_boxes=tf.expand_dims(boxs, 0), \
                                                   min_object_covered=0.0, aspect_ratio_range=[0.5, 2.0], \
                                                   area_range=[0.3, 1.0], max_attempts=50, \
                                                   use_image_if_no_bounding_boxes=True)
    else:
        raise ValueError("order must be in [0, 6]")
    
    return tf.stack([box_beg, box_siz], axis=0)



class GeneratorForMNIST(object):
    
    def __init__(self, mod_tra=True, dat_dir='Mybase/datasets/train', bat_siz=3, epc_num=20, \
                 min_after_dequeue=30, gpu_lst='0', fil_num=32):
        
        self.mod_tra            = mod_tra
        self.use_pad            = False
        self.use_exp            = False
        self.exp_rat            = 2.0
        self.img_avg            = np.array([0.0], dtype=np.float32)
        self.img_siz_min        = 32
        self.img_siz_max        = 32
        ############for crop###########
        self.min_object_covered = 0.70
        self.aspect_ratio_range = [0.75, 1.33] #(0.5, 2.0) #(3/4, 4/3)
        self.area_range         = [0.70, 1.00]
        self.max_attempts       = 200
        
        self.dat_dir            = dat_dir
        self.bat_siz            = bat_siz
        self.epc_num            = epc_num
        self.min_after_dequeue  = min_after_dequeue
        self.gpu_lst            = gpu_lst
        self.gpu_num            = len(self.gpu_lst.split(','))
        self.mdl_dev            = '/cpu:%d' if self.gpu_num == 0 else '/gpu:%d'
        self.gpu_num            = 1         if self.gpu_num == 0 else self.gpu_num
        self.fil_num            = fil_num
        self.num_readers        = 16
        self.num_threads        = 16
        self.capacity           = self.min_after_dequeue + 3 * self.bat_siz
        self.max_num            = 100
        
        self.cls_nams           = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.cls_num            = len(self.cls_nams)
        self.cls_idx_to_cls_nam = dict(zip(range(self.cls_num), self.cls_nams))
        self.cls_nam_to_cls_idx = dict(zip(self.cls_nams, range(self.cls_num)))
        
        ############for show###########
        self.title              = ''
        self.figsize            =  (3, 3)
        
    def make_input(self, num_per_sha=1000000, fil_nam='t10k'):
        
        fils_dir = 'Mybase/datasets/raw/mnist'
        rcds_dir = "Mybase/datasets/" + fil_nam
        
        lbls_dir = os.path.join(fils_dir, '%s-labels-idx1-ubyte' % fil_nam)
        imgs_dir = os.path.join(fils_dir, '%s-images-idx3-ubyte' % fil_nam)
        with open(lbls_dir, 'rb') as f: 
            magic, n = struct.unpack('>II', f.read(8))
            lbls = np.fromfile(f, dtype=np.uint8)
        with open(imgs_dir, 'rb') as f: 
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16)) 
            imgs = np.fromfile(f, dtype=np.uint8).reshape(len(lbls), 28, 28, 1)
        
        img_num = imgs.shape[0]
        print("The amount of images is %d!" %(img_num))
        idxs = np.arange(0, img_num)
        np.random.shuffle(idxs)
        imgs = [imgs[idx] for idx in idxs]
        lbls = [lbls[idx] for idx in idxs]
        
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            sha_num = int(img_num/num_per_sha)
            if sha_num == 0:
                sha_num = 1
                num_per_sha = img_num
            else:
                num_per_sha = int(math.ceil(img_num/sha_num))

            for sha_idx in range(sha_num):
                
                out_nam = 'mnist_%s.tfrecord' % (fil_nam)
                rcd_nam = os.path.join(rcds_dir, out_nam)
                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(rcd_nam, options=options) as writer:

                    sta_idx = sha_idx * num_per_sha
                    end_idx = min((sha_idx + 1) * num_per_sha, img_num)
                    for i in range(sta_idx, end_idx):
                        if i % 100 == 0:
                            print("Converting image %d/%d shard %d" % (i + 1, img_num, sha_idx))
                        img     = imgs[i]
                        img_hgt = img.shape[0]
                        img_wdh = img.shape[1]
                        #img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                        #img_tmp[:, :, :] = img
                        #img     = img_tmp
                        lbl     = lbls[i]
                        #写tfrecords
                        img_raw = img.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image/image':  _bytes_feature(img_raw),
                            'image/height': _int64_feature(img_hgt),
                            'image/width':  _int64_feature(img_wdh),
                            'label/label':  _int64_feature(lbl),
                        }))
                        writer.write(example.SerializeToString())
    
    def resize_image_with_pad(self, img=None):

        #####################按最短边进行比例缩放######################
        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        if self.use_pad:
            leh_min = tf.minimum(img_hgt, img_wdh)
            leh_max = tf.maximum(img_hgt, img_wdh)
            leh_rat = tf.minimum(self.img_siz_min/leh_min, self.img_siz_max/leh_max)
            img_hgt = tf.cast(img_hgt*leh_rat, dtype=tf.int32)
            img_wdh = tf.cast(img_wdh*leh_rat, dtype=tf.int32)
            #对image操作后对boxs操作
            img = tf.image.resize_images(img, [img_hgt, img_wdh], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
            ################如果最长边过长则按中心对称进行裁剪################
            #对image操作后对boxs操作
            img = tf.image.resize_image_with_crop_or_pad(img, self.img_siz_max, self.img_siz_max)
        else:
            hgt_rat = self.img_siz_max / img_hgt
            wdh_rat = self.img_siz_max / img_wdh
            leh_rat = tf.stack([hgt_rat, wdh_rat], axis=0)
            leh_rat = tf.tile(leh_rat, [2])
            #对image操作后对boxs操作
            img = tf.image.resize_images(img, [self.img_siz_max, self.img_siz_max], method=tf.image.ResizeMethod.BILINEAR, \
                                         align_corners=False)
        return img
    
    def distort_crop(self, img=None):
        
        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        boxs    = tf.stack([[0.0, 0.0, img_hgt-1.0, img_wdh-1.0]], axis=0)
        
        if self.use_exp:
            exp_rat     = tf.random.uniform(shape=[], minval=1.1, maxval=self.exp_rat, dtype=tf.float32)
            #exp_rat    = self.exp_rat
            pad_hgt_all = tf.cast(img_hgt*(exp_rat-1.0), dtype=tf.int32)
            pad_wdh_all = tf.cast(img_wdh*(exp_rat-1.0), dtype=tf.int32)
            pad_hgt_fnt = tf.random.uniform(shape=[], minval=0, maxval=pad_hgt_all, dtype=tf.int32)
            pad_wdh_fnt = tf.random.uniform(shape=[], minval=0, maxval=pad_wdh_all, dtype=tf.int32)
            pad_hgt_bak = pad_hgt_all - pad_hgt_fnt
            pad_wdh_bak = pad_wdh_all - pad_wdh_fnt
            paddings    = [[pad_hgt_fnt, pad_hgt_bak], [pad_wdh_fnt, pad_wdh_bak], [0, 0]]
            img         = tf.pad(img, paddings, "CONSTANT", constant_values=0)
            pad_hgt_fnt = tf.cast(pad_hgt_fnt, dtype=tf.float32)
            pad_wdh_fnt = tf.cast(pad_wdh_fnt, dtype=tf.float32)
            beg         = tf.stack([pad_hgt_fnt, pad_wdh_fnt], axis=0)
            beg         = tf.tile(beg, [2])
            boxs        = boxs + beg #padding中boxs不会超出边界，不用clip
            img_hgt     = tf.cast(tf.shape(img)[0], dtype=tf.float32)
            img_wdh     = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        ########################crop the image randomly########################
        boxs_tmp = boxs / tf.stack([img_hgt-1.0, img_wdh-1.0, img_hgt-1.0, img_wdh-1.0], axis=0)
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(tf.shape(img), bounding_boxes=tf.expand_dims(boxs_tmp, 0), \
                                                   min_object_covered=self.min_object_covered, \
                                                   aspect_ratio_range=self.aspect_ratio_range, \
                                                   area_range=self.area_range, max_attempts=self.max_attempts, \
                                                   use_image_if_no_bounding_boxes=True)
        img = tf.slice(img, box_beg, box_siz)
        ###########resize image to the expected size with paddings############
        img = self.resize_image_with_pad(img)
        return img

    def preprocessing(self, img=None):
        
        img = tf.cast(img, dtype=tf.float32)
        ####################归化到0、1之间######################
        #if img.dtype != tf.float32:
        #    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        if self.mod_tra == True:
            #######################光学畸变#########################
            # Randomly distort the colors. There are 4 ways to do it.
            #img= apply_with_random_selector(img, lambda x, order: distort_color(x, order), num_cases=4)
            img = img - self.img_avg
            img = img / 255.0
            #######################随机裁剪#########################
            #img= self.distort_crop(img)
            #####################随机左右翻转#######################
            img = tf.image.random_flip_left_right(img)
            img = self.resize_image_with_pad(img)
            #######################减去均值########################
            #img= tf.image.per_image_standardization(img)
        else:
            img = img - self.img_avg
            img = img / 255.0
            img = self.resize_image_with_pad(img)
        return img
    
    def get_input(self):

        def parse_function(serialized_example):
            '''
            定长特征解析：tf.FixedLenFeature(shape, dtype, default_value)
                        shape：可当 reshape 来用，如 vector 的 shape 从 (3,) 改动成了 (1,3)。
                               注：如果写入的 feature 使用了. tostring() 其 shape 就是 ()
                        dtype：必须是 tf.float32， tf.int64， tf.string 中的一种。
                        default_value：feature 值缺失时所指定的值。
            不定长特征解析：tf.VarLenFeature(dtype)
                          注：可以不明确指定 shape，但得到的 tensor 是 SparseTensor。
                              变长的tensor转化为string后，shape是固定的，shape=[]，所以可以用tf.FixedLenFeature进行解析
            '''
            parsed_example = tf.parse_single_example(
                serialized_example, 
                features = {
                    'image/image':  tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'image/height': tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'image/width':  tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/label':  tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    #'matrix':      tf.VarLenFeature(dtype=dtype('float32')), 
                    #'matrix_shape':tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                }
            )
            img_hgt = tf.cast(parsed_example['image/height'], tf.int32)
            img_wdh = tf.cast(parsed_example['image/width'],  tf.int32)
            lbl     = tf.cast(parsed_example['label/label'],  tf.int32)
            img     = tf.decode_raw(parsed_example['image/image'], tf.uint8)
            img     = tf.reshape(img, [img_hgt, img_wdh, 1])
            img     = self.preprocessing(img)
            img     = tf.reshape(img, [self.img_siz_max, self.img_siz_max, 1])
            parsed_example = {
                'image/image':  img,
                'image/height': img_hgt,
                'image/width':  img_wdh,
                'label/label':  lbl,
            }
            #parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
            #parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])
            return parsed_example
        
        imgs_lst     = []
        lbls_lst     = []
        
        if self.fil_num >= self.gpu_num:
            fil_pat  = os.path.join(self.dat_dir, 'mnist', '*.tfrecord')
            dataset  = tf.data.Dataset.list_files(file_pattern=fil_pat, shuffle=True, seed=None)
        else:
            fil_nam  = glob.glob(os.path.join(self.dat_dir, 'mnist', '*.tfrecord'))
            dataset  = tf.data.TFRecordDataset(fil_nam, compression_type='ZLIB', num_parallel_reads=self.num_readers)
            
        for i in range(self.gpu_num):
            dat_sha  = dataset.shard(num_shards=self.gpu_num, index=i)
            
            if self.fil_num >= self.gpu_num:
                #dat_sha= dat_sha.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                #                            cycle_length=self.num_readers, block_length=1, num_parallel_calls=1)
                dat_sha = dat_sha.apply(tf.data.experimental.\
                                        parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                                                            cycle_length=self.num_readers, block_length=1, sloppy=True, \
                                                            buffer_output_elements=None, \
                                                            prefetch_input_elements=None))
            #dat_sha = dat_sha.repeat(count=self.epc_num)
            #dat_sha = dat_sha.shuffle(buffer_size=self.capacity, seed=None, reshuffle_each_iteration=True)
            dat_sha  = dat_sha.apply(tf.data.experimental.\
                                     shuffle_and_repeat(buffer_size=self.capacity, count=self.epc_num, seed=None))
            #dat_sha = dat_sha.map(parse_function, num_parallel_calls=self.num_threads)
            #dat_sha = dat_sha.batch(batch_size=self.bat_siz, drop_remainder=False)
            dat_sha  = dat_sha.apply(tf.data.experimental.\
                                     map_and_batch(parse_function, batch_size=self.bat_siz, num_parallel_batches=None, \
                                                   drop_remainder=False, num_parallel_calls=self.num_threads))
            #dat_sha = dat_sha.cache(filename=os.path.join(self.dat_dir, 'cache'))
            #dat_sha = dat_sha.prefetch(buffer_size=self.bat_siz)
            dat_sha  = dat_sha.apply(tf.data.experimental.prefetch_to_device(self.mdl_dev%i, buffer_size=None))
            iterator = dat_sha.make_one_shot_iterator()
            example  = iterator.get_next()
            imgs_lst.append(example['image/image'])
            lbls_lst.append(example['label/label'])
        return imgs_lst, lbls_lst
    
    def display_instances(self, img=None, lbl=None, img_hgt=None, img_wdh=None):
        
        _, ax = plt.subplots(1, figsize=self.figsize)
            
        #img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        #Show area outside image boundaries.
        #ax.set_ylim(img_hgt + 10, -10)
        #ax.set_xlim(-10, img_wdh + 10)
        ax.axis('off')
        ax.set_title(str(lbl))
        
        img = img * 255.0
        img = img + self.img_avg
        img = np.clip(img, 0.0, 225.0)
        img = img.astype(dtype=np.uint8, copy=False)
        img_tmp          = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
        img_tmp[:, :, :] = img
        img              = img_tmp
        ax.imshow(img)
        plt.show()
        plt.close()
    
    def get_input_test(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs_lst, lbls_lst = self.get_input()
        imgs = tf.concat(imgs_lst, axis=0)
        lbls = tf.concat(lbls_lst, axis=0)

        with tf.Session() as sess:
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            imgs_kep, lbls_kep = sess.run([imgs, lbls])
            
            for i in range(self.bat_siz*self.gpu_num):
                img_tmp = imgs_kep[i]
                lbl_tmp = lbls_kep[i]
                self.display_instances(img_tmp, lbl_tmp)


class GeneratorForCIFAR(object):
    
    def __init__(self, mod_tra=True, dat_dir='Mybase/datasets/train', bat_siz=3, epc_num=20, \
                 min_after_dequeue=30, gpu_lst='0', fil_num=32):
        
        self.mod_tra            = mod_tra
        self.use_pad            = False
        self.use_exp            = False
        self.exp_rat            = 2.0
        self.img_avg            = np.array([125.30, 122.95, 113.87], dtype=np.float32)
        self.img_siz_min        = 32
        self.img_siz_max        = 32
        ############for crop###########
        self.min_object_covered = 0.50
        self.aspect_ratio_range = [0.75, 1.33] #(0.5, 2.0) #(3/4, 4/3)
        self.area_range         = [0.50, 1.00]
        self.max_attempts       = 200
        
        self.dat_dir            = dat_dir
        self.bat_siz            = bat_siz
        self.epc_num            = epc_num
        self.min_after_dequeue  = min_after_dequeue
        self.gpu_lst            = gpu_lst
        self.gpu_num            = len(self.gpu_lst.split(','))
        self.mdl_dev            = '/cpu:%d' if self.gpu_num == 0 else '/gpu:%d'
        self.gpu_num            = 1         if self.gpu_num == 0 else self.gpu_num
        self.fil_num            = fil_num
        self.num_readers        = 16
        self.num_threads        = 16
        self.capacity           = self.min_after_dequeue + 3 * self.bat_siz

        self.max_num            = 100
        
        self.cls_nams           = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.cls_num            = len(self.cls_nams)
        self.cls_idx_to_cls_nam = dict(zip(range(self.cls_num), self.cls_nams))
        self.cls_nam_to_cls_idx = dict(zip(self.cls_nams, range(self.cls_num)))
        
        ############for show###########
        self.title              = ''
        self.figsize            = (3, 3)
    
    def ZCA(self, X):
        X = X.reshape(X.shape[0], -1)
        sigma = np.dot(X.T, X)/X.shape[0]
        U,S,V = np.linalg.svd(sigma)
        eps = 0.001
        Z = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S+eps))), U.T)
        X = np.dot(X, Z)
        X = X.reshape(-1, 32, 32, 3)
        return X
    
    def make_input(self, num_per_sha=1000000, fil_nam='data_batch'):
        
        fils_dir = 'Mybase/datasets/raw/cifar10/cifar-10-batches-py'
        #fils_dir = 'Mybase/datasets/raw/cifar100/cifar-100-python'
        rcds_dir = "Mybase/datasets/" + fil_nam
        
        fils_lst = glob.glob(os.path.join(fils_dir, fil_nam+'*'))
        imgs_lst = []
        lbls_lst = []
        for fil in fils_lst:
            with open(fil, 'rb') as f:
                dats = pickle.load(f, encoding='iso-8859-1')
                imgs = dats['data']
                lbls = dats['labels']
                #lbls = dats['fine_labels']
                imgs = imgs.reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype(dtype=np.uint8, copy=False)
                lbls = np.asarray(lbls, dtype=np.int64)
                imgs_lst.append(imgs)
                lbls_lst.append(lbls)
        imgs = np.concatenate(imgs_lst)
        lbls = np.concatenate(lbls_lst)
        '''
        imgs = imgs - self.img_avg
        imgs = imgs / 255.0
        imgs = self.ZCA(imgs)
        imgs = imgs * 255.0 + self.img_avg
        '''
        img_num = imgs.shape[0]
        print("The amount of images is %d!" %(img_num))
        idxs = np.arange(0, img_num)
        np.random.shuffle(idxs)
        imgs = imgs[idxs]
        lbls = lbls[idxs]
        
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            sha_num = int(img_num/num_per_sha)
            if sha_num == 0:
                sha_num = 1
                num_per_sha = img_num
            else:
                num_per_sha = int(math.ceil(img_num/sha_num))

            for sha_idx in range(sha_num):
                
                out_nam = 'cifar_%s.tfrecord' % (fil_nam)
                rcd_nam = os.path.join(rcds_dir, out_nam)
                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(rcd_nam, options=options) as writer:

                    sta_idx = sha_idx * num_per_sha
                    end_idx = min((sha_idx + 1) * num_per_sha, img_num)
                    for i in range(sta_idx, end_idx):
                        if i % 100 == 0:
                            print("Converting image %d/%d shard %d" % (i + 1, img_num, sha_idx))
                        img = imgs[i]
                        img_hgt, img_wdh = img.shape[0], img.shape[1]
                        if img.size == img_hgt * img_wdh:
                            print ('Gray Image %s' %(imgs_lst[i]))
                            img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                            img_tmp[:, :, :] = img[:, :, np.newaxis]
                            img = img_tmp
                        #img = img.astype(np.uint8)
                        img = img.astype(np.float32)
                        assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                        lbl = lbls[i]
                        #写tfrecords
                        img_raw = img.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image/image':  _bytes_feature(img_raw),
                            'image/height': _int64_feature(img_hgt),
                            'image/width':  _int64_feature(img_wdh),
                            'label/label':  _int64_feature(lbl),
                        }))
                        writer.write(example.SerializeToString())
    
    def resize_image_with_pad(self, img=None):

        #####################按最短边进行比例缩放######################
        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        if self.use_pad:
            leh_min = tf.minimum(img_hgt, img_wdh)
            leh_max = tf.maximum(img_hgt, img_wdh)
            leh_rat = tf.minimum(self.img_siz_min/leh_min, self.img_siz_max/leh_max)
            img_hgt = tf.cast(img_hgt*leh_rat, dtype=tf.int32)
            img_wdh = tf.cast(img_wdh*leh_rat, dtype=tf.int32)
            #对image操作后对boxs操作
            img = tf.image.resize_images(img, [img_hgt, img_wdh], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
            ################如果最长边过长则按中心对称进行裁剪################
            #对image操作后对boxs操作
            img = tf.image.resize_image_with_crop_or_pad(img, self.img_siz_max, self.img_siz_max)
        else:
            hgt_rat = self.img_siz_max / img_hgt
            wdh_rat = self.img_siz_max / img_wdh
            leh_rat = tf.stack([hgt_rat, wdh_rat], axis=0)
            leh_rat = tf.tile(leh_rat, [2])
            #对image操作后对boxs操作
            img = tf.image.resize_images(img, [self.img_siz_max, self.img_siz_max], method=tf.image.ResizeMethod.BILINEAR, \
                                         align_corners=False)
        return img
    
    '''
    def distort_crop(self, img=None):
        
        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        boxs    = tf.stack([[0.0, 0.0, img_hgt-1.0, img_wdh-1.0]], axis=0)
        
        if self.use_exp:
            exp_rat     = tf.random.uniform(shape=[], minval=1.1, maxval=self.exp_rat, dtype=tf.float32)
            #exp_rat    = self.exp_rat
            pad_hgt_all = tf.cast(img_hgt*(exp_rat-1.0), dtype=tf.int32)
            pad_wdh_all = tf.cast(img_wdh*(exp_rat-1.0), dtype=tf.int32)
            pad_hgt_fnt = tf.random.uniform(shape=[], minval=0, maxval=pad_hgt_all, dtype=tf.int32)
            pad_wdh_fnt = tf.random.uniform(shape=[], minval=0, maxval=pad_wdh_all, dtype=tf.int32)
            pad_hgt_bak = pad_hgt_all - pad_hgt_fnt
            pad_wdh_bak = pad_wdh_all - pad_wdh_fnt
            paddings    = [[pad_hgt_fnt, pad_hgt_bak], [pad_wdh_fnt, pad_wdh_bak], [0, 0]]
            img         = tf.pad(img, paddings, "CONSTANT", constant_values=0)
            pad_hgt_fnt = tf.cast(pad_hgt_fnt, dtype=tf.float32)
            pad_wdh_fnt = tf.cast(pad_wdh_fnt, dtype=tf.float32)
            beg         = tf.stack([pad_hgt_fnt, pad_wdh_fnt], axis=0)
            beg         = tf.tile(beg, [2])
            boxs        = boxs + beg #padding中boxs不会超出边界，不用clip
            img_hgt     = tf.cast(tf.shape(img)[0], dtype=tf.float32)
            img_wdh     = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        ########################crop the image randomly########################
        boxs_tmp = boxs / tf.stack([img_hgt-1.0, img_wdh-1.0, img_hgt-1.0, img_wdh-1.0], axis=0)
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(tf.shape(img), bounding_boxes=tf.expand_dims(boxs_tmp, 0), \
                                                   min_object_covered=self.min_object_covered, \
                                                   aspect_ratio_range=self.aspect_ratio_range, \
                                                   area_range=self.area_range, max_attempts=self.max_attempts, \
                                                   use_image_if_no_bounding_boxes=True)
        img = tf.slice(img, box_beg, box_siz)
        ###########resize image to the expected size with paddings############
        img = self.resize_image_with_pad(img)
        return img
    '''
    
    def distort_crop(self, img=None):
        
        paddings = [[4, 4], [4, 4], [0, 0]]
        img      = tf.pad(img, paddings, mode='CONSTANT', constant_values=0)
        ymn      = tf.random.uniform(shape=[], minval=0, maxval=8, dtype=tf.int32)
        xmn      = tf.random.uniform(shape=[], minval=0, maxval=8, dtype=tf.int32)
        ymx      = ymn + 32
        xmx      = xmn + 32
        img      = img[ymn:ymx, xmn:xmx, :]
        return img
    
    def preprocessing(self, img=None):
        
        img = tf.cast(img, dtype=tf.float32)
        ####################归化到0、1之间######################
        #if img.dtype != tf.float32:
        #    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        if self.mod_tra == True:
            #######################光学畸变#########################
            # Randomly distort the colors. There are 4 ways to do it.
            # img = apply_with_random_selector(img, lambda x, order: distort_color(x, order), num_cases=4)
            img = img - self.img_avg
            img = img / 255.0
            #######################随机裁剪#########################
            img = self.distort_crop(img)
            #img= self.resize_image_with_pad(img)
            #####################随机左右翻转#######################
            img = tf.image.random_flip_left_right(img)
            #######################减去均值########################
            #img= tf.image.per_image_standardization(img)
        else:
            img = img - self.img_avg
            img = img / 255.0
            #img = self.resize_image_with_pad(img)
        return img
    
    def get_input(self):

        def parse_function(serialized_example):
            '''
            定长特征解析：tf.FixedLenFeature(shape, dtype, default_value)
                        shape：可当 reshape 来用，如 vector 的 shape 从 (3,) 改动成了 (1,3)。
                               注：如果写入的 feature 使用了. tostring() 其 shape 就是 ()
                        dtype：必须是 tf.float32， tf.int64， tf.string 中的一种。
                        default_value：feature 值缺失时所指定的值。
            不定长特征解析：tf.VarLenFeature(dtype)
                          注：可以不明确指定 shape，但得到的 tensor 是 SparseTensor。
                              变长的tensor转化为string后，shape是固定的，shape=[]，所以可以用tf.FixedLenFeature进行解析
            '''
            parsed_example = tf.parse_single_example(
                serialized_example, 
                features = {
                    'image/image':  tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'image/height': tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'image/width':  tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/label':  tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    #'matrix':      tf.VarLenFeature(dtype=dtype('float32')), 
                    #'matrix_shape':tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                }
            )
            img_hgt = tf.cast(parsed_example['image/height'], tf.int32)
            img_wdh = tf.cast(parsed_example['image/width'],  tf.int32)
            lbl     = tf.cast(parsed_example['label/label'],  tf.int32)
            #img    = tf.decode_raw(parsed_example['image/image'], tf.uint8)
            img     = tf.decode_raw(parsed_example['image/image'], tf.float32)
            img     = tf.reshape(img,  [img_hgt, img_wdh, 3])
            img     = self.preprocessing(img)
            img     = tf.reshape(img, [self.img_siz_max, self.img_siz_max, 3])
            parsed_example = {
                'image/image':  img,
                'image/height': img_hgt,
                'image/width':  img_wdh,
                'label/label':  lbl,
            }
            #parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
            #parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])
            return parsed_example
        
        imgs_lst = []
        lbls_lst = []
        
        if self.fil_num >= self.gpu_num:
            fil_pat  = os.path.join(self.dat_dir, 'cifar', '*.tfrecord')
            dataset  = tf.data.Dataset.list_files(file_pattern=fil_pat, shuffle=True, seed=None)
        else:
            fil_nam  = glob.glob(os.path.join(self.dat_dir, 'cifar', '*.tfrecord'))
            dataset  = tf.data.TFRecordDataset(fil_nam, compression_type='ZLIB', num_parallel_reads=self.num_readers)
        
        for i in range(self.gpu_num):
            dat_sha  = dataset.shard(num_shards=self.gpu_num, index=i)
            
            if self.fil_num >= self.gpu_num:
                #dat_sha= dat_sha.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                #                            cycle_length=self.num_readers//self.gpu_num, block_length=1, num_parallel_calls=1)
                dat_sha = dat_sha.apply(tf.data.experimental.\
                                        parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                                                            cycle_length=self.num_readers//self.gpu_num, \
                                                            block_length=1, sloppy=True, \
                                                            buffer_output_elements=None, \
                                                            prefetch_input_elements=None))
            #dat_sha = dat_sha.repeat(count=self.epc_num)
            #dat_sha = dat_sha.shuffle(buffer_size=self.capacity, seed=None, reshuffle_each_iteration=True)
            dat_sha  = dat_sha.apply(tf.data.experimental.\
                                     shuffle_and_repeat(buffer_size=self.capacity, count=self.epc_num, seed=None))
            #dat_sha = dat_sha.map(parse_function, num_parallel_calls=self.num_threads)
            #dat_sha = dat_sha.batch(batch_size=self.bat_siz, drop_remainder=False)
            dat_sha  = dat_sha.apply(tf.data.experimental.\
                                     map_and_batch(parse_function, batch_size=self.bat_siz, num_parallel_batches=None, \
                                                   drop_remainder=False, num_parallel_calls=self.num_threads//self.gpu_num))
            #dat_sha = dat_sha.cache(filename=os.path.join(self.dat_dir, 'cache'))
            #dat_sha = dat_sha.prefetch(buffer_size=self.bat_siz)
            dat_sha  = dat_sha.apply(tf.data.experimental.prefetch_to_device(self.mdl_dev%i, buffer_size=None))
            iterator = dat_sha.make_one_shot_iterator()
            example  = iterator.get_next()
            imgs_lst.append(example['image/image'])
            lbls_lst.append(example['label/label'])
        return imgs_lst, lbls_lst
    
    def display_instances(self, img=None, lbl=None, img_hgt=None, img_wdh=None):
        
        _, ax = plt.subplots(1, figsize=self.figsize)
        
        #img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        #Show area outside image boundaries.
        #ax.set_ylim(img_hgt + 10, -10)
        #ax.set_xlim(-10, img_wdh + 10)
        ax.axis('off')
        ax.set_title(str(lbl))
        
        img = img * 255.0
        img = img + self.img_avg
        img = np.clip(img, 0.0, 225.0)
        img = img.astype(dtype=np.uint8, copy=False)
        ax.imshow(img)
        plt.show()
        plt.close()
        
    def get_input_test(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs_lst, lbls_lst = self.get_input()
        imgs = tf.concat(imgs_lst, axis=0)
        lbls = tf.concat(lbls_lst, axis=0)

        with tf.Session() as sess:
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            imgs_kep, lbls_kep = sess.run([imgs, lbls])
            
            for i in range(self.bat_siz*self.gpu_num):
                img_tmp = imgs_kep[i]
                lbl_tmp = lbls_kep[i]
                self.display_instances(img_tmp, lbl_tmp)


class GeneratorForImageNet(object):
    
    def __init__(self, mod_tra=True, dat_dir='Mybase/datasets/train', bat_siz=3, epc_num=20, \
                 min_after_dequeue=30, gpu_lst='0', fil_num=32):

        self.mod_tra            = mod_tra
        self.use_pad            = True
        self.use_exp            = False
        self.exp_rat            = 2.0
        self.img_avg            = np.array([123.7, 116.8, 103.9], dtype=np.float32)
        self.img_siz_min        = 224
        self.img_siz_max        = 224
        self.box_siz_min        = 5
        self.box_isc_min        = 0.5
        ############for crop###########
        self.min_object_covered = 0.30
        self.aspect_ratio_range = [0.75, 1.33] #(0.5, 2.0) #(3/4, 4/3)
        self.area_range         = [0.30, 1.00]
        self.max_attempts       = 200
        
        self.dat_dir            = dat_dir
        self.bat_siz            = bat_siz
        self.epc_num            = epc_num
        self.min_after_dequeue  = min_after_dequeue
        self.gpu_lst            = gpu_lst
        self.gpu_num            = len(self.gpu_lst.split(','))
        
        self.mdl_dev            = '/cpu:%d' if self.gpu_num == 0 else '/gpu:%d'
        self.gpu_num            = 1         if self.gpu_num == 0 else self.gpu_num
        self.fil_num            = fil_num
        self.num_readers        = 2 
        self.num_threads        = 20
        self.bat_siz_all        = self.bat_siz * self.gpu_num
        self.capacity           = self.min_after_dequeue + 3 * self.bat_siz_all
        
        mets_dir                = 'Mybase/datasets/raw/ILSVRC/devkit/data'
        wnds                    = self.load_imagenet_meta(os.path.join(mets_dir, 'meta_clsloc.mat'))
        self.cls_nams           = [wnds[i] for i in range(1000)]
        self.cls_num            = len(self.cls_nams)
        self.cls_idx_to_cls_nam = dict(zip(range(self.cls_num), self.cls_nams))
        self.cls_nam_to_cls_idx = dict(zip(self.cls_nams, range(self.cls_num)))
        
        ############for show###########
        self.title              = ''
        self.figsize            = (12, 12)
        
    def load_imagenet_meta(self, met_dir):
        
        metadata = sio.loadmat(met_dir, struct_as_record=False)
        synsets  = np.squeeze(metadata['synsets'])
        wnids    = np.squeeze(np.array([s.WNID for s in synsets]))
        return wnids
    
    def make_input(self, num_per_sha=40000, fil_nam='train'):
        
        #################此处添加image文件路径##################
        imgs_dir = "Mybase/datasets/raw/ILSVRC/Data/CLS-LOC/" + fil_nam
        ###############此处添加annotation文件路径###############
        anns_dir = "Mybase/datasets/raw/ILSVRC/Annotations/CLS-LOC/" + fil_nam
        ##############此处添加tfrecords文件保存路径##############
        rcds_dir = "Mybase/datasets/" + fil_nam
        
        imgs_lst = []
        lbls_lst = []
        if fil_nam == "train":
            for key, value in self.cls_nam_to_cls_idx.items():
                #for ext in ['jpg', 'png', 'jpeg', 'JPG', 'JPEG']:
                    #imgs_cls = glob.glob(os.path.join(imgs_dir, key, '*.{}'.format(ext)))
                imgs_cls = glob.glob(os.path.join(imgs_dir, key, '*'))
                imgs_lst.extend(imgs_cls)
                lbls_lst.extend([value]*len(imgs_cls))
        else:
            #for ext in ['jpg', 'png', 'jpeg', 'JPG', 'JPEG']:
                #imgs_lst = glob.glob(os.path.join(imgs_dir, '*.{}'.format(ext)))
            imgs_lst = glob.glob(os.path.join(imgs_dir, '*'))
            lbls_lst = [-1] * len(imgs_lst)
            
        img_num = len(imgs_lst)
        print("The amount of images is %d!" %(img_num))
        print("The amount of files  is %d!" %(img_num//num_per_sha))
        idxs = np.arange(0, img_num)
        np.random.shuffle(idxs)
        imgs_lst = [imgs_lst[idx] for idx in idxs]
        lbls_lst = [lbls_lst[idx] for idx in idxs]
        #/data/ziyechen/ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_18.JPEG
            
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            sha_num = int(img_num/num_per_sha)
            if sha_num == 0:
                sha_num = 1
                num_per_sha = img_num
            else:
                num_per_sha = int(math.ceil(img_num/sha_num))

            for sha_idx in range(sha_num):
                '''
                #out_nam    = 'imagenet_%05d-of-%05d.tfrecord' % (sha_idx, sha_num)
                out_nam_cls = 'imagenet_cls_%s_%d.tfrecord' % (fil_nam, sha_idx)
                out_nam_det = 'imagenet_det_%s_%d.tfrecord' % (fil_nam, sha_idx)
                rcd_nam_cls = os.path.join(rcds_dir, out_nam_cls)
                rcd_nam_det = os.path.join(rcds_dir, out_nam_det)
                '''
                out_nam = 'imagenet_%s_%d.tfrecord' % (fil_nam, sha_idx)
                rcd_nam = os.path.join(rcds_dir, out_nam)
                
                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                '''
                with tf.python_io.TFRecordWriter(rcd_nam_cls, options=options) as writer_cls, \
                tf.python_io.TFRecordWriter(rcd_nam_det, options=options) as writer_det:
                '''
                with tf.python_io.TFRecordWriter(rcd_nam, options=options) as writer:
                    sta_idx = sha_idx * num_per_sha
                    end_idx = min((sha_idx + 1) * num_per_sha, img_num)
                    for i in range(sta_idx, end_idx):
                        if i % 100 == 0:
                            print("Converting image %d/%d shard %d" % (i + 1, img_num, sha_idx))
                        #读取图像
                        img_nam = imgs_lst[i]
                        img = cv2.imread(img_nam)
                        if type(img) != np.ndarray:
                            print("Failed to find image %s" %(imgs_lst[i]))
                            continue
                        img_hgt, img_wdh = img.shape[0], img.shape[1]
                        if img.size == img_hgt * img_wdh:
                            print ('Gray Image %s' %(imgs_lst[i]))
                            img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                            img_tmp[:, :, :] = img[:, :, np.newaxis]
                            img = img_tmp
                        img = img.astype(np.uint8)
                        assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                        img     = img[:, :, ::-1]
                        #读取标签
                        lbl     = lbls_lst[i]
                        ann     = img_nam.replace('Data', 'Annotations')
                        ann     = ann.split('.')
                        ann[-1] = 'xml'
                        ann     = '.'.join(ann)
                        
                        if not os.path.exists(ann):
                            gbxs = np.array([[0.0, 0.0, img_hgt-1.0, img_wdh-1.0, lbl]], dtype=np.float32)
                            det  = False
                        else:
                            tree = ET.parse(ann)
                            objs = tree.findall('object') #list
                            #img_siz = tree.find('size')
                            #img_hgt = float(img_siz.find('height').text)
                            #img_wdh = float(img_siz.find('width' ).text)
                            boxs = []
                            clss = []
                            for idx, obj in enumerate(objs):
                                box     = obj.find('bndbox')
                                box_ymn = float(box.find('ymin').text)
                                box_xmn = float(box.find('xmin').text)
                                box_ymx = float(box.find('ymax').text)
                                box_xmx = float(box.find('xmax').text)
                                cls     = self.cls_nam_to_cls_idx[obj.find('name').text.lower().strip()]
                                if lbl != -1:
                                    assert lbl == cls, "label is wrong!"
                                else:
                                    lbl = cls 
                                dif = obj.find('difficult')
                                dif = 0 if dif == None else int(dif.text)
                                if dif: cls *= -1
                                boxs.append([box_ymn, box_xmn, box_ymx, box_xmx])
                                clss.append([cls])
                                
                            boxs = np.asarray(boxs, dtype=np.float32)
                            clss = np.asarray(clss, dtype=np.float32)
                            boxs = bbox_clip_py(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
                            gbxs = np.concatenate([boxs, clss], axis=-1)
                        
                            if len(gbxs) == 0:
                                print("No gt_boxes in this image!")
                                continue
                            det = True
                        #写tfrecords
                        img_raw  = img.tostring()
                        gbxs_raw = gbxs.tostring()
                        example  = tf.train.Example(features=tf.train.Features(feature={
                            'image/image':  _bytes_feature(img_raw),
                            'image/height': _int64_feature(img_hgt),
                            'image/width':  _int64_feature(img_wdh),
                            'label/label':  _int64_feature(lbl),
                            'label/num_instances': _int64_feature(gbxs.shape[0]),  # N
                            'label/gt_boxes': _bytes_feature(gbxs_raw),  # of shape (N, 5), (ymin, xmin, ymax, xmax, classid)
                        }))
                        writer.write(example.SerializeToString())
                        '''
                        if det:
                            writer_det.write(example.SerializeToString())
                        else:
                            writer_cls.write(example.SerializeToString())
                        '''
    
    def resize_image_with_pad(self, img=None):

        #####################按最短边进行比例缩放######################
        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        if self.use_pad:
            leh_min = tf.minimum(img_hgt, img_wdh)
            leh_max = tf.maximum(img_hgt, img_wdh)
            leh_rat = tf.minimum(self.img_siz_min/leh_min, self.img_siz_max/leh_max)
            img_hgt = tf.cast(img_hgt*leh_rat, dtype=tf.int32)
            img_wdh = tf.cast(img_wdh*leh_rat, dtype=tf.int32)
            #对image操作后对boxs操作
            img     = tf.image.resize_images(img, [img_hgt, img_wdh], \
                                             method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
            ################如果最长边过长则按中心对称进行裁剪################
            #对image操作后对boxs操作
            img     = tf.image.resize_image_with_crop_or_pad(img, self.img_siz_max, self.img_siz_max)
        else:
            hgt_rat = self.img_siz_max / img_hgt
            wdh_rat = self.img_siz_max / img_wdh
            leh_rat = tf.stack([hgt_rat, wdh_rat], axis=0)
            leh_rat = tf.tile(leh_rat, [2])
            #对image操作后对boxs操作
            img     = tf.image.resize_images(img, [self.img_siz_max, self.img_siz_max], \
                                             method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        return img
    '''
    def distort_crop(self, img=None, gbxs=None):
        
        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        #boxs   = gbxs[:, :-1]
        #boxs   = bbox_clip(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        boxs    = tf.stack([0.0, 0.0, img_hgt-1.0, img_wdh-1.0], axis=0)
        
        if self.use_exp:
            exp_rat     = tf.random.uniform(shape=[], minval=1.1, maxval=self.exp_rat, dtype=tf.float32)
            #exp_rat    = self.exp_rat
            pad_hgt_all = tf.cast(img_hgt*(exp_rat-1.0), dtype=tf.int32)
            pad_wdh_all = tf.cast(img_wdh*(exp_rat-1.0), dtype=tf.int32)
            pad_hgt_fnt = tf.random.uniform(shape=[], minval=0, maxval=pad_hgt_all, dtype=tf.int32)
            pad_wdh_fnt = tf.random.uniform(shape=[], minval=0, maxval=pad_wdh_all, dtype=tf.int32)
            pad_hgt_bak = pad_hgt_all - pad_hgt_fnt
            pad_wdh_bak = pad_wdh_all - pad_wdh_fnt
            paddings    = [[pad_hgt_fnt, pad_hgt_bak], [pad_wdh_fnt, pad_wdh_bak], [0, 0]]
            img         = tf.pad(img, paddings, "CONSTANT", constant_values=0)
            pad_hgt_fnt = tf.cast(pad_hgt_fnt, dtype=tf.float32)
            pad_wdh_fnt = tf.cast(pad_wdh_fnt, dtype=tf.float32)
            beg         = tf.stack([pad_hgt_fnt, pad_wdh_fnt], axis=0)
            beg         = tf.tile(beg, [2])
            boxs        = boxs + beg #padding中boxs不会超出边界，不用clip
            img_hgt     = tf.cast(tf.shape(img)[0], dtype=tf.float32)
            img_wdh     = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        ########################crop the image randomly########################
        boxs_tmp = boxs / tf.stack([[img_hgt-1.0, img_wdh-1.0, img_hgt-1.0, img_wdh-1.0]], axis=0)
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(tf.shape(img), bounding_boxes=tf.expand_dims(boxs_tmp, 0), \
                                                   min_object_covered=self.min_object_covered, \
                                                   aspect_ratio_range=self.aspect_ratio_range, \
                                                   area_range=self.area_range, max_attempts=self.max_attempts, \
                                                   use_image_if_no_bounding_boxes=True)
        img = tf.slice(img, box_beg, box_siz)
        ###########resize image to the expected size with paddings############
        img = self.resize_image_with_pad(img)
        return img
    '''
    def distort_crop(self, img=None, img_siz_min=[256, 480]):
        img_siz_min = tf.random.uniform(shape=[], minval=img_siz_min[0], maxval=img_siz_min[1], dtype=tf.float32)
        img_hgt     = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh     = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        leh_min     = tf.minimum(img_hgt, img_wdh)
        leh_rat     = img_siz_min / leh_min
        img_hgt     = tf.cast(img_hgt*leh_rat, dtype=tf.int32)
        img_wdh     = tf.cast(img_wdh*leh_rat, dtype=tf.int32)
        img         = tf.image.resize_images(img, [img_hgt, img_wdh], \
                                             method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        img         = tf.image.random_crop(img, [self.img_siz_max, self.img_siz_max, 3])
        return img
    
    def preprocessing(self, img=None, gbxs=None):
        if self.mod_tra:
            #img= self.distort_crop(img, gbxs)
            #img= self.resize_image_with_pad(img)
            img = self.distort_crop(img, [256, 480])
        else:
            img = self.distort_crop(img, [256, 257])
            #img= self.resize_image_with_pad(img)
        return img
    
    def preprocessing1(self, imgs=None):
        imgs = tf.cast(imgs, dtype=tf.float32)
        if self.mod_tra:
            imgs = tf.image.random_flip_left_right(imgs)
            imgs = apply_with_random_selector(imgs, lambda x, order: distort_color(x, order), num_cases=4)
            imgs = imgs - self.img_avg
            imgs = imgs / 255.0
        else:
            imgs = imgs - self.img_avg
            imgs = imgs / 255.0
        return imgs
    
    def get_input(self):

        def parse_function(serialized_example):
            '''
            定长特征解析：tf.FixedLenFeature(shape, dtype, default_value)
                        shape：可当 reshape 来用，如 vector 的 shape 从 (3,) 改动成了 (1,3)。
                               注：如果写入的 feature 使用了. tostring() 其 shape 就是 ()
                        dtype：必须是 tf.float32， tf.int64， tf.string 中的一种。
                        default_value：feature 值缺失时所指定的值。
            不定长特征解析：tf.VarLenFeature(dtype)
                          注：可以不明确指定 shape，但得到的 tensor 是 SparseTensor。
                              变长的tensor转化为string后，shape是固定的，shape=[]，所以可以用tf.FixedLenFeature进行解析
            '''
            parsed_example = tf.parse_single_example(
                serialized_example, 
                features = {
                    'image/image':         tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'image/height':        tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'image/width':         tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/label':         tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/num_instances': tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/gt_boxes':      tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    #'matrix':      tf.VarLenFeature(dtype=dtype('float32')), 
                    #'matrix_shape':tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                }
            )
            img_hgt = tf.cast(parsed_example['image/height'],       tf.int32)
            img_wdh = tf.cast(parsed_example['image/width'],        tf.int32)
            lbl     = tf.cast(parsed_example['label/label'],        tf.int32)
            gbx_num = tf.cast(parsed_example['label/num_instances'],tf.int32)

            img     = tf.decode_raw(parsed_example['image/image'],    tf.uint8  )
            gbxs    = tf.decode_raw(parsed_example['label/gt_boxes'], tf.float32)
            img     = tf.reshape(img,  [img_hgt, img_wdh, 3])
            gbxs    = tf.reshape(gbxs, [gbx_num, 5])
            img     = self.preprocessing(img, gbxs)
            parsed_example = {
                'image/image':  img,
                'image/height': img_hgt,
                'image/width':  img_wdh,
                'label/label':  lbl,
            }
            #parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
            #parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])
            return parsed_example
        
        fil_pat  = os.path.join(self.dat_dir, 'imagenet', '*.tfrecord')
        dataset  = tf.data.Dataset.list_files(file_pattern=fil_pat, shuffle=True, seed=None)
        dataset  = dataset.prefetch(buffer_size=self.num_readers)
        dataset  = dataset.shuffle(buffer_size=self.fil_num+self.num_readers, seed=None, reshuffle_each_iteration=True)
        dataset  = dataset.apply(tf.data.experimental.\
                                 parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                                                     cycle_length=self.num_readers, \
                                                     block_length=10, sloppy=True, \
                                                     buffer_output_elements=10, \
                                                     prefetch_input_elements=10))
        dataset  = dataset.prefetch(buffer_size=self.bat_siz_all)
        dataset  = dataset.map(parse_function, num_parallel_calls=self.num_threads)
        dataset  = dataset.apply(tf.data.experimental.\
                                 shuffle_and_repeat(buffer_size=self.capacity, count=self.epc_num, seed=None))
        dataset  = dataset.batch(batch_size=self.bat_siz_all, drop_remainder=True)
        #dataset = dataset.apply(tf.data.experimental.\
        #                        map_and_batch(parse_function, batch_size=self.bat_siz, num_parallel_batches=None, \
        #                                      drop_remainder=False, num_parallel_calls=self.num_threads))
        dataset  = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_one_shot_iterator()
        example  = iterator.get_next()
        imgs_lst = tf.split(example['image/image'], self.gpu_num, axis=0)
        lbls_lst = tf.split(example['label/label'], self.gpu_num, axis=0)
        return imgs_lst, lbls_lst
    """
    def get_input(self):
        #创建文件列表，并通过文件列表创建输入文件队列。
        #在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中
        #文件列表应该包含所有提供训练数据的TFRecord文件
        filename = os.path.join(self.dat_dir, 'imagenet', '*.tfrecord')
        files    = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(files, shuffle=True, capacity=1000)

        #解析TFRecord文件里的数据
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)
        
        parsed_example = tf.parse_single_example(
            serialized_example,
            features = {
                'image/image':         tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                'image/height':        tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                'image/width':         tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                'label/label':         tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                'label/num_instances': tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                'label/gt_boxes':      tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
            }
        )

        img_hgt = tf.cast(parsed_example['image/height'],       tf.int32)
        img_wdh = tf.cast(parsed_example['image/width'],        tf.int32)
        lbl     = tf.cast(parsed_example['label/label'],        tf.int32)
        gbx_num = tf.cast(parsed_example['label/num_instances'],tf.int32)

        img     = tf.decode_raw(parsed_example['image/image'],    tf.uint8  )
        gbxs    = tf.decode_raw(parsed_example['label/gt_boxes'], tf.float32)
        img     = tf.reshape(img,  [img_hgt, img_wdh, 3])
        gbxs    = tf.reshape(gbxs, [gbx_num, 5])

        img     = self.preprocessing(img, gbxs)
        img     = tf.reshape(img,  [self.img_siz_max, self.img_siz_max, 3])
        
        #tf.train.shuffle_batch_join
        imgs, lbls = tf.train.shuffle_batch(tensors=[img, lbl], batch_size=self.bat_siz_all, num_threads=self.num_threads, \
                                            capacity=self.capacity, min_after_dequeue=self.min_after_dequeue)
        imgs_lst = tf.split(imgs, self.gpu_num, axis=0)
        lbls_lst = tf.split(lbls, self.gpu_num, axis=0)
        return imgs_lst, lbls_lst
    """
    def display_instances(self, img=None, lbl=None, img_hgt=None, img_wdh=None):
        
        _, ax = plt.subplots(1, figsize=self.figsize)
            
        #img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        #Show area outside image boundaries.
        #ax.set_ylim(img_hgt + 10, -10)
        #ax.set_xlim(-10, img_wdh + 10)
        ax.axis('off')
        ax.set_title(str(lbl))
        
        img = img * 255.0
        img = img + self.img_avg
        img = np.clip(img, 0.0, 225.0)
        img = img.astype(dtype=np.uint8, copy=False)
        ax.imshow(img)
        
        plt.show()
        plt.close()
        
    def get_input_test(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs_lst, lbls_lst = self.get_input()
        imgs = tf.concat(imgs_lst, axis=0)
        lbls = tf.concat(lbls_lst, axis=0)

        with tf.Session() as sess:
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            imgs_kep, lbls_kep = sess.run([imgs, lbls])
            
            for i in range(self.bat_siz*self.gpu_num):
                img_tmp = imgs_kep[i]
                lbl_tmp = lbls_kep[i]
                self.display_instances(img_tmp, lbl_tmp)


from Mybase.datasets.raw.coco.PythonAPI.pycocotools.coco import *

class GeneratorForCOCO(object):

    def __init__(self, mod_tra=True, dat_dir=None, bat_siz=3, epc_num=20, min_after_dequeue=20, \
                 gpu_lst='0', fil_num=32, tst_shw=True, tst_sav=False):
        
        self.mod_tra            = mod_tra
        self.use_pad            = True
        self.use_exp            = False
        self.exp_rat            = 2.0
        self.img_avg            = np.array([123.7, 116.8, 103.9], dtype=np.float32)
        self.img_siz_min        = 800  #800   #700  #400
        self.img_siz_max        = 1025 #1025  #897  #513
        self.box_siz_min        = 5
        self.box_isc_min        = 0.5
        self.box_msk_siz        = [126, 126]
        self.msk_min            = 0.5
        ############for crop###########
        self.min_object_covered = 0.5
        self.aspect_ratio_range = (0.5, 2.0)
        self.area_range         = (0.1, 1.0)
        self.max_attempts       = 200
        
        self.dat_dir            = dat_dir
        self.bat_siz            = bat_siz
        self.epc_num            = epc_num
        self.min_after_dequeue  = min_after_dequeue
        self.gpu_lst            = gpu_lst
        self.gpu_num            = len(self.gpu_lst.split(','))
        
        self.mdl_dev            = '/cpu:%d' if self.gpu_num == 0 else '/gpu:%d'
        self.gpu_num            = 1         if self.gpu_num == 0 else self.gpu_num
        self.fil_num            = fil_num
        self.num_readers        = 2 
        self.num_threads        = 20
        self.bat_siz_all        = self.bat_siz * self.gpu_num
        self.capacity           = self.min_after_dequeue + 3 * self.bat_siz_all

        
        self.max_num            = 100
        self.tst_shw            = tst_shw
        self.tst_sav            = tst_sav
        
        self.cls_nams = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                         'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                         'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                         'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                         'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                         'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                         'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                         'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                         'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                         'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.cls_num  = len(self.cls_nams)
        
        self.cls_idx_to_rel_idx = \
            {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
             18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
             35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
             50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
             64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
             82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
    
        self.rel_idx_to_cls_idx = \
            {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
             17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34,
             31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49,
             45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63,
             59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81,
             73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
        
        self.cls_idx_to_cls_nam = dict(zip(range(self.cls_num), self.cls_nams))
        self.cls_nam_to_cls_idx = dict(zip(self.cls_nams, range(self.cls_num)))
        
        ########for test######
        if self.tst_dir != None:
            self.imgs_lst_tst = []
            for ext in ['jpg', 'png', 'jpeg', 'JPG']:
                self.imgs_lst_tst.extend(glob.glob(os.path.join(self.tst_dir, '*.{}'.format(ext))))
            self.anns_lst_tst = []
            self.gbxs_lst_tst = [] #暂不支持use_gbx==True
            self.img_num_tst  = len(self.imgs_lst_tst)
            self.get_idx      = 0
        
        ########for show######
        self.title        = ''
        self.figsize      = (15, 15)
    
    def get_coco_masks(self, coco, img_idx, img_hgt, img_wdh, img_nam):

        ann_idxs = coco.getAnnIds(imgIds=[img_idx], iscrowd=None)
        if len(ann_idxs) == 0:
            print ('There is no annotations for %s' % img_nam)
            return None, None, None

        anns = coco.loadAnns(ann_idxs)

        boxs = []
        clss = []
        inss = []
        sems = np.zeros((img_hgt, img_wdh), dtype=np.uint8)
        for ann in anns:
            cls = self.cls_idx_to_rel_idx[ann['category_id']]
            ins = coco.annToMask(ann) # zero one mask, 此处ann为一个字典
            ins = ins.astype(dtype=np.uint8, copy=False)
            if ins.max() < 1:
                continue
            if ann['iscrowd']:
                cls *= -1
                if ins.shape[0]!=img_hgt or ins.shape[1]!=img_wdh:
                    ins = np.ones([img_hgt, img_wdh], dtype=np.uint8)
            assert ins.shape[0]==img_hgt and ins.shape[1]==img_wdh, 'image %s and ann %s do not match' % (img_idx, ann)
            boxs.append(ann['bbox'])
            clss.append(cls)
            inss.append(ins)
            sem         = ins * cls
            sems[sem>0] = sem[sem>0]
        boxs = np.asarray(boxs, dtype=np.float32)
        clss = np.asarray(clss, dtype=np.float32)
        inss = np.asarray(inss, dtype=np.uint8  )

        if boxs.shape[0] <= 0:
            print('There is no annotations for %s' % img_nam)
            return None, None

        boxs[:, 2] = boxs[:, 0] + boxs[:, 2]
        boxs[:, 3] = boxs[:, 1] + boxs[:, 3]
        boxs = np.stack([boxs[:, 1], boxs[:, 0], boxs[:, 3], boxs[:, 2]], axis=-1)#ymin, xmin, ymax, xmax
        gbxs = np.concatenate([boxs, clss[:, np.newaxis]], axis=-1)

        if inss.shape[0] != gbxs.shape[0]:
            print('Shape Error for %s' % img_nam)
            return None, None
        return gbxs, inss, sems
    
    def make_input(self, num_per_sha=500000):
    
        ##############此处添加image文件路径##############
        imgs_dir = "Mybase/datasets/raw/coco"
        ##############此处添加annotation文件路径##############
        anns_dir = "Mybase/datasets/raw/coco/annotations"
        ##############此处添加tfrecords文件保存路径##############
        rcds_dir = "Mybase/datasets"

        spl_nams = ['minival2014']
        #spl_nams = ['train2014', 'valminusminival2014', 'train2017', 'val2017']
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            imgs = []
            cocs = []
            for coc_idx, spl_nam in enumerate(spl_nams):
                ann_fil = os.path.join(anns_dir, 'instances_%s.json' % (spl_nam))
                cocs.append(COCO(ann_fil))
                imgs.extend([(coc_idx, img_idx, spl_nam, cocs[coc_idx].imgs[img_idx]) for img_idx in cocs[coc_idx].imgs])
            img_num = int(len(imgs))
            sha_num = int(img_num/num_per_sha)
            print('The dataset has %d images' %(img_num))
            np.random.shuffle(imgs)

            if sha_num == 0:
                sha_num = 1
                num_per_sha = img_num
            else:
                num_per_sha = int(math.ceil(img_num/sha_num))

            for sha_idx in range(sha_num):
                out_nam = 'coco_%05d-of-%05d.tfrecord' % (sha_idx, sha_num)
                rcd_nam = os.path.join(rcds_dir, out_nam)

                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(rcd_nam, options=options) as writer:

                    sta_idx = sha_idx * num_per_sha
                    end_idx = min((sha_idx+1)*num_per_sha, img_num)
                    for i in range(sta_idx, end_idx):
                        if i % 50 == 0:
                            print("Converting image %d shard %d" % (i+1, sha_idx))
                        
                        coc_idx = imgs[i][0]
                        img_idx = imgs[i][1]
                        spl_nam = imgs[i][2]
                        img_nam = imgs[i][3]['file_name']
                        #split = img_name.split('_')[1]
                        if spl_nam == 'valminusminival2014' or spl_nam == 'minival2014':
                            spl_nam = 'val2014'
                        img_nam = os.path.join(imgs_dir, spl_nam, img_nam)
                        img_hgt, img_wdh = imgs[i][3]['height'], imgs[i][3]['width']

                        gbxs, gmk_inss, gmk_sems = self.get_coco_masks(cocs[coc_idx], img_idx, img_hgt, img_wdh, img_nam)
                        if not isinstance(gbxs, np.ndarray):
                            continue
                        img = cv2.imread(img_nam)
                        if not isinstance(img,  np.ndarray):
                            print("Failed to find image %s" %(img_nam))
                            continue
                        img_hgt, img_wdh = img.shape[0], img.shape[1]
                        if img.size == img_hgt * img_wdh:
                            print ('Gray Image %s' %(img_nam))
                            img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                            img_tmp[:, :, :] = img[:, :, np.newaxis]
                            img = img_tmp
                        img = img.astype(np.uint8)
                        assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                        img = img[:, :, ::-1]

                        img_raw      = img.tostring()
                        gbxs_raw     = gbxs.tostring()
                        gmk_inss_raw = gmk_inss.tostring()
                        gmk_sems_raw = gmk_sems.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image/img_id': _int64_feature(img_idx),
                            'image/image':  _bytes_feature(img_raw),
                            'image/height': _int64_feature(img_hgt),
                            'image/width':  _int64_feature(img_wdh),

                            'label/num_instances': _int64_feature(gbxs.shape[0]),  # N
                            'label/gt_boxes':      _bytes_feature(gbxs_raw),       # (N, ymin, xmin, ymax, xmax, classid)
                            'label/gt_mask_inss':  _bytes_feature(gmk_inss_raw),   # (N, H, W)
                            'label/gt_mask_sems':  _bytes_feature(gmk_sems_raw)    # (H, W)
                        }))
                        writer.write(example.SerializeToString())
    
    def resize_image_with_pad(self, img=None, gbxs=None, gmk_inss=None, gmk_sems=None):
        #没有crop，不需要对gmk_inss操作
        img_hgt  = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh  = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        boxs     = gbxs[:, :-1]
        clss     = gbxs[:,  -1]
        boxs     = bbox_clip(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        gmk_sems = tf.expand_dims(gmk_sems, axis=-1)
        
        if self.use_pad:
            img      = tf.image.resize_image_with_pad(img,      self.img_siz, self.img_siz, \
                                                      method=tf.image.ResizeMethod.BILINEAR)
            gmk_sems = tf.image.resize_image_with_pad(gmk_sems, self.img_siz, self.img_siz, \
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #对image操作后对boxs操作
            leh_min  = tf.minimum(img_hgt, img_wdh)
            leh_max  = tf.maximum(img_hgt, img_wdh)
            leh_rat  = tf.minimum(self.img_siz/leh_min, self.img_siz/leh_max)
            boxs     = boxs * leh_rat
            #将实际坐标加上偏移值
            img_hgt  = img_hgt * leh_rat
            img_wdh  = img_wdh * leh_rat
            pad_hgt  = self.img_siz - img_hgt
            pad_wdh  = self.img_siz - img_wdh
            pad_top  = tf.round(pad_hgt/2.0)
            pad_lft  = tf.round(pad_wdh/2.0)
            beg      = tf.stack([pad_top, pad_lft], axis=0)
            beg      = tf.tile(beg, [2])
            boxs     = boxs + beg #该boxs在原真实图片内，不需要clip
            img_wdw  = tf.stack([pad_top, pad_lft, pad_top+img_hgt-1, pad_lft+img_wdh-1], axis=0)
        else:
            img      = tf.image.resize_images(img,      [self.img_siz, self.img_siz], \
                                              method=tf.image.ResizeMethod.BILINEAR,         align_corners=True)
            gmk_sems = tf.image.resize_images(gmk_sems, [self.img_siz, self.img_siz], \
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
            #对image操作后对boxs操作
            hgt_rat  = self.img_siz / img_hgt
            wdh_rat  = self.img_siz / img_wdh
            leh_rat  = tf.stack([hgt_rat, wdh_rat], axis=0)
            leh_rat  = tf.tile(leh_rat, [2])
            boxs     = boxs * leh_rat
            img_wdw  = tf.constant([0, 0, self.img_siz-1, self.img_siz-1], dtype=tf.int32)
        #合成gt_boxes
        clss     = tf.expand_dims(clss, axis=-1)
        gbxs     = tf.concat([boxs, clss], axis=-1)
        gbx_tmp  = tf.zeros(shape=[1, 5], dtype=tf.float32) #防止没有一个gt_box
        gbxs     = tf.concat([gbxs, gbx_tmp], axis=0)
        ins_tmp  = tf.zeros(shape=[1]+self.box_msk_siz, dtype=tf.float32)
        gmk_inss = tf.concat([gmk_inss, ins_tmp], axis=0)
        gmk_sems = tf.squeeze(gmk_sems, axis=[-1])
        return img, gbxs, gmk_inss, gmk_sems, img_wdw
    
    def distort_crop(self, img=None, gbxs=None, gmk_inss=None, gmk_sems=None):
        #gmk_inss --> (M, h, w)
        img_hgt  = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh  = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        boxs     = gbxs[:, :-1]
        clss     = gbxs[:,  -1]
        boxs     = bbox_clip(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        
        if self.use_exp:
            exp_rat  = tf.random.uniform(shape=[], minval=1.1, maxval=self.exp_rat, dtype=tf.float32)
            #exp_rat = self.exp_rat
            pad_hgt  = tf.cast(img_hgt*(exp_rat-1.0), dtype=tf.int32)
            pad_wdh  = tf.cast(img_wdh*(exp_rat-1.0), dtype=tf.int32)
            pad_top  = tf.random.uniform(shape=[], minval=0, maxval=pad_hgt_all, dtype=tf.int32)
            pad_lft  = tf.random.uniform(shape=[], minval=0, maxval=pad_wdh_all, dtype=tf.int32)
            pad_btm  = pad_hgt - pad_top
            pad_rgt  = pad_wdh - pad_lft
            paddings = [[pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0]]
            img      = tf.pad(img,      paddings, "CONSTANT", constant_values=0)
            paddings = [[pad_top, pad_btm], [pad_lft, pad_rgt]]
            gmk_sems = tf.pad(gmk_sems, paddings, "CONSTANT", constant_values=0)
            #padding中boxs不会超出边界，不用对boxs和gmk_inss操作
            pad_top  = tf.cast(pad_top, dtype=tf.float32)
            pad_lft  = tf.cast(pad_lft, dtype=tf.float32)
            beg      = tf.stack([pad_top, pad_lft], axis=0)
            beg      = tf.tile(beg, [2])
            boxs     = boxs + beg
            img_hgt  = tf.cast(tf.shape(img)[0], dtype=tf.float32)
            img_wdh  = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        ########################crop the image randomly########################
        ncw_idxs = tf.where(clss>0)
        boxs_tmp = tf.gather_nd(boxs, ncw_idxs)
        boxs_tmp = boxs_tmp / tf.stack([img_hgt-1.0, img_wdh-1.0, img_hgt-1.0, img_wdh-1.0], axis=0)
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(tf.shape(img), bounding_boxes=tf.expand_dims(boxs_tmp, 0), \
                                                   min_object_covered=self.min_object_covered, \
                                                   aspect_ratio_range=self.aspect_ratio_range, \
                                                   area_range=self.area_range, max_attempts=self.max_attempts, \
                                                   use_image_if_no_bounding_boxes=True)
        #对image操作后对boxs操作
        img      = tf.slice(img,      box_beg,      box_siz)
        gmk_sems = tf.slice(gmk_sems, box_beg[:-1], box_siz[:-1])
        
        img_hgt  = tf.cast(box_siz[0], dtype=tf.float32)
        img_wdh  = tf.cast(box_siz[1], dtype=tf.float32)
        #将实际坐标加上偏移值
        beg      = tf.cast(box_beg[0:2], dtype=tf.float32)
        beg      = tf.tile(beg, [2])
        boxs_tmp = boxs - beg
        #防止box超出边界
        boxs     = bbox_clip(boxs_tmp, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        #剔除过小的boxs和msks
        box_iscs = bbox_intersects1(boxs_tmp, boxs)
        idxs     = tf.where(box_iscs<self.box_isc_min)
        clss     = tensor_update(clss, idxs, -1)
        #idxs    = tf.where(box_iscs>=self.box_isc_min)
        #boxs    = tf.gather_nd(boxs,     idxs)
        #clss    = tf.gather_nd(clss,     idxs)
        #gmk_inss= tf.gather_nd(gmk_inss, idxs)
        clss     = tf.expand_dims(clss, axis=-1)
        gbxs     = tf.concat([boxs, clss], axis=-1)
        #计算crop后的gmk_inss
        begs     = tf.stack([boxs_tmp[:, 0], boxs_tmp[:, 1], boxs_tmp[:, 0], boxs_tmp[:, 1]], axis=-1)
        lehs     = tf.stack([boxs_tmp[:, 2]-boxs_tmp[:, 0], boxs_tmp[:, 3]-boxs_tmp[:, 1],
                             boxs_tmp[:, 2]-boxs_tmp[:, 0], boxs_tmp[:, 3]-boxs_tmp[:, 1]], axis=-1)
        boxs_tmp = (boxs - begs) / lehs
        idxs     = tf.range(tf.shape(boxs_tmp)[0])
        gmk_inss = tf.expand_dims(gmk_inss, axis=-1) #(M, H, W, 1)
        gmk_inss = tf.image.crop_and_resize(gmk_inss, boxs_tmp, idxs, self.box_msk_siz, method='bilinear') #(M, 256, 256)
        gmk_inss = tf.squeeze(gmk_inss, axis=[-1])   #(M, H, W)
        
        ###########resize image to the expected size with paddings############
        img, gbxs, gmk_inss, gmk_sems, img_wdw = self.resize_image_with_pad(img, gbxs, gmk_inss, gmk_sems)
        return img, gbxs, gmk_inss, gmk_sems, img_wdw
    
    def preprocessing(self, img=None, gbxs=None, gmk_inss=None):
        img_shp  = tf.shape(img)                     #[H, W, C]
        box_leh  = tf.cast(tf.tile(img_shp[:2], [2]), dtype=tf.float32) #[H, W, H, W]
        boxs_tmp = gbxs[:,:-1] / box_leh             #(M, 4)
        idxs     = tf.range(tf.shape(boxs_tmp)[0])
        gmk_inss = tf.expand_dims(gmk_inss, axis=-1) #(M, H, W, 1)
        gmk_inss = tf.image.crop_and_resize(gmk_inss, boxs_tmp, idxs, self.box_msk_siz, method='bilinear') #(M, 256, 256)
        gmk_inss = tf.squeeze(gmk_inss, axis=[-1])   #(M, H, W)
        return gmk_inss
    
    def preprocessing0(self, elms=None):
        img, gbxs, gmk_inss, gmk_sems, gbx_num, img_hgt, img_wdh = elms
        img      = img[:img_hgt, :img_wdh, :]
        gbxs     = gbxs[:gbx_num, :]
        gmk_inss = gmk_inss[:gbx_num, :, :]
        gmk_sems = gmk_sems[:img_hgt, :img_wdh]
        if self.mod_tra:
            img, gbxs, gmk_inss, gmk_sems, img_wdw = self.distort_crop(img, gbxs, gmk_inss, gmk_sems)
        else:
            img, gbxs, gmk_inss, gmk_sems, img_wdw = self.resize_image_with_pad(img, gbxs, gmk_inss, gmk_sems)
        gbx_num  = tf.shape(gbxs)[0]
        paddings = [[0, self.max_num-gbx_num], [0, 0]]
        gbxs     = tf.pad(gbxs,     paddings, "CONSTANT", constant_values=0)
        paddings = [[0, self.max_num-gbx_num], [0, 0], [0, 0]]
        gmk_inss = tf.pad(gmk_inss, paddings, "CONSTANT", constant_values=0)
        return img, gbxs, gmk_inss, gmk_sems, gbx_num, img_wdw
    
    def preprocessing1(self, imgs=None, gbxs=None, gmk_inss=None, gmk_sems=None, gbx_nums=None, img_hgts=None, img_wdhs=None):
        imgs     = tf.cast(imgs,     dtype=tf.float32)
        gmk_sems = tf.cast(gmk_sems, dtype=tf.int32  )
        elms     = [imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_hgts, img_wdhs]
        
        if self.mod_tra:
            imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                tf.map_fn(self.preprocessing0, elms, \
                          dtype=(tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32),
                          parallel_iterations=self.bat_siz, back_prop=False, swap_memory=False, infer_shape=True)
            ######################随机翻转##########################
            sig          = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
            #imgs        = tf.image.random_flip_left_right(imgs)  #(N, H, W, C)
            imgs_flr     = tf.image.flip_left_right(imgs)         #(N, H, W, C)
            gmk_sems_flr = tf.expand_dims(gmk_sems, axis=-1)      #(N, H, W, 1)
            gmk_sems_flr = tf.image.flip_left_right(gmk_sems_flr) #(N, H, W, 1)
            gmk_sems_flr = tf.squeeze(gmk_sems_flr, axis=[-1])    #(N, H, W)
            gmk_inss_flr = tf.transpose(gmk_inss,     [0,2,3,1])  #(N, H, W, M)
            gmk_inss_flr = tf.image.flip_left_right(gmk_inss_flr) #(N, H, W, M)
            gmk_inss_flr = tf.transpose(gmk_inss_flr, [0,3,1,2])  #(N, M, H, W)
            gbxs_flr     = tf.stack([gbxs[:,:,0], self.img_siz-1.0-gbxs[:,:,3], \
                                     gbxs[:,:,2], self.img_siz-1.0-gbxs[:,:,1], gbxs[:,:,4]], axis=-1)
            img_wdws_flr = tf.stack([img_wdws[:,0], self.img_siz-1.0-img_wdws[:,3], \
                                     img_wdws[:,2], self.img_siz-1.0-img_wdws[:,1]], axis=-1)
            imgs         = tf.cond(sig<0.5, lambda: imgs_flr,     lambda: imgs    )
            gmk_sems     = tf.cond(sig<0.5, lambda: gmk_sems_flr, lambda: gmk_sems)
            gmk_inss     = tf.cond(sig<0.5, lambda: gmk_inss_flr, lambda: gmk_inss)
            gbxs         = tf.cond(sig<0.5, lambda: gbxs_flr,     lambda: gbxs    )
            img_wdws     = tf.cond(sig<0.5, lambda: img_wdws_flr, lambda: img_wdws)
            #####################光学畸变###########################
            imgs = apply_with_random_selector(imgs, lambda x, order: distort_color(x, order), num_cases=4)
            imgs = imgs - self.img_avg
            imgs = imgs / 255.0
        else:
            imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                tf.map_fn(self.preprocessing0, elms, \
                          dtype=(tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32),
                          parallel_iterations=self.bat_siz, back_prop=False, swap_memory=False, infer_shape=True)
            imgs = imgs - self.img_avg
            imgs = imgs / 255.0
        return imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws
    
    def get_input(self):

        def parse_function(serialized_example):
            '''
            定长特征解析：tf.FixedLenFeature(shape, dtype, default_value)
                        shape：可当 reshape 来用，如 vector 的 shape 从 (3,) 改动成了 (1,3)。
                               注：如果写入的 feature 使用了. tostring() 其 shape 就是 ()
                        dtype：必须是 tf.float32， tf.int64， tf.string 中的一种。
                        default_value：feature 值缺失时所指定的值。
            不定长特征解析：tf.VarLenFeature(dtype)
                          注：可以不明确指定 shape，但得到的 tensor 是 SparseTensor。
                              变长的tensor转化为string后，shape是固定的，shape=[]，所以可以用tf.FixedLenFeature进行解析
            '''
            parsed_example = tf.parse_single_example(
                serialized_example, 
                features = {
                    'image/image':         tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'image/height':        tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'image/width':         tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/num_instances': tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/gt_boxes':      tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'label/gt_mask_inss':  tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'label/gt_mask_sems':  tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    #'matrix':             tf.VarLenFeature(dtype=dtype('float32')), 
                    #'matrix_shape':       tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                }
            )
            img_hgt  = tf.cast(parsed_example['image/height'],        tf.int32)
            img_wdh  = tf.cast(parsed_example['image/width'],         tf.int32)
            gbx_num  = tf.cast(parsed_example['label/num_instances'], tf.int32)

            img      = tf.decode_raw(parsed_example['image/image'],        tf.uint8  )
            gbxs     = tf.decode_raw(parsed_example['label/gt_boxes'],     tf.float32)
            gmk_inss = tf.decode_raw(parsed_example['label/gt_mask_inss'], tf.uint8  )
            gmk_sems = tf.decode_raw(parsed_example['label/gt_mask_sems'], tf.uint8  )
            
            img      = tf.reshape(img,      [img_hgt, img_wdh, 3])
            gbxs     = tf.reshape(gbxs,     [gbx_num, 5])
            gmk_inss = tf.reshape(gmk_inss, [gbx_num, img_hgt, img_wdh])
            gmk_sems = tf.reshape(gmk_sems, [img_hgt, img_wdh])
            
            gmk_inss = self.preprocessing(img, gbxs, gmk_inss)
            
            parsed_example = {
                'image/image':         img,
                'image/height':        img_hgt,
                'image/width':         img_wdh,
                'label/num_instances': gbx_num,
                'label/gt_boxes':      gbxs,
                'label/gt_mask_inss':  gmk_inss,
                'label/gt_mask_sems':  gmk_sems
            }
            #parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
            #parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])
            return parsed_example
        
        fil_pat  = os.path.join(self.dat_dir, 'coco', '*.tfrecord')
        dataset  = tf.data.Dataset.list_files(file_pattern=fil_pat, shuffle=True, seed=None)
        dataset  = dataset.prefetch(buffer_size=self.num_readers)
        dataset  = dataset.shuffle(buffer_size=self.fil_num+self.num_readers, seed=None, reshuffle_each_iteration=True)
        dataset  = dataset.apply(tf.data.experimental.\
                                 parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                                                     cycle_length=self.num_readers, \
                                                     block_length=10, sloppy=True, \
                                                     buffer_output_elements=10, \
                                                     prefetch_input_elements=10))
        dataset  = dataset.prefetch(buffer_size=self.bat_siz_all)
        dataset  = dataset.map(parse_function, num_parallel_calls=self.num_threads)
        dataset  = dataset.apply(tf.data.experimental.\
                                 shuffle_and_repeat(buffer_size=self.capacity, count=self.epc_num, seed=None))
        dataset  = dataset.padded_batch(batch_size=self.bat_siz_all, \
                                        padded_shapes={'image/image':         [-1, -1, 3],
                                                       'image/height':        [],
                                                       'image/width':         [],
                                                       'label/num_instances': [],
                                                       'label/gt_boxes':      [-1, 5],
                                                       'label/gt_mask_inss':  [-1]+self.box_msk_siz,
                                                       'label/gt_mask_sems':  [-1, -1]},
                                        padding_values=None, drop_remainder=True)
        #dataset = dataset.batch(batch_size=self.bat_siz_all, drop_remainder=True)
        #dataset = dataset.apply(tf.data.experimental.\
        #                        map_and_batch(parse_function, batch_size=self.bat_siz_all, num_parallel_batches=None, \
        #                                      drop_remainder=True, num_parallel_calls=self.num_threads))
        dataset  = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_one_shot_iterator()
        example  = iterator.get_next()
        imgs_lst     = tf.split(example['image/image'],         self.gpu_num, axis=0)
        gbxs_lst     = tf.split(example['label/gt_boxes'],      self.gpu_num, axis=0)
        gmk_inss_lst = tf.split(example['label/gt_mask_inss'],  self.gpu_num, axis=0)
        gmk_sems_lst = tf.split(example['label/gt_mask_sems'],  self.gpu_num, axis=0)
        gbx_nums_lst = tf.split(example['label/num_instances'], self.gpu_num, axis=0)
        img_hgts_lst = tf.split(example['image/height'],        self.gpu_num, axis=0)
        img_wdhs_lst = tf.split(example['image/width'],         self.gpu_num, axis=0)
        return imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, img_hgts_lst, img_wdhs_lst
    
    def get_input2(self):
        
        imgs_lst_tst = []
        for ext in ['jpg', 'png', 'jpeg', 'JPG']:
            imgs_lst_tst.extend(glob.glob(os.path.join(self.dat_dir, '*.{}'.format(ext))))
        print('There are {:d} images to test!'.format(len(imgs_lst_tst)))
        
        def data_generator():
            for img_fil in imgs_lst_tst:
                assert os.path.exists(img_fil), 'The image file does not exist: {:s}'.format(img_fil)
                img     = cv2.imread(img_fil)
                img_nam = img_fil.split('/')[-1]
                img_hgt = img.shape[0]
                img_wdh = img.shape[1]
                if img.size == img_hgt * img_wdh:
                    print ('Gray Image %s' %(imgs_lst[i]))
                    img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                    img_tmp[:, :, :] = img[:, :, np.newaxis]
                    img = img_tmp
                img = img.astype(np.uint8)
                assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                img = img[:, :, ::-1]
                generated_example = {
                    'image/image':  img,
                    'image/height': img_hgt,
                    'image/width':  img_wdh,
                    'image/name':   img_nam
                }
                yield generated_example
        
        def parse_function(generated_example):
            img     = generated_example['image/image']
            img_hgt = generated_example['image/height']
            img_wdh = generated_example['image/width']
            img_nam = generated_example['image/name']
            
            parsed_example = {
                'image/image':         img,
                'image/height':        img_hgt,
                'image/width':         img_wdh,
                'image/name':          img_nam,
                'label/num_instances': 1,
                'label/gt_boxes':      tf.zeros(shape=[1, 5],               dtype=tf.float32),
                'label/gt_mask_inss':  tf.zeros(shape=[1]+self.box_msk_siz, dtype=tf.float32),
                'label/gt_mask_sems':  tf.zeros(shape=[img_hgt, img_wdh],   dtype=tf.uint8)
            }
            #parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
            #parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])
            return parsed_example
        
        dataset = tf.data.Dataset.from_generator(data_generator, 
                                                 output_types ={'image/image': tf.uint8, 'image/height': tf.int32,
                                                                'image/width': tf.int32, 'image/name':   tf.string}, \
                                                 output_shapes={'image/image': [None, None, 3], 'image/height': [],
                                                                'image/width': [],              'image/name':   []}, args=None)
        dataset  = dataset.repeat(count=1)
        dataset  = dataset.prefetch(buffer_size=self.bat_siz_all)
        dataset  = dataset.map(parse_function, num_parallel_calls=self.num_threads)
        #dataset = dataset.batch(batch_size=self.bat_siz_all, drop_remainder=False)
        #dataset = dataset.apply(tf.data.experimental.\
        #                        map_and_batch(parse_function, batch_size=self.bat_siz_all, num_parallel_batches=None, \
        #                                      drop_remainder=False, num_parallel_calls=self.num_threads))
        dataset  = dataset.padded_batch(batch_size=self.bat_siz_all, \
                                        padded_shapes={'image/image':         [-1, -1, 3],
                                                       'image/height':        [],
                                                       'image/width':         [],
                                                       'image/name':          [],
                                                       'label/num_instances': [],
                                                       'label/gt_boxes':      [-1, 5],
                                                       'label/gt_mask_inss':  [-1]+self.box_msk_siz,
                                                       'label/gt_mask_sems':  [-1, -1]},
                                        padding_values=None, drop_remainder=True)
        #dataset = dataset.cache(filename=os.path.join(self.dat_dir, 'cache'))
        dataset  = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_one_shot_iterator()
        example  = iterator.get_next()
        imgs_lst     = tf.split(example['image/image'],         self.gpu_num, axis=0)
        gbxs_lst     = tf.split(example['label/gt_boxes'],      self.gpu_num, axis=0)
        gmk_inss_lst = tf.split(example['label/gt_mask_inss'],  self.gpu_num, axis=0)
        gmk_sems_lst = tf.split(example['label/gt_mask_sems'],  self.gpu_num, axis=0)
        gbx_nums_lst = tf.split(example['label/num_instances'], self.gpu_num, axis=0)
        img_hgts_lst = tf.split(example['image/height'],        self.gpu_num, axis=0)
        img_wdhs_lst = tf.split(example['image/width'],         self.gpu_num, axis=0)
        img_nams_lst = tf.split(example['image/name'],          self.gpu_num, axis=0)
        return imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, img_hgts_lst, img_wdhs_lst, img_nams_lst
    
    def random_colors(self, N, bright=True):
        '''
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        '''
        brightness = 1.0 if bright else 0.7
        hsv        = [(i / N, 1, brightness) for i in range(N)]
        colors     = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask(self, image, mask, color, alpha=0.5):
        '''
        Apply the given mask to the image.
        '''
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c] * 255.0,
                                      image[:, :, c])
        return image
    
    def recover_instances(self, img=None, boxs=None, msk_inss=None, msk_sems=None, img_wdw=None, img_hgt=None, img_wdh=None):
        
        img_wdw_tmp = img_wdw.astype(dtype=np.int32, copy=False)
        
        if isinstance(img, np.ndarray):
            img = img[img_wdw_tmp[0]:img_wdw_tmp[2]+1, img_wdw_tmp[1]:img_wdw_tmp[3]+1, :] #因为window在原真实图片内
            img = img * 255.0
            img = img + self.img_avg
            img = np.clip(img, 0.0, 255.0)
            img = cv2.resize(img, (img_wdh, img_hgt), interpolation=cv2.INTER_LINEAR)
            img = img.astype(dtype=np.uint8, copy=False)
        
        if isinstance(msk_sems, np.ndarray):
            msk_sems = msk_sems[img_wdw_tmp[0]:img_wdw_tmp[2]+1, img_wdw_tmp[1]:img_wdw_tmp[3]+1]
            msk_sems = cv2.resize(msk_sems, (img_wdh, img_hgt), interpolation=cv2.INTER_NEAREST)
            msk_sems = np.eye(self.cls_num, dtype=np.uint8)[msk_sems]
            
        if isinstance(boxs, np.ndarray):
            box_num  = np.shape(boxs)[0]
            boxs_tmp = boxs.astype(dtype=np.int32, copy=False)
            
            if isinstance(msk_inss, np.ndarray):
                msk_inss_lst = []
                for i in range(box_num):
                    box_tmp  = boxs_tmp[i]
                    msk_ins  = msk_inss[i]
                    y1, x1, y2, x2 = box_tmp
                    msk_ins  = cv2.resize(msk_ins, (x2-x1+1, y2-y1+1), interpolation=cv2.INTER_LINEAR)
                    paddings = [[y1, self.img_siz-y2-1], [x1, self.img_siz-x2-1]]
                    msk_ins  = np.pad(msk_ins, paddings, mode='constant')
                    msk_inss_lst.append(msk_ins)
                msk_inss = np.asarray(msk_inss_lst, dtype=np.float32)
                msk_inss = np.transpose(msk_inss, [1, 2, 0])
                
                msk_inss = msk_inss[img_wdw_tmp[0]:img_wdw_tmp[2]+1, img_wdw_tmp[1]:img_wdw_tmp[3]+1, :]
                msk_inss = cv2.resize(msk_inss, (img_wdh, img_hgt), interpolation=cv2.INTER_LINEAR)
                msk_inss = np.reshape(msk_inss, [img_hgt, img_wdh, box_num])
                msk_inss = np.transpose(msk_inss, [2, 0, 1]) #(N, h, w)
                msk_inss = msk_inss >= self.msk_min
                msk_inss = msk_inss.astype(dtype=np.uint8, copy=False)
            
            img_hgt_ = img_wdw[2] - img_wdw[0] + 1.0
            img_wdh_ = img_wdw[3] - img_wdw[1] + 1.0
            beg      = np.array([img_wdw[0], img_wdw[1]], dtype=np.float32)
            beg      = np.tile(beg, [2])
            rat      = np.array([img_hgt/img_hgt_, img_wdh/img_wdh_], dtype=np.float32)
            rat      = np.tile(rat, [2])
            boxs     = boxs - beg
            boxs     = boxs * rat
            boxs     = bbox_clip_py(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        return img, boxs, msk_inss, msk_sems
    
    def display_instances(self, img=None, boxs=None, box_clss=None, box_prbs=None, \
                          msk_inss=None, msk_sems=None, img_nam=None):
        
        _, ax = plt.subplots(1, figsize=self.figsize)
        random.seed(520)
        
        img_hgt, img_wdh = np.shape(img)[0:2]
        # Show area outside image boundaries.
        #ax.set_ylim(img_hgt + 5, -5)
        #ax.set_xlim(-5, img_wdh + 5)
        ax.set_ylim(img_hgt, 1)
        ax.set_xlim(0, img_wdh)
        ax.axis('off')
        ax.set_title(self.title)
        
        if isinstance(msk_sems, np.ndarray):
            colors  = self.random_colors(self.cls_num)
            for i in range(1, self.cls_num):
                img = self.apply_mask(img, msk_sems[:,:,i], colors[i], 0.5)
        
        if isinstance(boxs, np.ndarray):
            box_num = np.shape(boxs)[0]
            if not box_num:
                print("No instances to display!")
                return
            color   = self.random_colors(1)[0]
            colors  = self.random_colors(box_num)
            #colors = self.random_colors(self.cls_num)
            
            boxs    = boxs.astype(np.int32, copy=False)
            #boxs   = boxs.reshape([-1, 4, 2])[:, :, ::-1]
            for i in range(box_num):
                
                y1, x1, y2, x2 = boxs[i]
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                      alpha=0.7, linestyle="solid",
                                      edgecolor=color, facecolor='none')
                ax.add_patch(p)
                '''
                x1 = boxs[i, 0, 0]
                y1 = boxs[i, 0, 1]
                p = patches.Polygon(boxs[i], facecolor='none', edgecolor=color, linewidth=2, linestyle='-', fill=True)
                ax.add_patch(p)
                '''
                box_cls = box_clss[i]
                box_cls = int(box_cls)
                if box_cls < 0:
                    box_cls = 0
                
                if isinstance(msk_inss, np.ndarray):
                    img  = self.apply_mask(img, msk_inss[i], colors[i], 0.5)
                    #img = self.apply_mask(img, msk_inss[i], colors[box_cls], 0.5)
                    cons = find_contours(msk_inss[i], 0.5)
                    for con in cons:
                        #Subtract the padding and flip (y, x) to (x, y)
                        con = np.fliplr(con) - 1
                        p   = Polygon(con, facecolor="none", edgecolor=colors[i])
                        #p  = Polygon(con, facecolor="none", edgecolor=colors[box_cls])
                        ax.add_patch(p)
                box_prb = box_prbs[i] if box_prbs is not None else None
                box_cls = self.cls_idx_to_cls_nam[box_cls]
                caption = "{} {:.3f}".format(box_cls, box_prb) if box_prb else box_cls
                xx = max(min(x1,   img_wdh-100), 0)
                yy = max(min(y1+8, img_hgt-20 ), 0)
                ax.text(xx, yy, caption, color='k', bbox=dict(facecolor='w', alpha=0.5), size=11, backgroundcolor="none")
                
        img = img.astype(dtype=np.uint8, copy=False)
        ax.imshow(img)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        if self.tst_sav:
            img_fil = os.path.join(self.dat_dir, 'result', img_nam)
            plt.savefig(img_fil, format='jpg', bbox_inches='tight', pad_inches=0)
        if self.tst_shw: plt.show()
        plt.close()
    
    def get_input_test(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, img_hgts_lst, img_wdhs_lst = self.get_input()
            imgs     = tf.concat(imgs_lst,     axis=0)
            gbxs     = tf.concat(gbxs_lst,     axis=0)
            gmk_inss = tf.concat(gmk_inss_lst, axis=0)
            gmk_sems = tf.concat(gmk_sems_lst, axis=0)
            gbx_nums = tf.concat(gbx_nums_lst, axis=0)
            img_hgts = tf.concat(img_hgts_lst, axis=0)
            img_wdhs = tf.concat(img_wdhs_lst, axis=0)
            with tf.device("/gpu:0"):
                imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                    self.preprocessing1(imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_hgts, img_wdhs)
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            imgs_kep, gbxs_kep, gmk_inss_kep, gmk_sems_kep, gbx_nums_kep, img_wdws_kep, img_hgts_kep, img_wdhs_kep = \
                sess.run([imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws, img_hgts, img_wdhs])
            
            for i in range(self.bat_siz*self.gpu_num):
                img_tmp      = imgs_kep[i]
                gbx_num_tmp  = gbx_nums_kep[i]
                gbxs_tmp     = gbxs_kep[i][:gbx_num_tmp]
                gmk_inss_tmp = gmk_inss_kep[i][:gbx_num_tmp]
                gmk_sems_tmp = gmk_sems_kep[i]
                img_wdw_tmp  = img_wdws_kep[i]
                img_hgt_tmp  = img_hgts_kep[i]
                img_wdh_tmp  = img_wdhs_kep[i]
                boxs_tmp     = gbxs_tmp[:, :-1]
                box_clss_tmp = gbxs_tmp[:,  -1]
                img_tmp, boxs_tmp, msk_inss_tmp, msk_sems_tmp = \
                    self.recover_instances(img_tmp, boxs_tmp, gmk_inss_tmp, gmk_sems_tmp, \
                                           img_wdw_tmp, img_hgt_tmp, img_wdh_tmp)
                img_nam = str(i) + ".jpg"
                self.display_instances(img_tmp, boxs_tmp, box_clss_tmp, None, None, msk_sems_tmp, img_nam)

    def get_input_test2(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, \
                img_hgts_lst, img_wdhs_lst, img_nams_lst = self.get_input2()
            imgs     = tf.concat(imgs_lst,     axis=0)
            gbxs     = tf.concat(gbxs_lst,     axis=0)
            gmk_inss = tf.concat(gmk_inss_lst, axis=0)
            gmk_sems = tf.concat(gmk_sems_lst, axis=0)
            gbx_nums = tf.concat(gbx_nums_lst, axis=0)
            img_hgts = tf.concat(img_hgts_lst, axis=0)
            img_wdhs = tf.concat(img_wdhs_lst, axis=0)
            img_nams = tf.concat(img_nams_lst, axis=0)
            with tf.device("/gpu:0"):
                imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                    self.preprocessing1(imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_hgts, img_wdhs)
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            imgs_kep, gbxs_kep, gmk_inss_kep, gmk_sems_kep, gbx_nums_kep, img_wdws_kep, img_hgts_kep, img_wdhs_kep, img_nams_kep = \
                sess.run([imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws, img_hgts, img_wdhs, img_nams])
            
            for i in range(self.bat_siz_all):
                img_tmp      = imgs_kep[i]
                gbx_num_tmp  = gbx_nums_kep[i]
                gbxs_tmp     = gbxs_kep[i][:gbx_num_tmp]
                gmk_inss_tmp = gmk_inss_kep[i][:gbx_num_tmp]
                gmk_sems_tmp = gmk_sems_kep[i]
                img_wdw_tmp  = img_wdws_kep[i]
                img_hgt_tmp  = img_hgts_kep[i]
                img_wdh_tmp  = img_wdhs_kep[i]
                img_nam_tmp  = img_nams_kep[i]
                boxs_tmp     = gbxs_tmp[:, :-1]
                box_clss_tmp = gbxs_tmp[:,  -1]
                img_tmp, boxs_tmp, msk_inss_tmp, msk_sems_tmp = \
                    self.recover_instances(img_tmp, boxs_tmp, None, msk_sems_tmp, \
                                           img_wdw_tmp, img_hgt_tmp, img_wdh_tmp)
                img_nam = str(i) + ".jpg"
                self.display_instances(img_tmp, boxs_tmp, box_clss_tmp, None, None, msk_sems_tmp, img_nam_tmp)
        
        
class GeneratorForVOC(object):
    
    def __init__(self, mod_tra=True, dat_dir='Mybase/datasets/train', bat_siz=6, epc_num=20, min_after_dequeue=20, \
                 gpu_lst='0', fil_num=32, tst_shw=True, tst_sav=False):
        
        self.mod_tra            = mod_tra
        self.use_pad            = True
        self.use_exp            = False
        self.exp_rat            = 2.0
        self.img_avg            = np.array([123.7, 116.8, 103.9], dtype=np.float32)
        self.img_siz            = 512  #1025  #897  #513
        self.box_siz_min        = 5
        self.box_isc_min        = 0.5
        self.box_msk_siz        = [126, 126]
        self.msk_min            = 0.5
        ############for crop###########
        self.min_object_covered = 0.3
        self.aspect_ratio_range = (0.5, 2.0)
        self.area_range         = (0.1, 1.0)
        self.max_attempts       = 200
        
        self.dat_dir            = dat_dir
        self.bat_siz            = bat_siz
        self.epc_num            = epc_num
        self.min_after_dequeue  = min_after_dequeue
        self.gpu_lst            = gpu_lst
        self.gpu_num            = len(self.gpu_lst.split(','))
        
        self.mdl_dev            = '/cpu:%d' if self.gpu_num == 0 else '/gpu:%d'
        self.gpu_num            = 1         if self.gpu_num == 0 else self.gpu_num
        self.fil_num            = fil_num
        self.num_readers        = 2
        self.num_threads        = 16
        self.bat_siz_all        = self.bat_siz * self.gpu_num
        self.capacity           = self.min_after_dequeue + 3 * self.bat_siz_all
        
        self.max_num            = 100
        self.tst_shw            = tst_shw
        self.tst_sav            = tst_sav

        self.cls_nams = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv/monitor']
        '''
        self.cls_nams = ['background', "dachicun", "daodixian", "ganta", "jueyuanzi"]
        '''
        self.cls_num  = len(self.cls_nams)
        self.cls_idx_to_cls_nam = dict(zip(range(self.cls_num), self.cls_nams))
        self.cls_nam_to_cls_idx = dict(zip(self.cls_nams, range(self.cls_num)))
        
        ########for show######
        self.title        = ""
        self.figsize      = (15, 15)
        
    def make_input(self, train=True):
        
        imgs_dir = 'Mybase/datasets/raw/voc/VOCdevkit/VOCSDS/img'
        inss_dir = 'Mybase/datasets/raw/voc/VOCdevkit/VOCSDS/inst'
        sems_dir = 'Mybase/datasets/raw/voc/VOCdevkit/VOCSDS/cls'
        sets_dir = 'Mybase/datasets/raw/voc/VOCdevkit/VOCSDS/ImageSets/Main'
        sets_fil = 'train.txt' if train else 'val.txt'
        sets_fil = os.path.join(sets_dir, sets_fil)
        rcds_dir = 'Mybase/datasets/train' if train else 'Mybase/datasets/val'
        
        assert os.path.exists(sets_fil), 'The sets file does not exist: {:s}'.format(sets_fil)
        with open(sets_fil) as f:
            sets_idx = [x.strip() for x in f.readlines()]
        img_num = len(sets_idx)
        print('The number of images is {:d}'.format(img_num))
        np.random.shuffle(sets_idx)
        
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            out_nam = 'voc_train.tfrecord' if train else 'voc_val.tfrecord'
            rcd_nam = os.path.join(rcds_dir, out_nam)
            
            options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
            with tf.python_io.TFRecordWriter(rcd_nam, options=options) as writer:
                
                for i, set_idx in enumerate(sets_idx):
                    if i % 50 == 0:
                        print('Converting image %d' % (i))
                    
                    ###read image file###
                    img_fil = os.path.join(imgs_dir, set_idx+'.jpg')
                    assert os.path.exists(img_fil), 'The image file does not exist: {:s}'.format(img_fil)
                    img = cv2.imread(img_fil)
                    img_hgt, img_wdh = img.shape[0], img.shape[1]
                    if img.size == img_hgt * img_wdh:
                        print ('Gray Image %s' %(imgs_lst[i]))
                        img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                        img_tmp[:, :, :] = img[:, :, np.newaxis]
                        img = img_tmp
                    img = img.astype(np.uint8)
                    assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                    img = img[:, :, ::-1]
                    
                    ###read semantic file###
                    sems_fil = os.path.join(sems_dir, set_idx+'.mat')
                    assert os.path.exists(sems_fil), 'The semantic file does not exist: {:s}'.format(sems_fil)
                    sems     = sio.loadmat(sems_fil)
                    sems     = sems['GTcls']['Segmentation'][0][0]
                    sems     = sems.astype(dtype=np.uint8, copy=False)
                    
                    ###read instance file###
                    inss_fil = os.path.join(inss_dir, set_idx+'.mat')
                    assert os.path.exists(inss_fil), 'The instance file does not exist: {:s}'.format(inss_fil)
                    inss     = sio.loadmat(inss_fil)
                    inss     = inss['GTinst']['Segmentation'][0][0]
                    inss     = inss.astype(dtype=np.uint8, copy=False)
                    
                    ###split the instances###
                    inss_uni = np.unique(inss)
                    bgd_idxs = np.where(inss_uni ==   0)[0]
                    inss_uni = np.delete(inss_uni, bgd_idxs)
                    bod_idxs = np.where(inss_uni == 255)[0] #border idexes
                    inss_uni = np.delete(inss_uni, bod_idxs)
                    
                    gbx_num  = len(inss_uni)
                    boxs     = np.zeros((gbx_num, 4), dtype=np.float32)
                    clss     = np.zeros((gbx_num, 1), dtype=np.float32)
                    gmk_inss = np.zeros((gbx_num, img_hgt, img_wdh), dtype=np.uint8)

                    if gbx_num == 0:
                        print('There is no instances in the instance file: {:s}'.format(inss_fil))
                        continue

                    for idx, ins_uni in enumerate(inss_uni):
                        [r, c] = np.where(inss==ins_uni)
                        x1  = np.min(c)
                        x2  = np.max(c)
                        y1  = np.min(r)
                        y2  = np.max(r)
                        ins = (inss == ins_uni)
                        cls = np.unique(sems[ins])
                        assert cls.shape[0] == 1
                        ins = ins.astype(dtype=np.uint8, copy=False)
                        cls = cls[0]
                        clss[idx, :]        = cls
                        boxs[idx, :]        = [y1, x1, y2, x2]
                        gmk_inss[idx, :, :] = ins
                    gbxs     = np.concatenate([boxs, clss], axis=-1)
                    gmk_sems = sems
                    
                    #写tfrecords
                    img_raw      = img.tostring()
                    gbxs_raw     = gbxs.tostring()
                    gmk_inss_raw = gmk_inss.tostring()
                    gmk_sems_raw = gmk_sems.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image/image':         _bytes_feature(img_raw),
                        'image/height':        _int64_feature(img_hgt),
                        'image/width':         _int64_feature(img_wdh),

                        'label/num_instances': _int64_feature(gbx_num),       # N
                        'label/gt_boxes':      _bytes_feature(gbxs_raw),      # of shape (N, 5), (ymin, xmin, ymax, xmax, classid)
                        'label/gt_mask_inss':  _bytes_feature(gmk_inss_raw),  # of shape (N, H, W)
                        'label/gt_mask_sems':  _bytes_feature(gmk_sems_raw)   # of shape (H, W)
                    }))
                    writer.write(example.SerializeToString())
    
    def resize_image_with_pad(self, img=None, gbxs=None, gmk_inss=None, gmk_sems=None):
        #没有crop，不需要对gmk_inss操作
        img_hgt  = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh  = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        boxs     = gbxs[:, :-1]
        clss     = gbxs[:,  -1]
        boxs     = bbox_clip(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        gmk_sems = tf.expand_dims(gmk_sems, axis=-1)
        
        if self.use_pad:
            img      = tf.image.resize_image_with_pad(img,      self.img_siz, self.img_siz, \
                                                      method=tf.image.ResizeMethod.BILINEAR)
            gmk_sems = tf.image.resize_image_with_pad(gmk_sems, self.img_siz, self.img_siz, \
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #对image操作后对boxs操作
            leh_min  = tf.minimum(img_hgt, img_wdh)
            leh_max  = tf.maximum(img_hgt, img_wdh)
            leh_rat  = tf.minimum(self.img_siz/leh_min, self.img_siz/leh_max)
            boxs     = boxs * leh_rat
            #将实际坐标加上偏移值
            img_hgt  = img_hgt * leh_rat
            img_wdh  = img_wdh * leh_rat
            pad_hgt  = self.img_siz - img_hgt
            pad_wdh  = self.img_siz - img_wdh
            pad_top  = tf.round(pad_hgt/2.0)
            pad_lft  = tf.round(pad_wdh/2.0)
            beg      = tf.stack([pad_top, pad_lft], axis=0)
            beg      = tf.tile(beg, [2])
            boxs     = boxs + beg #该boxs在原真实图片内，不需要clip
            img_wdw  = tf.stack([pad_top, pad_lft, pad_top+img_hgt-1, pad_lft+img_wdh-1], axis=0)
        else:
            img      = tf.image.resize_images(img,      [self.img_siz, self.img_siz], \
                                              method=tf.image.ResizeMethod.BILINEAR,         align_corners=True)
            gmk_sems = tf.image.resize_images(gmk_sems, [self.img_siz, self.img_siz], \
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
            #对image操作后对boxs操作
            hgt_rat  = self.img_siz / img_hgt
            wdh_rat  = self.img_siz / img_wdh
            leh_rat  = tf.stack([hgt_rat, wdh_rat], axis=0)
            leh_rat  = tf.tile(leh_rat, [2])
            boxs     = boxs * leh_rat
            img_wdw  = tf.constant([0, 0, self.img_siz-1, self.img_siz-1], dtype=tf.int32)
        #合成gt_boxes
        clss     = tf.expand_dims(clss, axis=-1)
        gbxs     = tf.concat([boxs, clss], axis=-1)
        gbx_tmp  = tf.zeros(shape=[1, 5], dtype=tf.float32) #防止没有一个gt_box
        gbxs     = tf.concat([gbxs, gbx_tmp], axis=0)
        ins_tmp  = tf.zeros(shape=[1]+self.box_msk_siz, dtype=tf.float32)
        gmk_inss = tf.concat([gmk_inss, ins_tmp], axis=0)
        gmk_sems = tf.squeeze(gmk_sems, axis=[-1])
        return img, gbxs, gmk_inss, gmk_sems, img_wdw
    
    def distort_crop(self, img=None, gbxs=None, gmk_inss=None, gmk_sems=None):
        #gmk_inss --> (M, h, w)
        img_hgt  = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh  = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        boxs     = gbxs[:, :-1]
        clss     = gbxs[:,  -1]
        boxs     = bbox_clip(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        
        if self.use_exp:
            exp_rat  = tf.random.uniform(shape=[], minval=1.1, maxval=self.exp_rat, dtype=tf.float32)
            #exp_rat = self.exp_rat
            pad_hgt  = tf.cast(img_hgt*(exp_rat-1.0), dtype=tf.int32)
            pad_wdh  = tf.cast(img_wdh*(exp_rat-1.0), dtype=tf.int32)
            pad_top  = tf.random.uniform(shape=[], minval=0, maxval=pad_hgt_all, dtype=tf.int32)
            pad_lft  = tf.random.uniform(shape=[], minval=0, maxval=pad_wdh_all, dtype=tf.int32)
            pad_btm  = pad_hgt - pad_top
            pad_rgt  = pad_wdh - pad_lft
            paddings = [[pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0]]
            img      = tf.pad(img,      paddings, "CONSTANT", constant_values=0)
            paddings = [[pad_top, pad_btm], [pad_lft, pad_rgt]]
            gmk_sems = tf.pad(gmk_sems, paddings, "CONSTANT", constant_values=0)
            #padding中boxs不会超出边界，不用对boxs和gmk_inss操作
            pad_top  = tf.cast(pad_top, dtype=tf.float32)
            pad_lft  = tf.cast(pad_lft, dtype=tf.float32)
            beg      = tf.stack([pad_top, pad_lft], axis=0)
            beg      = tf.tile(beg, [2])
            boxs     = boxs + beg
            img_hgt  = tf.cast(tf.shape(img)[0], dtype=tf.float32)
            img_wdh  = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        ########################crop the image randomly########################
        ncw_idxs = tf.where(clss>0)
        boxs_tmp = tf.gather_nd(boxs, ncw_idxs)
        boxs_tmp = boxs_tmp / tf.stack([img_hgt-1.0, img_wdh-1.0, img_hgt-1.0, img_wdh-1.0], axis=0)
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(tf.shape(img), bounding_boxes=tf.expand_dims(boxs_tmp, 0), \
                                                   min_object_covered=self.min_object_covered, \
                                                   aspect_ratio_range=self.aspect_ratio_range, \
                                                   area_range=self.area_range, max_attempts=self.max_attempts, \
                                                   use_image_if_no_bounding_boxes=True)
        #对image操作后对boxs操作
        img      = tf.slice(img,      box_beg,      box_siz)
        gmk_sems = tf.slice(gmk_sems, box_beg[:-1], box_siz[:-1])
        
        img_hgt  = tf.cast(box_siz[0], dtype=tf.float32)
        img_wdh  = tf.cast(box_siz[1], dtype=tf.float32)
        #将实际坐标加上偏移值
        beg      = tf.cast(box_beg[0:2], dtype=tf.float32)
        beg      = tf.tile(beg, [2])
        boxs_tmp = boxs - beg
        #防止box超出边界
        boxs     = bbox_clip(boxs_tmp, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        #剔除过小的boxs和msks
        box_iscs = bbox_intersects1(boxs_tmp, boxs)
        idxs     = tf.where(box_iscs<self.box_isc_min)
        clss     = tensor_update(clss, idxs, -1)
        #idxs    = tf.where(box_iscs>=self.box_isc_min)
        #boxs    = tf.gather_nd(boxs,     idxs)
        #clss    = tf.gather_nd(clss,     idxs)
        #gmk_inss= tf.gather_nd(gmk_inss, idxs)
        clss     = tf.expand_dims(clss, axis=-1)
        gbxs     = tf.concat([boxs, clss], axis=-1)
        #计算crop后的gmk_inss
        begs     = tf.stack([boxs_tmp[:, 0], boxs_tmp[:, 1], boxs_tmp[:, 0], boxs_tmp[:, 1]], axis=-1)
        lehs     = tf.stack([boxs_tmp[:, 2]-boxs_tmp[:, 0], boxs_tmp[:, 3]-boxs_tmp[:, 1],
                             boxs_tmp[:, 2]-boxs_tmp[:, 0], boxs_tmp[:, 3]-boxs_tmp[:, 1]], axis=-1)
        boxs_tmp = (boxs - begs) / lehs
        idxs     = tf.range(tf.shape(boxs_tmp)[0])
        gmk_inss = tf.expand_dims(gmk_inss, axis=-1) #(M, H, W, 1)
        gmk_inss = tf.image.crop_and_resize(gmk_inss, boxs_tmp, idxs, self.box_msk_siz, method='bilinear') #(M, 256, 256)
        gmk_inss = tf.squeeze(gmk_inss, axis=[-1])   #(M, H, W)
        
        ###########resize image to the expected size with paddings############
        img, gbxs, gmk_inss, gmk_sems, img_wdw = self.resize_image_with_pad(img, gbxs, gmk_inss, gmk_sems)
        return img, gbxs, gmk_inss, gmk_sems, img_wdw
    
    def preprocessing(self, img=None, gbxs=None, gmk_inss=None):
        img_shp  = tf.shape(img)                     #[H, W, C]
        box_leh  = tf.cast(tf.tile(img_shp[:2], [2]), dtype=tf.float32) #[H, W, H, W]
        boxs_tmp = gbxs[:,:-1] / box_leh             #(M, 4)
        idxs     = tf.range(tf.shape(boxs_tmp)[0])
        gmk_inss = tf.expand_dims(gmk_inss, axis=-1) #(M, H, W, 1)
        gmk_inss = tf.image.crop_and_resize(gmk_inss, boxs_tmp, idxs, self.box_msk_siz, method='bilinear') #(M, 256, 256)
        gmk_inss = tf.squeeze(gmk_inss, axis=[-1])   #(M, H, W)
        return gmk_inss
    
    def preprocessing0(self, elms=None):
        img, gbxs, gmk_inss, gmk_sems, gbx_num, img_hgt, img_wdh = elms
        img      = img[:img_hgt, :img_wdh, :]
        gbxs     = gbxs[:gbx_num, :]
        gmk_inss = gmk_inss[:gbx_num, :, :]
        gmk_sems = gmk_sems[:img_hgt, :img_wdh]
        if self.mod_tra:
            img, gbxs, gmk_inss, gmk_sems, img_wdw = self.distort_crop(img, gbxs, gmk_inss, gmk_sems)
        else:
            img, gbxs, gmk_inss, gmk_sems, img_wdw = self.resize_image_with_pad(img, gbxs, gmk_inss, gmk_sems)
        gbx_num  = tf.shape(gbxs)[0]
        paddings = [[0, self.max_num-gbx_num], [0, 0]]
        gbxs     = tf.pad(gbxs,     paddings, "CONSTANT", constant_values=0)
        paddings = [[0, self.max_num-gbx_num], [0, 0], [0, 0]]
        gmk_inss = tf.pad(gmk_inss, paddings, "CONSTANT", constant_values=0)
        return img, gbxs, gmk_inss, gmk_sems, gbx_num, img_wdw
    
    def preprocessing1(self, imgs=None, gbxs=None, gmk_inss=None, gmk_sems=None, gbx_nums=None, img_hgts=None, img_wdhs=None):
        imgs     = tf.cast(imgs,     dtype=tf.float32)
        gmk_sems = tf.cast(gmk_sems, dtype=tf.int32  )
        elms     = [imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_hgts, img_wdhs]
        
        if self.mod_tra:
            imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                tf.map_fn(self.preprocessing0, elms, \
                          dtype=(tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32),
                          parallel_iterations=self.bat_siz, back_prop=False, swap_memory=False, infer_shape=True)
            ######################随机翻转##########################
            sig          = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
            #imgs        = tf.image.random_flip_left_right(imgs)  #(N, H, W, C)
            imgs_flr     = tf.image.flip_left_right(imgs)         #(N, H, W, C)
            gmk_sems_flr = tf.expand_dims(gmk_sems, axis=-1)      #(N, H, W, 1)
            gmk_sems_flr = tf.image.flip_left_right(gmk_sems_flr) #(N, H, W, 1)
            gmk_sems_flr = tf.squeeze(gmk_sems_flr, axis=[-1])    #(N, H, W)
            gmk_inss_flr = tf.transpose(gmk_inss,     [0,2,3,1])  #(N, H, W, M)
            gmk_inss_flr = tf.image.flip_left_right(gmk_inss_flr) #(N, H, W, M)
            gmk_inss_flr = tf.transpose(gmk_inss_flr, [0,3,1,2])  #(N, M, H, W)
            gbxs_flr     = tf.stack([gbxs[:,:,0], self.img_siz-1.0-gbxs[:,:,3], \
                                     gbxs[:,:,2], self.img_siz-1.0-gbxs[:,:,1], gbxs[:,:,4]], axis=-1)
            img_wdws_flr = tf.stack([img_wdws[:,0], self.img_siz-1.0-img_wdws[:,3], \
                                     img_wdws[:,2], self.img_siz-1.0-img_wdws[:,1]], axis=-1)
            imgs         = tf.cond(sig<0.5, lambda: imgs_flr,     lambda: imgs    )
            gmk_sems     = tf.cond(sig<0.5, lambda: gmk_sems_flr, lambda: gmk_sems)
            gmk_inss     = tf.cond(sig<0.5, lambda: gmk_inss_flr, lambda: gmk_inss)
            gbxs         = tf.cond(sig<0.5, lambda: gbxs_flr,     lambda: gbxs    )
            img_wdws     = tf.cond(sig<0.5, lambda: img_wdws_flr, lambda: img_wdws)
            #####################光学畸变###########################
            imgs = apply_with_random_selector(imgs, lambda x, order: distort_color(x, order), num_cases=4)
            imgs = imgs - self.img_avg
            imgs = imgs / 255.0
        else:
            imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                tf.map_fn(self.preprocessing0, elms, \
                          dtype=(tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32),
                          parallel_iterations=self.bat_siz, back_prop=False, swap_memory=False, infer_shape=True)
            imgs = imgs - self.img_avg
            imgs = imgs / 255.0
        return imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws
    
    def get_input(self):

        def parse_function(serialized_example):
            '''
            定长特征解析：tf.FixedLenFeature(shape, dtype, default_value)
                        shape：可当 reshape 来用，如 vector 的 shape 从 (3,) 改动成了 (1,3)。
                               注：如果写入的 feature 使用了. tostring() 其 shape 就是 ()
                        dtype：必须是 tf.float32， tf.int64， tf.string 中的一种。
                        default_value：feature 值缺失时所指定的值。
            不定长特征解析：tf.VarLenFeature(dtype)
                          注：可以不明确指定 shape，但得到的 tensor 是 SparseTensor。
                              变长的tensor转化为string后，shape是固定的，shape=[]，所以可以用tf.FixedLenFeature进行解析
            '''
            parsed_example = tf.parse_single_example(
                serialized_example, 
                features = {
                    'image/image':         tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'image/height':        tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'image/width':         tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/num_instances': tf.FixedLenFeature(shape=[], dtype=tf.int64,  default_value=None),
                    'label/gt_boxes':      tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'label/gt_mask_inss':  tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    'label/gt_mask_sems':  tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
                    #'matrix':             tf.VarLenFeature(dtype=dtype('float32')), 
                    #'matrix_shape':       tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                }
            )
            img_hgt  = tf.cast(parsed_example['image/height'],        tf.int32)
            img_wdh  = tf.cast(parsed_example['image/width'],         tf.int32)
            gbx_num  = tf.cast(parsed_example['label/num_instances'], tf.int32)

            img      = tf.decode_raw(parsed_example['image/image'],        tf.uint8  )
            gbxs     = tf.decode_raw(parsed_example['label/gt_boxes'],     tf.float32)
            gmk_inss = tf.decode_raw(parsed_example['label/gt_mask_inss'], tf.uint8  )
            gmk_sems = tf.decode_raw(parsed_example['label/gt_mask_sems'], tf.uint8  )
            
            img      = tf.reshape(img,      [img_hgt, img_wdh, 3])
            gbxs     = tf.reshape(gbxs,     [gbx_num, 5])
            gmk_inss = tf.reshape(gmk_inss, [gbx_num, img_hgt, img_wdh])
            gmk_sems = tf.reshape(gmk_sems, [img_hgt, img_wdh])
            
            gmk_inss = self.preprocessing(img, gbxs, gmk_inss)
            
            parsed_example = {
                'image/image':         img,
                'image/height':        img_hgt,
                'image/width':         img_wdh,
                'label/num_instances': gbx_num,
                'label/gt_boxes':      gbxs,
                'label/gt_mask_inss':  gmk_inss,
                'label/gt_mask_sems':  gmk_sems
            }
            #parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
            #parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])
            return parsed_example
        
        fil_pat  = os.path.join(self.dat_dir, 'voc', '*.tfrecord')
        dataset  = tf.data.Dataset.list_files(file_pattern=fil_pat, shuffle=True, seed=None)
        dataset  = dataset.prefetch(buffer_size=self.num_readers)
        dataset  = dataset.shuffle(buffer_size=self.fil_num+self.num_readers, seed=None, reshuffle_each_iteration=True)
        dataset  = dataset.apply(tf.data.experimental.\
                                 parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                                                     cycle_length=self.num_readers, \
                                                     block_length=10, sloppy=True, \
                                                     buffer_output_elements=10, \
                                                     prefetch_input_elements=10))
        dataset  = dataset.prefetch(buffer_size=self.bat_siz_all)
        dataset  = dataset.map(parse_function, num_parallel_calls=self.num_threads)
        dataset  = dataset.apply(tf.data.experimental.\
                                 shuffle_and_repeat(buffer_size=self.capacity, count=self.epc_num, seed=None))
        dataset  = dataset.padded_batch(batch_size=self.bat_siz_all, \
                                        padded_shapes={'image/image':         [-1, -1, 3],
                                                       'image/height':        [],
                                                       'image/width':         [],
                                                       'label/num_instances': [],
                                                       'label/gt_boxes':      [-1, 5],
                                                       'label/gt_mask_inss':  [-1]+self.box_msk_siz,
                                                       'label/gt_mask_sems':  [-1, -1]},
                                        padding_values=None, drop_remainder=True)
        #dataset = dataset.batch(batch_size=self.bat_siz_all, drop_remainder=True)
        #dataset = dataset.apply(tf.data.experimental.\
        #                        map_and_batch(parse_function, batch_size=self.bat_siz_all, num_parallel_batches=None, \
        #                                      drop_remainder=True, num_parallel_calls=self.num_threads))
        dataset  = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_one_shot_iterator()
        example  = iterator.get_next()
        imgs_lst     = tf.split(example['image/image'],         self.gpu_num, axis=0)
        gbxs_lst     = tf.split(example['label/gt_boxes'],      self.gpu_num, axis=0)
        gmk_inss_lst = tf.split(example['label/gt_mask_inss'],  self.gpu_num, axis=0)
        gmk_sems_lst = tf.split(example['label/gt_mask_sems'],  self.gpu_num, axis=0)
        gbx_nums_lst = tf.split(example['label/num_instances'], self.gpu_num, axis=0)
        img_hgts_lst = tf.split(example['image/height'],        self.gpu_num, axis=0)
        img_wdhs_lst = tf.split(example['image/width'],         self.gpu_num, axis=0)
        return imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, img_hgts_lst, img_wdhs_lst
    
    def get_input2(self):
        
        imgs_lst_tst = []
        for ext in ['jpg', 'png', 'jpeg', 'JPG']:
            imgs_lst_tst.extend(glob.glob(os.path.join(self.dat_dir, '*.{}'.format(ext))))
        print('There are {:d} images to test!'.format(len(imgs_lst_tst)))
        
        def data_generator():
            for img_fil in imgs_lst_tst:
                assert os.path.exists(img_fil), 'The image file does not exist: {:s}'.format(img_fil)
                img     = cv2.imread(img_fil)
                img_nam = img_fil.split('/')[-1]
                img_hgt = img.shape[0]
                img_wdh = img.shape[1]
                if img.size == img_hgt * img_wdh:
                    print ('Gray Image %s' %(imgs_lst[i]))
                    img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                    img_tmp[:, :, :] = img[:, :, np.newaxis]
                    img = img_tmp
                img = img.astype(np.uint8)
                assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                img = img[:, :, ::-1]
                generated_example = {
                    'image/image':  img,
                    'image/height': img_hgt,
                    'image/width':  img_wdh,
                    'image/name':   img_nam
                }
                yield generated_example
        
        def parse_function(generated_example):
            img     = generated_example['image/image']
            img_hgt = generated_example['image/height']
            img_wdh = generated_example['image/width']
            img_nam = generated_example['image/name']
            
            parsed_example = {
                'image/image':         img,
                'image/height':        img_hgt,
                'image/width':         img_wdh,
                'image/name':          img_nam,
                'label/num_instances': 1,
                'label/gt_boxes':      tf.zeros(shape=[1, 5],               dtype=tf.float32),
                'label/gt_mask_inss':  tf.zeros(shape=[1]+self.box_msk_siz, dtype=tf.float32),
                'label/gt_mask_sems':  tf.zeros(shape=[img_hgt, img_wdh],   dtype=tf.uint8)
            }
            #parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])
            #parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], parsed_example['matrix_shape'])
            return parsed_example
        
        dataset = tf.data.Dataset.from_generator(data_generator, 
                                                 output_types ={'image/image': tf.uint8, 'image/height': tf.int32,
                                                                'image/width': tf.int32, 'image/name':   tf.string}, \
                                                 output_shapes={'image/image': [None, None, 3], 'image/height': [],
                                                                'image/width': [],              'image/name':   []}, args=None)
        dataset  = dataset.repeat(count=1)
        dataset  = dataset.prefetch(buffer_size=self.bat_siz_all)
        dataset  = dataset.map(parse_function, num_parallel_calls=self.num_threads)
        #dataset = dataset.batch(batch_size=self.bat_siz_all, drop_remainder=False)
        #dataset = dataset.apply(tf.data.experimental.\
        #                        map_and_batch(parse_function, batch_size=self.bat_siz_all, num_parallel_batches=None, \
        #                                      drop_remainder=False, num_parallel_calls=self.num_threads))
        dataset  = dataset.padded_batch(batch_size=self.bat_siz_all, \
                                        padded_shapes={'image/image':         [-1, -1, 3],
                                                       'image/height':        [],
                                                       'image/width':         [],
                                                       'image/name':          [],
                                                       'label/num_instances': [],
                                                       'label/gt_boxes':      [-1, 5],
                                                       'label/gt_mask_inss':  [-1]+self.box_msk_siz,
                                                       'label/gt_mask_sems':  [-1, -1]},
                                        padding_values=None, drop_remainder=True)
        #dataset = dataset.cache(filename=os.path.join(self.dat_dir, 'cache'))
        dataset  = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_one_shot_iterator()
        example  = iterator.get_next()
        imgs_lst     = tf.split(example['image/image'],         self.gpu_num, axis=0)
        gbxs_lst     = tf.split(example['label/gt_boxes'],      self.gpu_num, axis=0)
        gmk_inss_lst = tf.split(example['label/gt_mask_inss'],  self.gpu_num, axis=0)
        gmk_sems_lst = tf.split(example['label/gt_mask_sems'],  self.gpu_num, axis=0)
        gbx_nums_lst = tf.split(example['label/num_instances'], self.gpu_num, axis=0)
        img_hgts_lst = tf.split(example['image/height'],        self.gpu_num, axis=0)
        img_wdhs_lst = tf.split(example['image/width'],         self.gpu_num, axis=0)
        img_nams_lst = tf.split(example['image/name'],          self.gpu_num, axis=0)
        return imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, img_hgts_lst, img_wdhs_lst, img_nams_lst
    
    def random_colors(self, N, bright=True):
        '''
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        '''
        brightness = 1.0 if bright else 0.7
        hsv        = [(i / N, 1, brightness) for i in range(N)]
        colors     = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask(self, image, mask, color, alpha=0.5):
        '''
        Apply the given mask to the image.
        '''
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c] * 255.0,
                                      image[:, :, c])
        return image
    
    def recover_instances(self, img=None, boxs=None, msk_inss=None, msk_sems=None, img_wdw=None, img_hgt=None, img_wdh=None):
        
        img_wdw_tmp = img_wdw.astype(dtype=np.int32, copy=False)
        
        if isinstance(img, np.ndarray):
            img = img[img_wdw_tmp[0]:img_wdw_tmp[2]+1, img_wdw_tmp[1]:img_wdw_tmp[3]+1, :] #因为window在原真实图片内
            img = img * 255.0
            img = img + self.img_avg
            img = np.clip(img, 0.0, 255.0)
            img = cv2.resize(img, (img_wdh, img_hgt), interpolation=cv2.INTER_LINEAR)
            img = img.astype(dtype=np.uint8, copy=False)
        
        if isinstance(msk_sems, np.ndarray):
            msk_sems = msk_sems[img_wdw_tmp[0]:img_wdw_tmp[2]+1, img_wdw_tmp[1]:img_wdw_tmp[3]+1]
            msk_sems = cv2.resize(msk_sems, (img_wdh, img_hgt), interpolation=cv2.INTER_NEAREST)
            msk_sems = np.eye(self.cls_num, dtype=np.uint8)[msk_sems]
            
        if isinstance(boxs, np.ndarray):
            box_num  = np.shape(boxs)[0]
            boxs_tmp = boxs.astype(dtype=np.int32, copy=False)
            
            if isinstance(msk_inss, np.ndarray):
                msk_inss_lst = []
                for i in range(box_num):
                    box_tmp  = boxs_tmp[i]
                    msk_ins  = msk_inss[i]
                    y1, x1, y2, x2 = box_tmp
                    msk_ins  = cv2.resize(msk_ins, (x2-x1+1, y2-y1+1), interpolation=cv2.INTER_LINEAR)
                    paddings = [[y1, self.img_siz-y2-1], [x1, self.img_siz-x2-1]]
                    msk_ins  = np.pad(msk_ins, paddings, mode='constant')
                    msk_inss_lst.append(msk_ins)
                msk_inss = np.asarray(msk_inss_lst, dtype=np.float32)
                msk_inss = np.transpose(msk_inss, [1, 2, 0])
                
                msk_inss = msk_inss[img_wdw_tmp[0]:img_wdw_tmp[2]+1, img_wdw_tmp[1]:img_wdw_tmp[3]+1, :]
                msk_inss = cv2.resize(msk_inss, (img_wdh, img_hgt), interpolation=cv2.INTER_LINEAR)
                msk_inss = np.reshape(msk_inss, [img_hgt, img_wdh, box_num])
                msk_inss = np.transpose(msk_inss, [2, 0, 1]) #(N, h, w)
                msk_inss = msk_inss >= self.msk_min
                msk_inss = msk_inss.astype(dtype=np.uint8, copy=False)
            
            img_hgt_ = img_wdw[2] - img_wdw[0] + 1.0
            img_wdh_ = img_wdw[3] - img_wdw[1] + 1.0
            beg      = np.array([img_wdw[0], img_wdw[1]], dtype=np.float32)
            beg      = np.tile(beg, [2])
            rat      = np.array([img_hgt/img_hgt_, img_wdh/img_wdh_], dtype=np.float32)
            rat      = np.tile(rat, [2])
            boxs     = boxs - beg
            boxs     = boxs * rat
            boxs     = bbox_clip_py(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        return img, boxs, msk_inss, msk_sems
    
    def display_instances(self, img=None, boxs=None, box_clss=None, box_prbs=None, \
                          msk_inss=None, msk_sems=None, img_nam=None):
        
        _, ax = plt.subplots(1, figsize=self.figsize)
        random.seed(520)
        
        img_hgt, img_wdh = np.shape(img)[0:2]
        # Show area outside image boundaries.
        #ax.set_ylim(img_hgt + 5, -5)
        #ax.set_xlim(-5, img_wdh + 5)
        ax.set_ylim(img_hgt, 1)
        ax.set_xlim(0, img_wdh)
        ax.axis('off')
        ax.set_title(self.title)
        
        if isinstance(msk_sems, np.ndarray):
            colors  = self.random_colors(self.cls_num)
            for i in range(1, self.cls_num):
                img = self.apply_mask(img, msk_sems[:,:,i], colors[i], 0.5)
        
        if isinstance(boxs, np.ndarray):
            box_num = np.shape(boxs)[0]
            if not box_num:
                print("No instances to display!")
                return
            color   = self.random_colors(1)[0]
            colors  = self.random_colors(box_num)
            #colors = self.random_colors(self.cls_num)
            
            boxs    = boxs.astype(np.int32, copy=False)
            #boxs   = boxs.reshape([-1, 4, 2])[:, :, ::-1]
            for i in range(box_num):
                
                y1, x1, y2, x2 = boxs[i]
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                      alpha=0.7, linestyle="solid",
                                      edgecolor=color, facecolor='none')
                ax.add_patch(p)
                '''
                x1 = boxs[i, 0, 0]
                y1 = boxs[i, 0, 1]
                p = patches.Polygon(boxs[i], facecolor='none', edgecolor=color, linewidth=2, linestyle='-', fill=True)
                ax.add_patch(p)
                '''
                box_cls = box_clss[i]
                box_cls = int(box_cls)
                if box_cls < 0:
                    box_cls = 0
                
                if isinstance(msk_inss, np.ndarray):
                    img  = self.apply_mask(img, msk_inss[i], colors[i], 0.5)
                    #img = self.apply_mask(img, msk_inss[i], colors[box_cls], 0.5)
                    cons = find_contours(msk_inss[i], 0.5)
                    for con in cons:
                        #Subtract the padding and flip (y, x) to (x, y)
                        con = np.fliplr(con) - 1
                        p   = Polygon(con, facecolor="none", edgecolor=colors[i])
                        #p  = Polygon(con, facecolor="none", edgecolor=colors[box_cls])
                        ax.add_patch(p)
                box_prb = box_prbs[i] if box_prbs is not None else None
                box_cls = self.cls_idx_to_cls_nam[box_cls]
                caption = "{} {:.3f}".format(box_cls, box_prb) if box_prb else box_cls
                xx = max(min(x1,   img_wdh-100), 0)
                yy = max(min(y1+8, img_hgt-20 ), 0)
                ax.text(xx, yy, caption, color='k', bbox=dict(facecolor='w', alpha=0.5), size=11, backgroundcolor="none")
                
        img = img.astype(dtype=np.uint8, copy=False)
        ax.imshow(img)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        if self.tst_sav:
            img_fil = os.path.join(self.dat_dir, 'result', img_nam)
            plt.savefig(img_fil, format='jpg', bbox_inches='tight', pad_inches=0)
        if self.tst_shw: plt.show()
        plt.close()
    
    def get_input_test(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, img_hgts_lst, img_wdhs_lst = self.get_input()
            imgs     = tf.concat(imgs_lst,     axis=0)
            gbxs     = tf.concat(gbxs_lst,     axis=0)
            gmk_inss = tf.concat(gmk_inss_lst, axis=0)
            gmk_sems = tf.concat(gmk_sems_lst, axis=0)
            gbx_nums = tf.concat(gbx_nums_lst, axis=0)
            img_hgts = tf.concat(img_hgts_lst, axis=0)
            img_wdhs = tf.concat(img_wdhs_lst, axis=0)
            with tf.device("/gpu:0"):
                imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                    self.preprocessing1(imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_hgts, img_wdhs)
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            imgs_kep, gbxs_kep, gmk_inss_kep, gmk_sems_kep, gbx_nums_kep, img_wdws_kep, img_hgts_kep, img_wdhs_kep = \
                sess.run([imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws, img_hgts, img_wdhs])
            
            for i in range(self.bat_siz*self.gpu_num):
                img_tmp      = imgs_kep[i]
                gbx_num_tmp  = gbx_nums_kep[i]
                gbxs_tmp     = gbxs_kep[i][:gbx_num_tmp]
                gmk_inss_tmp = gmk_inss_kep[i][:gbx_num_tmp]
                gmk_sems_tmp = gmk_sems_kep[i]
                img_wdw_tmp  = img_wdws_kep[i]
                img_hgt_tmp  = img_hgts_kep[i]
                img_wdh_tmp  = img_wdhs_kep[i]
                boxs_tmp     = gbxs_tmp[:, :-1]
                box_clss_tmp = gbxs_tmp[:,  -1]
                img_tmp, boxs_tmp, msk_inss_tmp, msk_sems_tmp = \
                    self.recover_instances(img_tmp, boxs_tmp, gmk_inss_tmp, gmk_sems_tmp, \
                                           img_wdw_tmp, img_hgt_tmp, img_wdh_tmp)
                img_nam = str(i) + ".jpg"
                self.display_instances(img_tmp, boxs_tmp, box_clss_tmp, None, None, msk_sems_tmp, img_nam)

    def get_input_test2(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, \
                img_hgts_lst, img_wdhs_lst, img_nams_lst = self.get_input2()
            imgs     = tf.concat(imgs_lst,     axis=0)
            gbxs     = tf.concat(gbxs_lst,     axis=0)
            gmk_inss = tf.concat(gmk_inss_lst, axis=0)
            gmk_sems = tf.concat(gmk_sems_lst, axis=0)
            gbx_nums = tf.concat(gbx_nums_lst, axis=0)
            img_hgts = tf.concat(img_hgts_lst, axis=0)
            img_wdhs = tf.concat(img_wdhs_lst, axis=0)
            img_nams = tf.concat(img_nams_lst, axis=0)
            with tf.device("/gpu:0"):
                imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws = \
                    self.preprocessing1(imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_hgts, img_wdhs)
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            imgs_kep, gbxs_kep, gmk_inss_kep, gmk_sems_kep, gbx_nums_kep, img_wdws_kep, img_hgts_kep, img_wdhs_kep, img_nams_kep = \
                sess.run([imgs, gbxs, gmk_inss, gmk_sems, gbx_nums, img_wdws, img_hgts, img_wdhs, img_nams])
            
            for i in range(self.bat_siz_all):
                img_tmp      = imgs_kep[i]
                gbx_num_tmp  = gbx_nums_kep[i]
                gbxs_tmp     = gbxs_kep[i][:gbx_num_tmp]
                gmk_inss_tmp = gmk_inss_kep[i][:gbx_num_tmp]
                gmk_sems_tmp = gmk_sems_kep[i]
                img_wdw_tmp  = img_wdws_kep[i]
                img_hgt_tmp  = img_hgts_kep[i]
                img_wdh_tmp  = img_wdhs_kep[i]
                img_nam_tmp  = img_nams_kep[i]
                boxs_tmp     = gbxs_tmp[:, :-1]
                box_clss_tmp = gbxs_tmp[:,  -1]
                img_tmp, boxs_tmp, msk_inss_tmp, msk_sems_tmp = \
                    self.recover_instances(img_tmp, boxs_tmp, None, msk_sems_tmp, \
                                           img_wdw_tmp, img_hgt_tmp, img_wdh_tmp)
                img_nam = str(i) + ".jpg"
                self.display_instances(img_tmp, boxs_tmp, box_clss_tmp, None, None, msk_sems_tmp, img_nam_tmp)
"""        
#from Mybase.leye_utils.proposals_target_layer import *
            
class GeneratorForSynthText(object):
    
    def __init__(self, mod_tra=False, dat_dir='train', bat_siz=5, min_after_dequeue=20):

        self.mod_tra            = mod_tra
        self.use_pad            = False
        
        self.img_siz_min        = 400
        self.img_siz_max        = 513
        self.box_siz_min        = 5
        self.box_isc_min        = 0.5
        ############for crop###########
        self.min_object_covered = 1.0
        self.aspect_ratio_range = (0.5, 2.0)
        self.area_range         = (0.3, 1.0)
        self.max_attempts       = 200
        
        self.dat_dir            = dat_dir
        self.max_num            = 100
        self.bat_siz            = bat_siz
        self.min_after_dequeue  = min_after_dequeue
        self.num_threads        = 16

        self.cls_nams = ['background', 'text']
        self.cls_num  = len(self.cls_nams)
        self.cls_idx_to_cls_nam = dict(zip(range(self.cls_num), self.cls_nams))
        self.cls_nam_to_cls_idx = dict(zip(self.cls_nams, range(self.cls_num)))
        
        ########for test######
        self.imgs_lst_tst = []
        self.imgs_dir_lst = ["Mybase/datasets/test"]
        for img_dir in self.imgs_dir_lst:
            for ext in ['jpg', 'png', 'jpeg', 'JPG']:
                self.imgs_lst_tst.extend(glob.glob(os.path.join(img_dir, '*.{}'.format(ext))))
        self.anns_lst_tst = []
        self.gbxs_lst_tst = [] #暂不支持use_gbx==True
        self.img_num_tst  = len(self.imgs_lst_tst)
        self.get_idx = 0
    
    def load_meta(self, mets_dir=None):

        met_dats = loadmat(mets_dir, struct_as_record=False)
        img_nams = list(met_dats['imnames'][0])
        wrd_gbxs = list(met_dats['wordBB' ][0])
        cha_gbxs = list(met_dats['charBB' ][0])
        gbx_lbls = list(met_dats['txt'    ][0])
        return img_nams, wrd_gbxs, cha_gbxs, gbx_lbls

    def make_input(self, num_per_sha=3000):

        ##############此处添加image文件路径##############
        imgs_dir = "Mybase/datasets/SynthText"
        ##############此处添加annotation文件路径##############
        mets_dir = "Mybase/datasets/SynthText"
        ##############此处添加tfrecords文件保存路径##############
        rcds_dir = "Mybase/tfrecords"

        img_nams_lst, wrd_gbxs_lst, cha_gbxs_lst, gbx_lbls_lst = self.load_meta(os.path.join(mets_dir, 'gt.mat'))
        assert len(img_nams_lst) == len(wrd_gbxs_lst) == len(cha_gbxs_lst) == len(gbx_lbls_lst), 'img_num wrong!'
        
        imgs_lst = [os.path.join(mets_dir, x[0]) for x in img_nams_lst]
        img_num = len(imgs_lst)
        print("The datasets have a total of {:d} images!".format(img_num))
        
        idxs = np.arange(img_num)
        np.random.shuffle(idxs)
        imgs_lst = [imgs_lst[x] for x in idxs]
        wrd_gbxs_lst = [wrd_gbxs_lst[x] for x in idxs]
        cha_gbxs_lst = [cha_gbxs_lst[x] for x in idxs]
        gbx_lbls_lst = [gbx_lbls_lst[x] for x in idxs]
        
        wrd_gbxs_kep = []
        cha_gbxs_kep = []
        for i in range(img_num):
            wrd_gbxs = np.asarray(wrd_gbxs_lst[i], dtype=np.float32)
            cha_gbxs = np.asarray(cha_gbxs_lst[i], dtype=np.float32)
            wrd_gbxs = np.reshape(np.transpose(np.reshape(wrd_gbxs, [2, 4, -1]), [2, 1, 0])[:, :, ::-1], [-1, 8])
            cha_gbxs = np.reshape(np.transpose(np.reshape(wrd_gbxs, [2, 4, -1]), [2, 1, 0])[:, :, ::-1], [-1, 8])
            #boxs = bbox_clip_py2(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
            wrd_clss = np.ones(shape=[wrd_gbxs.shape[0], 1], dtype=np.float32)
            wrd_gbxs = np.concatenate([wrd_gbxs, wrd_clss], axis=-1)
            cha_clss = np.ones(shape=[cha_gbxs.shape[0], 1], dtype=np.float32)
            cha_gbxs = np.concatenate([cha_gbxs, cha_clss], axis=-1)
            wrd_gbxs_kep.append(wrd_gbxs)
            cha_gbxs_kep.append(cha_gbxs)

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            sha_num = int(img_num/num_per_sha)
            if sha_num == 0:
                sha_num = 1
                num_per_sha = img_num
            else:
                num_per_sha = int(math.ceil(img_num/sha_num))

            for sha_idx in range(sha_num):
                out_nam = 'synthtext_%05d-of-%05d.tfrecord' % (sha_idx, sha_num)
                rcd_nam = os.path.join(rcds_dir, out_nam)

                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(rcd_nam, options=options) as writer:

                    sta_idx = sha_idx * num_per_sha
                    end_idx = min((sha_idx + 1) * num_per_sha, img_num)
                    for i in range(sta_idx, end_idx):
                        if i % 50 == 0:
                            print("Converting image %d/%d shard %d" % (i + 1, img_num, sha_idx))
                        #读取图像
                        img = imgs_lst[i]
                        img = cv2.imread(img)
                        if type(img) != np.ndarray:
                            print("Failed to find image %s" %(imgs_lst[i]))
                            continue
                        img_hgt, img_wdh = img.shape[0], img.shape[1]
                        if img.size == img_hgt * img_wdh:
                            print ('Gray Image %s' %(imgs_lst[i]))
                            img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                            img_tmp[:, :, :] = img[:, :, np.newaxis]
                            img = img_tmp
                        img = img.astype(np.uint8)
                        assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                        img = img[:, :, ::-1]
                        #读取标签
                        wrd_gbxs = wrd_gbxs_kep[i]
                        cha_gbxs = cha_gbxs_kep[i]
                        gbx_lbls = gbx_lbls_lst[i]
                        if len(wrd_gbxs)==0 or len(cha_gbxs)==0 or len(gbx_lbls)==0:
                            print("No gt_boxes in this image!")
                            continue
                        #写tfrecords
                        img_raw  = img.tostring()
                        wrd_gbxs_raw = wrd_gbxs.tostring()
                        cha_gbxs_raw = cha_gbxs.tostring()
                        gbx_lbls_raw = gbx_lbls.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image/image':  _bytes_feature(img_raw),
                            'image/height': _int64_feature(img_hgt),
                            'image/width':  _int64_feature(img_wdh),

                            'label/wrds_num': _int64_feature(wrd_gbxs.shape[0]),  # N
                            'label/chas_num': _int64_feature(cha_gbxs.shape[0]),  # N
                            'label/wrd_gbxs': _bytes_feature(wrd_gbxs_raw),  #(N, 8), (y0, x0, y1, x1, y2, x2, y3, x3, cls)
                            'label/cha_gbxs': _bytes_feature(cha_gbxs_raw),
                            'label/gbx_lbls': _bytes_feature(gbx_lbls_raw),
                        }))
                        writer.write(example.SerializeToString())
                        
                        
    def resize_image_with_pad(self, img=None, boxs=None, clss=None):

        #####################按最短边进行比例缩放######################
        im_h = tf.shape(img)[0]
        im_w = tf.shape(img)[1]
        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        if self.use_pad:
            leh_min = tf.minimum(img_hgt, img_wdh)
            leh_max = tf.maximum(img_hgt, img_wdh)
            leh_rat = tf.minimum(self.img_siz_min/leh_min, self.img_siz_max/leh_max)
            img_hgt = tf.cast(img_hgt*leh_rat, dtype=tf.int32)
            img_wdh = tf.cast(img_wdh*leh_rat, dtype=tf.int32)
            #对image操作后对boxs操作
            img  = tf.image.resize_images(img, [img_hgt, img_wdh], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
            boxs = boxs * leh_rat
            ################如果最长边过长则按中心对称进行裁剪################
            img_hgt = tf.cast(img_hgt, dtype=tf.float32)
            img_wdh = tf.cast(img_wdh, dtype=tf.float32)
            pad_hgt_all = tf.cast(self.img_siz_max-img_hgt, dtype=tf.float32)
            pad_wdh_all = tf.cast(self.img_siz_max-img_wdh, dtype=tf.float32)
            pad_hgt_fnt = tf.round(pad_hgt_all/2.0)
            pad_wdh_fnt = tf.round(pad_wdh_all/2.0)
            img_wdw = tf.stack([pad_hgt_fnt, pad_wdh_fnt, pad_hgt_fnt+img_hgt-1, pad_wdh_fnt+img_wdh-1], axis=0) #该边框在原真实图片内
            #对image操作后对boxs操作
            img = tf.image.resize_image_with_crop_or_pad(img, self.img_siz_max, self.img_siz_max)
            #将实际坐标加上偏移值
            beg = tf.stack([pad_hgt_fnt, pad_wdh_fnt], axis=0)
            beg = tf.tile(beg, [4])
            boxs_tmp = boxs + beg
            #防止box超出边界
            #防止box超出边界
            boxs = bbox_clip2(boxs_tmp, [0.0, 0.0, self.img_siz_max-1.0, self.img_siz_max-1.0])
            #box_iscs = bbox_intersects1(boxs_tmp, boxs)
            #idxs = tf.where(box_iscs<self.box_isc_min)
            box_edgs = bbox_edges2(boxs)
            #idxs = tf.where(tf.not_equal(tf.reduce_sum(tf.cast(box_edgs>=self.box_siz_min, tf.int32), axis=1), 4))
            #clss = tensor_update(clss, idxs, -1)
            idxs = tf.where(tf.equal(tf.reduce_sum(tf.cast(box_edgs>=self.box_siz_min, tf.int32), axis=1), 4))
            boxs = tf.gather_nd(boxs, idxs)
            clss = tf.gather_nd(clss, idxs)
        else:
            hgt_rat = self.img_siz_max / img_hgt
            wdh_rat = self.img_siz_max / img_wdh
            leh_rat = tf.stack([hgt_rat, wdh_rat], axis=0)
            leh_rat = tf.tile(leh_rat, [4])
            #对image操作后对boxs操作
            img  = tf.image.resize_images(img, [self.img_siz_max, self.img_siz_max], method=tf.image.ResizeMethod.BILINEAR, \
                                         align_corners=False)
            boxs = boxs * leh_rat
            img_wdw = tf.constant([0, 0, self.img_siz_max-1, self.img_siz_max-1], dtype=tf.float32)
        #合成gt_boxes
        clss = tf.expand_dims(clss, axis=-1)
        gbxs = tf.concat([boxs, clss], axis=-1)
        gbxs = tf.cast(gbxs, dtype=tf.float32)
        gbx_tmp = tf.zeros(shape=[1, 9], dtype=tf.float32) #防止没有一个gt_box
        gbxs = tf.concat([gbxs, gbx_tmp], axis=0)
        return img, gbxs, img_wdw, im_h, im_w
    
    
    def distort_crop(self, img=None, gbxs=None):

        img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        boxs = gbxs[:, :-1]
        clss = gbxs[:,  -1]
        boxs = bbox_clip2(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        
        ########################crop the image randomly########################
        ncw_idxs = tf.where(clss>0)
        boxs_tmp = tf.gather_nd(boxs, ncw_idxs)
        boxs_tmp = bbox_bound2(boxs_tmp)
        boxs_tmp = boxs_tmp / tf.stack([img_hgt-1.0, img_wdh-1.0, img_hgt-1.0, img_wdh-1.0], axis=0)
        box_beg, box_siz, box_bnd = \
            tf.image.sample_distorted_bounding_box(tf.shape(img), bounding_boxes=tf.expand_dims(boxs_tmp, 0), \
                                                   min_object_covered=self.min_object_covered, \
                                                   aspect_ratio_range=self.aspect_ratio_range, \
                                                   area_range=self.area_range, max_attempts=self.max_attempts, \
                                                   use_image_if_no_bounding_boxes=True)
        #对image操作后对boxs操作
        img = tf.slice(img, box_beg, box_siz)
        img_hgt = tf.cast(box_siz[0], dtype=tf.float32)
        img_wdh = tf.cast(box_siz[1], dtype=tf.float32)
        #将实际坐标加上偏移值
        beg = tf.cast(box_beg[0:2], dtype=tf.float32)
        beg = tf.tile(beg, [4])
        boxs_tmp = boxs - beg
        #防止box超出边界
        boxs = bbox_clip2(boxs_tmp, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
        #box_iscs = bbox_intersects1(boxs_tmp, boxs)
        #idxs = tf.where(box_iscs<self.box_isc_min)
        box_edgs = bbox_edges2(boxs)
        #idxs = tf.where(tf.not_equal(tf.reduce_sum(tf.cast(box_edgs>=self.box_siz_min, tf.int32), axis=1), 4))
        #clss = tensor_update(clss, idxs, -1)
        idxs = tf.where(tf.equal(tf.reduce_sum(tf.cast(box_edgs>=self.box_siz_min, tf.int32), axis=1), 4))
        boxs = tf.gather_nd(boxs, idxs)
        clss = tf.gather_nd(clss, idxs)
        ###########resize image to the expected size with paddings############
        img, gbxs, img_wdw, im_h, im_w = self.resize_image_with_pad(img, boxs, clss)
        return img, gbxs, img_wdw, im_h, im_w

    def preprocessing(self, img=None, gbxs=None):

        self.img_avg = tf.constant([123.7, 116.8, 103.9], dtype=tf.float32)
        img = tf.cast(img, dtype=tf.float32)
        ####################归化到0、1之间######################
        #if img.dtype != tf.float32:
        #    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if self.mod_tra == True:
            #####################光学畸变###########################
            # Randomly distort the colors. There are 4 ways to do it.
            img = apply_with_random_selector(img, lambda x, order: distort_color(x, order), num_cases=4)
            #####################随机裁剪###########################
            img, gbxs, img_wdw, im_h, im_w = self.distort_crop(img, gbxs)
            #boxs = gbxs[:, :-1] 
            #clss = gbxs[:,  -1]
            #img, gbxs, img_wdw = self.resize_image_with_pad(img, boxs, clss)
            ######################随机翻转##########################
            sig = tf.random.uniform([2])
            #####################随机左右翻转#######################
            img_hgt = tf.cast(tf.shape(img)[0], dtype=tf.float32)
            img_wdh = tf.cast(tf.shape(img)[1], dtype=tf.float32)
            #img = tf.image.random_flip_left_right(img)
            img_lft_rgt  = tf.image.flip_left_right(img)
            gbxs_lft_rgt = tf.stack([gbxs[:, 2], img_wdh-1.0-gbxs[:, 3], \
                                     gbxs[:, 0], img_wdh-1.0-gbxs[:, 1], \
                                     gbxs[:, 6], img_wdh-1.0-gbxs[:, 7], \
                                     gbxs[:, 4], img_wdh-1.0-gbxs[:, 5], \
                                     gbxs[:, 8]], axis=-1)
            img_wdw_lft_rgt = tf.stack([img_wdw[0], img_wdh-1.0-img_wdw[3], img_wdw[2], img_wdh-1.0-img_wdw[1]], axis=-1)
            img     = tf.cond(sig[0]<0.5, lambda: img_lft_rgt,     lambda: img    )
            gbxs    = tf.cond(sig[0]<0.5, lambda: gbxs_lft_rgt,    lambda: gbxs   )
            img_wdw = tf.cond(sig[0]<0.5, lambda: img_wdw_lft_rgt, lambda: img_wdw)
            #img = tf.image.per_image_standardization(img)
            img  = img - self.img_avg
            return img, gbxs, img_wdw, im_h, im_w
        else:
            boxs = gbxs[:, :-1] 
            clss = gbxs[:,  -1]
            img, gbxs, img_wdw, im_h, im_w = self.resize_image_with_pad(img, boxs, clss)   
            img  = img - self.img_avg
            return img, gbxs, img_wdw, im_h, im_w
    
    
    def get_input(self):
        #创建文件列表，并通过文件列表创建输入文件队列。
        #在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中
        #文件列表应该包含所有提供训练数据的TFRecord文件
        filename = os.path.join(self.dat_dir, "*.tfrecord")
        files = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(files, shuffle=True, capacity=1000)

        #解析TFRecord文件里的数据
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        reader  = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/image':    tf.FixedLenFeature([], tf.string),
                'image/height':   tf.FixedLenFeature([], tf.int64 ),
                'image/width':    tf.FixedLenFeature([], tf.int64 ),

                'label/wrds_num': tf.FixedLenFeature([], tf.int64 ),
                'label/chas_num': tf.FixedLenFeature([], tf.int64 ),
                'label/wrd_gbxs': tf.FixedLenFeature([], tf.string),
                'label/cha_gbxs': tf.FixedLenFeature([], tf.string),
                'label/gbx_lbls': tf.FixedLenFeature([], tf.string),
            }
        )
        
        img_hgt = tf.cast(features['image/height'  ], tf.int32)
        img_wdh = tf.cast(features['image/width'   ], tf.int32)
        gbx_num = tf.cast(features['label/wrds_num'], tf.int32)

        img  = tf.decode_raw(features['image/image'   ], tf.uint8  )
        gbxs = tf.decode_raw(features['label/wrd_gbxs'], tf.float32)
        gbx_lbls = features['label/gbx_lbls']
    
        img  = tf.reshape(img,  [img_hgt, img_wdh, 3])
        gbxs = tf.reshape(gbxs, [gbx_num, 9])
        
        img, gbxs, img_wdw, img_hgt, img_wdh = self.preprocessing(img, gbxs)
        img = tf.reshape(img, [self.img_siz_max, self.img_siz_max, 3])
        
        gbx_num = tf.shape(gbxs)[0]
        paddings = [[0, self.max_num-gbx_num], [0, 0]]
        gbxs = tf.pad(gbxs, paddings)
        gbxs = tf.reshape(gbxs, [self.max_num, 9])

        
        #tf.train.shuffle_batch_join
        imgs, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums = tf.train.shuffle_batch(
            tensors=[img, img_hgt, img_wdh, img_wdw, gbxs, gbx_num], batch_size=self.bat_siz, \
            num_threads=self.num_threads, capacity=capacity, min_after_dequeue=self.min_after_dequeue)
        '''
        imgs, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums = tf.train.batch(
            tensors=[img, img_hgt, img_wdh, img_wdw, gbxs, gbx_num], batch_size=self.bat_siz, \
            num_threads=self.num_threads, capacity=capacity)
        '''
        return imgs, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums
    
    
    def get_input2(self, sess=None, use_gbx=False):
        
        with tf.device("/cpu:0"):
            img_tmp  = tf.placeholder(dtype=tf.uint8,   shape=[None, None, 3], name="image")
            gbxs_tmp = tf.placeholder(dtype=tf.float32, shape=[None, 9], name="gt_boxes")
            img, gbxs, img_wdw, im_h, im_w = self.preprocessing(img_tmp, gbxs_tmp)
            img = tf.reshape(img, [self.img_siz_max, self.img_siz_max, 3])
            img_wdw = tf.reshape(img_wdw, [4])
            gbx_num = tf.shape(gbxs)[0]
            paddings = [[0, self.max_num-gbx_num], [0, 0]]
            gbxs = tf.pad(gbxs, paddings)
            gbxs = tf.reshape(gbxs, [self.max_num, 9])
        imgs_lst     = []
        img_nams_lst = []
        img_hgts_lst = []
        img_wdhs_lst = []
        img_wdws_lst = []
        gbxs_lst     = []
        gbx_nums_lst = []
        self.get_idx      = 0
        while True:
            try:
                #读取图像
                img_kep = self.imgs_lst_tst[self.get_idx]
                img_nams_lst.append(img_kep.split('/')[-1])
                img_kep = cv2.imread(img_kep)
                if type(img_kep) != np.ndarray:
                    print("Failed to find image %s" %(img_kep))
                    continue
                img_hgt, img_wdh = img_kep.shape[0], img_kep.shape[1]
                if img_kep.size == img_hgt * img_wdh:
                    print ('Gray Image %s' % str(self.get_idx))
                    img_zro = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                    img_zro[:, :, :] = img_kep[:, :, np.newaxis]
                    img_kep = img_zro
                img_kep = img_kep.astype(np.uint8)
                assert img_kep.size == img_wdh * img_hgt * 3, '%s' % str(self.get_idx)
                img_kep = img_kep[:, :, ::-1]
                #读取标签
                if use_gbx:
                    gbxs_kep = self.gbxs_lst_tst[self.get_idx]
                else:
                    gbxs_kep = np.zeros(shape=[1, 9], dtype=np.float32)
                
                img_kep, img_wdw_kep, gbxs_kep, gbx_num_kep = sess.run([img, img_wdw, gbxs, gbx_num], \
                                                                       feed_dict={img_tmp: img_kep, gbxs_tmp: gbxs_kep})
                imgs_lst.append(img_kep)
                img_hgts_lst.append(img_hgt)
                img_wdhs_lst.append(img_wdh)
                img_wdws_lst.append(img_wdw_kep)
                gbxs_lst.append(gbxs_kep)
                gbx_nums_lst.append(gbx_num_kep)
                self.get_idx = self.get_idx + 1
                self.get_idx = self.get_idx % self.img_num_tst
                if len(imgs_lst) == self.bat_siz:
                    imgs_lst     = np.asarray(imgs_lst,     dtype=np.float32) #4维
                    img_hgts_lst = np.asarray(img_hgts_lst, dtype=np.float32) #1维
                    img_wdhs_lst = np.asarray(img_wdhs_lst, dtype=np.float32) #1维
                    img_wdws_lst = np.asarray(img_wdws_lst, dtype=np.float32) #4维
                    gbxs_lst     = np.asarray(gbxs_lst,     dtype=np.float32) #3维
                    gbx_nums_lst = np.asarray(gbx_nums_lst, dtype=np.int32  ) #1维
                    yield imgs_lst, img_nams_lst, img_hgts_lst, img_wdhs_lst, img_wdws_lst, gbxs_lst, gbx_nums_lst
                    imgs_lst     = []
                    img_nams_lst = []
                    img_hgts_lst = []
                    img_wdhs_lst = []
                    img_wdws_lst = []
                    gbxs_lst     = []
                    gbx_nums_lst = []
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                continue
    
    
    def random_colors(self, N, bright=True):
        '''
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        '''
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors


    def apply_mask(self, image, mask, color, alpha=0.5):
        '''
        Apply the given mask to the image.
        '''
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
        
    
    def display_instances(self, img=None, boxs=None, box_clss=None, box_scrs=None, boxs_tmp=None, box_msks=None, 
                          img_hgt=None, img_wdh=None, img_wdw=None, title="", figsize=(16, 16), ax=None):
        '''
        Args:
            boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
            figsize: (optional) the size of the image.
        '''
        self.img_avg = np.array([123.7, 116.8, 103.9], dtype=np.float32)
        
        # Generate random colors
        colors = self.random_colors(9)
        # Number of instances
        N = boxs.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            return
        if not ax:
            _, ax = plt.subplots(1, figsize=figsize)
        
        # Show area outside image boundaries.
        ax.set_ylim(img_hgt + 20, -20)
        ax.set_xlim(-20, img_wdh + 20)
        #ax.set_ylim(img_hgt, 0)
        #ax.set_xlim(0, img_wdh)
        ax.axis('off')
        ax.set_title(title)
        
        if len(img_wdw) != 0:
            hgt_tmp = img_wdw[2] - img_wdw[0] + 1
            wdh_tmp = img_wdw[3] - img_wdw[1] + 1
            beg  = np.array([img_wdw[0], img_wdw[1]], dtype=np.float32)
            beg  = np.tile(beg, [4])
            rat  = np.array([img_hgt/hgt_tmp, img_wdh/wdh_tmp], dtype=np.float32)
            rat  = np.tile(rat, [4])
            boxs = boxs - beg
            boxs = boxs * rat
            boxs = bbox_clip_py2(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
            if len(boxs_tmp) > 0:
                boxs_tmp = boxs_tmp.reshape([-1, 4])
                beg  = np.array([img_wdw[0], img_wdw[1]], dtype=np.float32)
                beg  = np.tile(beg, [2])
                boxs_tmp = boxs_tmp - beg
                boxs_tmp = boxs_tmp.reshape([4, -1, 4]) #(ymn, xmn, ymx, xmx)
                boxs_tmp = boxs_tmp * img_hgt / img_hgt_tmp
                #boxs_tmp = bbox_clip_py(boxs_tmp, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
                boxs_tmp = np.around(boxs_tmp).astype(np.int32, copy=False)

            img_wdw = img_wdw.astype(np.int32, copy=False)
            '''
            img_zro = np.empty((self.img_siz_max, self.img_siz_max, 3), dtype=np.float32)
            img_zro[:, :, :] = box_msks[3][:, :, np.newaxis] * 255.0
            img_zro = img_zro[img_wdw[0]:img_wdw[2]+1, img_wdw[1]:img_wdw[3]+1, :] #因为window在原真实图片内
            img_zro = np.clip(img_zro, 0.0, 255.0)
            img_zro = img_zro.astype(np.uint8)
            img_zro = cv2.resize(img_zro, (int(img_wdh), int(img_hgt)), interpolation=cv2.INTER_LINEAR)
            ax.imshow(img_zro)
            '''
            img = img + self.img_avg
            if len(box_msks) > 0:
                for i in range(4):
                    color = colors[2*i+2]
                    for c in range(3):
                        img[:, :, c] = np.where(box_msks[i]==1.0, 0.5*img[:, :, c] + 0.5*color[c]*255.0, img[:, :, c])
            
            img = img[img_wdw[0]:img_wdw[2]+1, img_wdw[1]:img_wdw[3]+1, :] #因为window在原真实图片内
            img = np.clip(img, 0.0, 255.0)
            img = img.astype(np.uint8, copy=False)
            img = cv2.resize(img, (int(img_wdh), int(img_hgt)), interpolation=cv2.INTER_LINEAR)
            
        ax.imshow(img)
        boxs = np.around(boxs).astype(np.int32, copy=False)
        boxs = boxs.reshape([-1, 4, 2])[:, :, ::-1]
        
        color = colors[0]        
        for i in range(N):
            '''
            y1, x1, y2, x2 = boxs[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="solid",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            '''
            x1 = boxs[i, 0, 0]
            y1 = boxs[i, 0, 1]
            p = patches.Polygon(boxs[i], facecolor='none', edgecolor=color, linewidth=2, linestyle='-', fill=True)
            ax.add_patch(p)
            # Label
            box_cls = box_clss[i]
            box_cls = int(box_cls)
            if box_cls < 0:
                box_cls = 0
            box_scr = box_scrs[i] if box_scrs is not None else None
            box_cls = self.cls_idx_to_cls_nam[box_cls]
            caption = "{} {:.3f}".format(box_cls, box_scr) if box_scr else box_cls
            ax.text(x1, y1+8, caption, color='k', bbox=dict(facecolor='w', alpha=0.5), size=11, backgroundcolor="none")
        if len(boxs_tmp) > 0:
            for i in range(4):
                color = colors[2*i+1]
                boxs_kep = boxs_tmp[i]
                for j in range(len(boxs_kep)):
                    y1, x1, y2, x2 = boxs_kep[j]
                    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                          alpha=0.7, linestyle="solid",
                                          edgecolor=color, facecolor='none')
                    '''
                    p = patches.Circle(((x1+x2)/2, (y1+y2)/2), radius=3, alpha=0.7, linestyle="solid",
                                       edgecolor=color, facecolor=color)
                    '''
                    ax.add_patch(p)
        plt.show()
        plt.close()

    
    def display_detections(imgs, img_hgts, img_wdhs, img_wdws, boxs, box_imxs, box_clss, box_prbs, box_msks):

        img_num = len(imgs)

        for i in range(img_num):

            img = imgs[i]
            img_hgt = img_hgts[i]
            img_wdh = img_wdhs[i]
            img_wdw = img_wdws[i]

            idxs = np.where(box_imxs==i)[0]
            boxs_img = boxs[idxs]
            box_clss_img = box_clss[idxs]
            box_prbs_img = box_prbs[idxs]
            box_msks_img = box_msks[idxs]

            self.display_instances(img, box_msks_img, boxs_img, box_clss_img, box_prbs_img, 1024, \
                                   img_hgt, img_wdh, img_wdw, title="", figsize=(13, 13), ax=None)
    
    '''
    def write_image(self):
        
        if boxes is not None:
            res_file = os.path.join(FLAGS.output_dir, '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            if not FLAGS.no_write_images:
                img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                cv2.imwrite(img_path, im[:, :, ::-1])
    '''    
        
    def get_input_test(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums = self.get_input()

        with tf.Session() as sess:

            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            imgs_kep, img_hgts_kep, img_wdhs_kep, img_wdws_kep, gbxs_kep, gbx_nums_kep = \
                sess.run([imgs, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums])
            
            for i in range(self.bat_siz):
                img     = imgs_kep[i]
                gbx_num = gbx_nums_kep[i]
                boxs    = gbxs_kep[i][:gbx_num]
                #print(boxs)
                img_hgt = img_hgts_kep[i]
                img_wdh = img_wdhs_kep[i]
                img_wdw = img_wdws_kep[i]
                self.display_instances(img, boxs[:, 0:-1], boxs[:, -1], None, [], [], 
                                       img_hgt, img_wdh, img_wdw, title="", figsize=(12, 12), ax=None)
            coord.request_stop()
            coord.join(threads)
            
            
    def get_input_test2(self):
        
        tf.reset_default_graph()
        
        with tf.Session() as sess:
            
            imgs, img_nams, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums = next(self.get_input2(sess))            
            for i in range(self.bat_siz):
                img     = imgs[i]
                img_nam = img_nams[i]
                gbx_num = gbx_nums[i]
                boxs    = gbxs[i][:gbx_num]
                #print(boxs)
                img_hgt = img_hgts[i]
                img_wdh = img_wdhs[i]
                img_wdw = img_wdws[i]
                print(img_nam)
                self.display_instances(img, boxs[:, 0:-1], boxs[:, -1], None, [], [], 
                                       img_hgt, img_wdh, img_wdw, title="", figsize=(12, 12), ax=None)
                
    
    def get_input_test3(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            imgs, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums = self.get_input()
        
        PT = ProposalsTargetLayer(img_shp=[self.img_siz_max, self.img_siz_max])
        gbxs_tmp, gbx_msks, gbx_nums_tmp = PT.generate_gbxs(gbxs, gbx_nums) #(img_num, 4, -1, 5)/(img_num, 4, H, W)/(img_num, 4)
        
        with tf.Session() as sess:

            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            imgs_kep, img_hgts_kep, img_wdhs_kep, img_wdws_kep, \
            gbxs_kep, gbx_nums_kep, gbxs_tmp_kep, gbx_msks_kep, gbx_nums_tmp_kep = \
                sess.run([imgs, img_hgts, img_wdhs, img_wdws, gbxs, gbx_nums, gbxs_tmp, gbx_msks, gbx_nums_tmp])
            #print(gbxs_tmp_kep.shape)
            #print(gbx_msks_kep.shape)
            #print(gbx_nums_tmp_kep)
            for i in range(self.bat_siz):
                img         = imgs_kep[i]
                gbx_num     = gbx_nums_kep[i]
                boxs        = gbxs_kep[i][:gbx_num]
                gbx_num_tmp = gbx_nums_tmp_kep[i][0]
                boxs_tmp    = gbxs_tmp_kep[i][:, :gbx_num_tmp, :] #(4, -1, 5)
                box_msks    = gbx_msks_kep[i] #(4, H, W)
                #print(boxs)
                img_hgt     = img_hgts_kep[i]
                img_wdh     = img_wdhs_kep[i]
                img_wdw     = img_wdws_kep[i]
                self.display_instances(img, boxs[:, 0:-1], boxs[:, -1], None, boxs_tmp[:, :, :-1], box_msks,
                                       img_hgt, img_wdh, img_wdw, title="", figsize=(12, 12), ax=None)
            coord.request_stop()
            coord.join(threads)

    
#from .bboxes_target_layer import generate_bboxes_pre_py
#from Mybase.leye_utils.keys import charset

class GeneratorForICDAR(object):
    
    def __init__(self, fil_nam='train', bat_siz=3, min_after_dequeue=3):
        
        self.mod_tra            = False
        self.use_pad            = False
            
        self.img_siz_min        = 400
        self.img_siz_max        = 512
        self.box_siz_min        = 0.001
        ############for crop###########
        self.min_object_covered = 0.6
        self.aspect_ratio_range = (0.9, 1.1)
        self.area_range         = (0.3, 1.0)
        self.max_attempts       = 200
        
        self.fil_nam            = fil_nam
        self.max_num1           = 100
        self.max_num2           = 300
        self.bat_siz            = bat_siz
        self.min_after_dequeue  = min_after_dequeue
        self.num_threads        = 16
    
        self.cls_nams = ['__background__', 'text']
        self.cat_id_to_cls_name = dict(zip(range(len(self.cls_nams)), self.cls_nams))
        self.cls_name_to_cat_id = dict(zip(self.cls_nams, range(len(self.cls_nams))))

        self.enc_maps = {}
        self.dec_maps = {}
        for i, cha in enumerate(charset, 0):
            self.enc_maps[cha] = i
            self.dec_maps[i] = cha
            
    def make_input(self, num_per_shard=300):
        #image_path_list record_path
        img_pat_lst = ["Mybase/datasets/ali/image_1000", 
                       "Mybase/datasets/ali/image_9000"]
        rcd_pat = "Mybase/tfrecords"

        img_lst = []
        for img_pat in img_pat_lst:
            for ext in ['jpg', 'png', 'jpeg', 'JPG']:
                img_lst.extend(glob.glob(os.path.join(img_pat, '*.{}'.format(ext))))
        np.random.shuffle(img_lst)
        print(len(img_lst))
        gdt_lst = []
        for img in img_lst:
            img_ext = img.split('.')[-1]
            img_bas = img.split('.')[:-1]
            gdt = ".".join(img_bas+["txt"])
            gdt_lst.append(gdt)
        print(len(gdt_lst))
        gbxs_lst = []
        gbx_lbls_lst = []
        gbx_clss_lst = []
        bad_chas_set = []
        for gdt in gdt_lst:
            with open(gdt) as f:
                anns = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf').strip('\ue76c') for x in f.readlines()]
                anns = [x.split(',') for x in anns]
                gbxs = [x[0:8] for x in anns]
                gbxs = [list(map(float, x)) for x in gbxs]
                gbxs = np.asarray(gbxs, dtype=np.float32).reshape(-1, 4, 2)[:, :, ::-1]
                gbx_lbls = [x[8] for x in anns]
                bad_chas = [y for x in gbx_lbls for y in list(x) if y not in charset]
                bad_chas_set.extend(bad_chas)
        bad_chas_set = list(set(bad_chas_set))
        bad_chas_set = "".join(bad_chas_set)
        print(bad_chas_set)
        return bad_chas_set
        '''
                elif "ICDAR2015/" in gt:
                    ann = [x.split(',') for x in ann]
                    gt_boxes = [x[0: 8] for x in ann]
                    gt_labls = [x[8] for x in ann]
                    gt_boxes = [list(map(float, x)) for x in gt_boxes]
                    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4, 2)[:, :, ::-1]
                    gt_labls = np.asarray(gt_labls, dtype=np.str)
                    gt_clses = []
                    for lbl in gt_labls:
                        if lbl == '*' or lbl == '###':
                            gt_clses.append(-1.0)
                        else:
                            gt_clses.append(1.0)
                    gt_clses = np.asarray(gt_clses, dtype=np.float32)

                    gt_boxes = [[x[0], x[1], x[2], x[1], x[2], x[3], x[0], x[3]]  for x in ann]
                    gt_labls = [x[4].strip('\"') for x in ann]
                    gt_boxes = [list(map(float, x)) for x in gt_boxes]
                    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4, 2)[:, :, ::-1]
                    gt_labls = np.asarray(gt_labls, dtype=np.str)
                    gt_clses = []
                    for lbl in gt_labls:
                        if lbl == '*' or lbl == '###':
                            gt_clses.append(-1.0)
                        else:
                            gt_clses.append(1.0)
                    gt_clses = np.asarray(gt_clses, dtype=np.float32)

                else:
                    print("Invalid dataset!")

                area = bbox_area_py2(gt_boxes.reshape(-1, 8))>0
                if np.sum(area>0) != gt_boxes.shape[0]:
                    print("gt_boxes wrong!")
                    print(gt)
                    print(area)
                    print(gt_boxes)

                gt_boxes_list.append(gt_boxes)
                gt_labls_list.append(gt_labls)
                gt_clses_list.append(gt_clses)
        '''
        '''
        print('The dataset has a total of %d images' %(len(im_list)))
        num_shards = int(len(im_list) / num_per_shard)
        if num_shards == 0:
            num_shards = 1
            num_per_shard = len(im_list)
        else:
            num_per_shard = int(math.ceil(len(im_list) / float(num_shards)))
            
        for shard_id in range(num_shards):
                
            output_filename = 'icdar_%05d-of-%05d.tfrecord' % (shard_id, num_shards)
            record_filename = os.path.join(record_path, output_filename)

            options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
            with tf.python_io.TFRecordWriter(record_filename, options=options) as writer:
                    
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(im_list))

                for i in range(start_ndx, end_ndx):
                    if i % 50 == 0:
                        print("Converting image %d/%d shard %d" % (i + 1, len(im_list), shard_id))
                        
                    img = cv2.imread(im_list[i])
                    if type(img) != np.ndarray:
                        print("Failed to find image %s" %(img_ind))
                        continue

                    height, width = img.shape[0], img.shape[1]
                    if img.size == height * width:
                        print ('Gray Image %s' % str(img_ind))
                        im = np.empty((height, width, 3), dtype=np.uint8)
                        im[:, :, :] = img[:, :, np.newaxis]
                        img = im
                    assert img.size == width * height * 3, '%d' % i
                    img = img.astype(np.uint8)
                        
                    gt_boxes = gt_boxes_list[i]
                    gt_labls = gt_labls_list[i]
                    gt_clses = gt_clses_list[i]
                    if len(gt_boxes) == 0:
                        print("No gt_boxes!")
                        continue
                    gt_boxes = gt_boxes / np.array([height-1, width-1], dtype=np.float32)
                    gt_boxes = gt_boxes.reshape(-1, 8)
                    gt_boxes = np.concatenate([gt_boxes, gt_clses[:, np.newaxis]], axis=-1)
                    gt_boxes = gt_boxes.astype(dtype=np.float32, copy=False)

                    img_raw = img.tostring()
                    gt_boxes_raw = gt_boxes.tostring()
                    gt_labls_raw = gt_labls.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image/image':  _bytes_feature(img_raw),
                        'image/height': _int64_feature(height),
                        'image/width':  _int64_feature(width),

                        'label/gt_num': _int64_feature(gt_boxes.shape[0]),  # N
                        'label/gt_boxes': _bytes_feature(gt_boxes_raw),  # of shape (N, 9), (p0, p1, p2, p3, classid)
                        'label/gt_labls': _bytes_feature(gt_labls_raw)
                    }))
                    writer.write(example.SerializeToString())
        '''
        
    def resize_image_with_pad(self, image=None, boxes=None, clses=None):

        #####################按最短边进行比例缩放######################
        im_h = tf.cast(tf.shape(image)[0], dtype=tf.float32)
        im_w = tf.cast(tf.shape(image)[1], dtype=tf.float32)

        l_min = tf.minimum(im_w, im_h)
        l_max = tf.maximum(im_w, im_h)
        rat = tf.minimum(self.img_siz_min/l_min, self.img_siz_max/l_max)
        im_h = tf.cast(im_h*rat, dtype=tf.int32)
        im_w = tf.cast(im_w*rat, dtype=tf.int32)

        image = tf.image.resize_images(image, [im_h, im_w], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        image = tf.image.resize_image_with_crop_or_pad(image, self.img_siz_max, self.img_siz_max)
        
        ################如果最长边过长则按中心对称进行裁剪################
        im_h = tf.cast(im_h, dtype=tf.float32)
        im_w = tf.cast(im_w, dtype=tf.float32)

        pad_h_all = tf.cast(self.img_siz_max-im_h, dtype=tf.float32)
        pad_w_all = tf.cast(self.img_siz_max-im_w, dtype=tf.float32)
        pad_h_fnt = tf.round(pad_h_all/2.0)
        pad_w_fnt = tf.round(pad_w_all/2.0)
        window = tf.stack([pad_h_fnt, pad_w_fnt, pad_h_fnt+im_h-1, pad_w_fnt+im_w-1], axis=0) #该边框在原真实图片内
        window = window / (self.img_siz_max-1)

        box_tmp = tf.stack([im_h-1, im_w-1])
        box_tmp = tf.tile(box_tmp, [4])
        boxes = boxes * box_tmp
        box_tmp = tf.stack([pad_h_fnt, pad_w_fnt])
        box_tmp = tf.tile(box_tmp, [4])
        boxes = boxes + box_tmp
        box_tmp = tf.constant([0, 0, self.img_siz_max-1, self.img_siz_max-1], dtype=tf.float32)
        boxes = bbox_clip2(boxes, box_tmp)
        box_tmp = tf.constant(self.img_siz_max-1, dtype=tf.float32, shape=[8])
        boxes = boxes / box_tmp
        
        box_edgs = bbox_edges2(boxes)
        vld_inds = tf.where(tf.equal(tf.reduce_sum(tf.cast(box_edgs>self.box_siz_min, tf.int32), axis=1), 4))
        boxes = tf.gather_nd(boxes, vld_inds)
        clses = tf.gather_nd(clses, vld_inds)

        clses = tf.expand_dims(clses, axis=-1)
        gt_boxes = tf.concat([boxes, clses], axis=-1)
        gt_boxes = tf.cast(gt_boxes, dtype=tf.float32)

        return image, gt_boxes, window


    def distort_crop(self, image=None, gt_boxes=None):

        im_h = tf.cast(tf.shape(image)[0], dtype=tf.float32)
        im_w = tf.cast(tf.shape(image)[1], dtype=tf.float32)

        boxes = gt_boxes[:, 0:8]
        clses = gt_boxes[:, 8]

        ########################crop the image randomly########################
        no_crowd_id = tf.where(clses>0)
        bbox_tmp = tf.gather_nd(boxes, no_crowd_id)
        bbox_tmp = bbox_bound2(bbox_tmp)
        bbox_begin, bbox_size, distort_bbox = \
            tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=tf.expand_dims(bbox_tmp, 0), \
                                                   min_object_covered=self.min_object_covered, \
                                                   aspect_ratio_range=self.aspect_ratio_range, \
                                                   area_range=self.area_range, max_attempts=self.max_attempts, \
                                                   use_image_if_no_bounding_boxes=True)
            
        distort_bbox = distort_bbox[0, 0] #(batch, N, 4)

        image = tf.slice(image, bbox_begin, bbox_size)

        beg = tf.stack([distort_bbox[0], distort_bbox[1]])
        beg = tf.tile(beg, [4])
        leh = tf.stack([distort_bbox[2]-distort_bbox[0], distort_bbox[3]-distort_bbox[1]])
        leh = tf.tile(leh, [4])
        boxes = boxes - beg
        boxes = boxes / leh
        boxes = bbox_clip2(boxes, [0.0, 0.0, 1.0, 1.0])
        
        box_edgs = bbox_edges2(boxes)
        vld_inds = tf.where(tf.equal(tf.reduce_sum(tf.cast(box_edgs>self.box_siz_min, tf.int32), axis=1), 4))
        boxes = tf.gather_nd(boxes, vld_inds)
        clses = tf.gather_nd(clses, vld_inds)

        ########################resize image to the expected size with paddings########################
        image, gt_boxes, window = self.resize_image_with_pad(image, boxes, clses)

        return image, gt_boxes, window


    def preprocessing(self, image=None, gt_boxes=None):

        self.img_avg = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        self.img_avg = self.img_avg / 255.0

        ####################归化到0、1之间######################
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.mod_tra == True:
            #####################光学畸变###########################
            # Randomly distort the colors. There are 4 ways to do it.
            distorted_image = apply_with_random_selector(image, lambda x, order: distort_color(x, order), num_cases=4)

            #####################随机裁剪###########################
            distorted_image, distorted_gt_boxes, distorted_window = self.distort_crop(distorted_image, gt_boxes)
            '''
            boxes = gt_boxes[:, 0:8] 
            clses = gt_boxes[:, 8]
            distorted_image, distorted_gt_boxes, distorted_window = self.resize_image_with_pad(distorted_image, boxes, clses)
            '''
            #####################随机翻转##########################
            signal = tf.random.uniform([2])

            #####################随机左右翻转#######################
            #distorted_image = tf.image.random_flip_left_right(distorted_image)
            image_left_right = tf.image.flip_left_right(distorted_image)
            gt_boxes_left_right = tf.stack([distorted_gt_boxes[:, 0], 1.0-distorted_gt_boxes[:, 3], \
                                            distorted_gt_boxes[:, 2], 1.0-distorted_gt_boxes[:, 1], \
                                            distorted_gt_boxes[:, 4], 1.0-distorted_gt_boxes[:, 7], \
                                            distorted_gt_boxes[:, 6], 1.0-distorted_gt_boxes[:, 5], \
                                            distorted_gt_boxes[:, 8]], axis=-1)
            window_left_right = tf.stack([distorted_window[0], 1.0-distorted_window[3], \
                                          distorted_window[2], 1.0-distorted_window[1]], axis=-1)

            distorted_image    = tf.cond(signal[0]<0.5, lambda: image_left_right,    lambda: distorted_image)
            distorted_gt_boxes = tf.cond(signal[0]<0.5, lambda: gt_boxes_left_right, lambda: distorted_gt_boxes)
            distorted_window   = tf.cond(signal[0]<0.5, lambda: window_left_right,   lambda: distorted_window)

            #distorted_image = tf.image.per_image_standardization(image)
            distorted_image = distorted_image - self.img_avg
            return distorted_image, distorted_gt_boxes, distorted_window
        else:
            boxes = gt_boxes[:, 0:8] 
            clses = gt_boxes[:, 8]
            image, gt_boxes, window = self.resize_image_with_pad(image, boxes, clses)   
            image = image - self.img_avg
            return image, gt_boxes, window


    def get_input(self):
        #创建文件列表，并通过文件列表创建输入文件队列。
        #在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中
        #文件列表应该包含所有提供训练数据的TFRecord文件
        filename = os.path.join("Mybase/tfrecords", self.filename, "*.tfrecord")
        files = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(files, shuffle = True, capacity=1000)

        #解析TFRecord文件里的数据
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/image':  tf.FixedLenFeature([], tf.string),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width':  tf.FixedLenFeature([], tf.int64),

                'label/gt_num':   tf.FixedLenFeature([], tf.int64),
                'label/gt_boxes': tf.FixedLenFeature([], tf.string),
                'label/gt_labls': tf.FixedLenFeature([], tf.string),
            }
        )
        
        height = tf.cast(features['image/height'], tf.int32)
        width  = tf.cast(features['image/width'],  tf.int32)
        gt_num = tf.cast(features['label/gt_num'], tf.int32)

        image    = tf.decode_raw(features['image/image'],    tf.uint8)
        gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
        gt_labls = features['label/gt_labls']
    
        image    = tf.reshape(image, [height, width, 3])
        gt_boxes = tf.reshape(gt_boxes, [gt_num, 9])
        
        image, gt_boxes, window = self.preprocessing(image, gt_boxes)

        image  = tf.reshape(image, [self.img_siz_max, self.img_siz_max, 3])
        window = tf.reshape(window, [4])
        gt_num = tf.shape(gt_boxes)[0]

        paddings = [[0, self.max_num-gt_num], [0, 0]]
        gt_boxes = tf.pad(gt_boxes, paddings)
        gt_boxes = tf.reshape(gt_boxes, [self.max_num, 9])

        capacity = self.min_after_dequeue + 3 * self.batch_size
        #tf.train.shuffle_batch_join
        
        images, heights, widths, gt_boxes, gt_nums, windows = tf.train.shuffle_batch(
            tensors=[image, height, width, gt_boxes, gt_num, window], batch_size=self.batch_size, \
            num_threads=self.num_threads, capacity=capacity, min_after_dequeue=self.min_after_dequeue)
        '''
        boxes = []
        for i in range(batch_size):
            boxes.append(gt_boxes[i][:gt_nums[i]])
        '''
        return images, heights, widths, gt_boxes, gt_nums, windows
    
    
    def random_colors(self, N, bright=True):
        '''
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        '''
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors


    def apply_mask(self, image, mask, color, alpha=0.5):
        '''
        Apply the given mask to the image.
        '''
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

        
    def display_instances(self, image=None, boxes=None, cls_ids=None, scores=None,
                          height=None, width=None, window=None, title="", figsize=(16, 16), ax=None):
        '''
        Args:
            boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
            figsize: (optional) the size of the image.
        '''
        self.img_avg = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        self.img_avg = self.img_avg / 255.0
        
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            return

        if not ax:
            _, ax = plt.subplots(1, figsize=figsize)

        # Generate random colors
        colors = self.random_colors(1)

        beg = np.array([window[0], window[1]], dtype=np.float32)
        beg = np.tile(beg, [4])
        leh = np.array([window[2]-window[0], window[3]-window[1]], dtype=np.float32)
        leh = np.tile(leh, [4])
        boxes = (boxes - beg) / leh
        boxes = bbox_clip_py2(boxes, [0.0, 0.0, 1.0, 1.0])
        
        box_tmp = np.array([height-1, width-1], dtype=np.float32)
        box_tmp = np.tile(box_tmp, [4])
        boxes = boxes * box_tmp
        boxes = np.around(boxes).astype(np.int32, copy=False)
        boxes = boxes.reshape([-1, 4, 2])[:, :, ::-1]

        window = (window * (self.img_siz_max-1)).astype(dtype=np.int32)
        image = image[window[0]:window[2]+1, window[1]:window[3]+1, :] #因为window在原真实图片内
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        # Show area outside image boundaries.
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)

        #image = image * 255
        #image = image.astype(np.float32, copy=False)
        image = image + self.img_avg
        image = np.clip(image, 0.0, 1.0)

        for i in range(N):

            color = colors[0]
            '''
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="solid",
                                  edgecolor=color, facecolor='none')
            '''
            x1 = boxes[i, 0, 0]
            y1 = boxes[i, 0, 1]
            p = patches.Polygon(boxes[i], facecolor='none', edgecolor=color, linewidth=2, linestyle='-', fill=True)
            ax.add_patch(p)

            # Label
            cls_id = cls_ids[i]
            if cls_id < 0:
                cls_id *= -1
            scor = scores[i] if scores is not None else None
            clss = self.cat_id_to_cls_name[cls_id]

            caption = "{} {:.3f}".format(clss, scor) if scor else clss
            ax.text(x1-15, y1-8, caption, color='w', size=11, backgroundcolor="none")

        ax.imshow(image)
        plt.show()
        plt.close()

    
    def display_detections(images, heights, widths, windows, bboxes, bbox_imid, bbox_clss, bbox_prbs, bbox_msks):

        img_num = len(images)

        for i in range(img_num):

            image  = images[i]
            height = heights[i]
            width  = widths[i]
            window = windows[i]

            inds = np.where(bbox_imid==i)[0]
            bboxes_img = bboxes[inds]
            bbox_clss_img = bbox_clss[inds]
            bbox_prbs_img = bbox_prbs[inds]
            bbox_msks_img = bbox_msks[inds]

            display_instances(image, bbox_msks_img, bboxes_img, bbox_clss_img, bbox_prbs_img, 1024, \
                              height, width, window, title="", figsize=(13, 13), ax=None)
    
    
    def write_image(self):
        '''
        if boxes is not None:
            res_file = os.path.join(FLAGS.output_dir, '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            if not FLAGS.no_write_images:
                img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                cv2.imwrite(img_path, im[:, :, ::-1])
        '''
        
    def get_input_test(self):
        
        tf.reset_default_graph()
        with tf.device("/cpu:0"):
            images, heights, widths, gt_boxes, gt_nums, windows = self.get_input()

        with tf.Session() as sess:

            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            images, heights, widths, gt_boxes, gt_nums, windows \
                = sess.run([images, heights, widths, gt_boxes, gt_nums, windows])
            
            #print(images.shape)
            #print(heights)
            #print(widths)
            #print(gt_boxes)
            #print(gt_nums)
            #print(windows)
            
            for i in range(self.batch_size):

                image  = images[i]
                boxes  = gt_boxes[i]
                height = heights[i]
                width  = widths[i]
                window = windows[i]

                #boxid = np.zeros(shape=[boxes.shape[0]], dtype=np.int32)
                #probs = np.ones(shape=[boxes.shape[0]], dtype=np.float32)
                #stats = generate_bboxes_pre_py(boxes[:, 0:4], boxid, boxes[:, 4], probs, masks, \
                #                               [imid], [window], [height], [width])
                #print(stats)

                self.display_instances(image, boxes[:, 0:8], boxes[:, 8], None, 
                                       height, width, window, title="", figsize=(12, 12), ax=None)
            
            coord.request_stop()
            coord.join(threads)
"""

"""
class GeneratorForCRNN(object):
    
    def __init__(self, batch_size=32, min_len=4, max_len=10, box_h=32, pool_scale=4):
        
        #from .keys import alphabet
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.box_h   = box_h
        self.pool_scale = pool_scale
        #self.charset = alphabet[:]
        self.charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        #self.font    = '/home/ziyechen/MyCTPN_CRNN_CTC/Mybase/ctpn_crnn_ctc_utils/fonts/simsun.ttf'
        self.num_classes = len(self.charset) + 2
        
        self.encode_maps = {}
        self.decode_maps = {}
        for i, char in enumerate(self.charset, 1):
            self.encode_maps[char] = i
            self.decode_maps[i] = char
        self.encode_maps[' '] = 0
        self.decode_maps[0] = ' '

    def randRGB(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def gen_rand(self):
        buf = ""
        max_len = random.randint(self.min_len, self.max_len)
        for i in range(max_len):
            buf += random.choice(self.charset)
        
        buf = list(buf)
        bnk_loc = random.randint(1, max_len-1)
        buf[bnk_loc] = ' '
        buf = "".join(buf)
        return buf

    def generateImg(self):
        #captcha = ImageCaptcha(fonts=[self.font])
        #if not os.path.exists(self.font):
        #    print('cannot open the font')
        captcha = ImageCaptcha(width=560, height=70)
        label = self.gen_rand()
        image = captcha.generate_image(label)
        return np.array(image), label

    def groupBatch(self, images, labels):
        
        max_w = 0
        imgs_keep = []
        slns_keep = [] #seq_lengths
        
        indices = []
        values = []
        
        for img in range(len(images)):
            
            label = labels[img]
            label = [self.encode_maps[c] for c in list(label)]
            indices.extend(zip([img]*len(label), [i for i in range(len(label))]))
            values.extend(label)
            
            image = images[img]
            img_h, img_w = image.shape[:2]

            ratio = self.box_h / img_h
            box_w = int(ratio * img_w)
            max_w = max(max_w, box_w)
            image = cv2.resize(image, (box_w, self.box_h), interpolation=cv2.INTER_LINEAR)
            
            box_w = len(label) * 13 #!!!!!!!!根据验证码产生的特殊情况而设定
            #box_w = 256 #!!!!!!!!根据验证码产生的特殊情况而设定
            slen = int(box_w/self.pool_scale) - 1
            imgs_keep.append(image)
            slns_keep.append(slen)

        imgs_keep = np.stack(imgs_keep, axis=0)
        slns_keep = np.stack(slns_keep, axis=0)
        
        indices = np.asarray(indices, dtype=np.int64)
        values  = np.asarray(values,  dtype=np.int32)
        shape   = np.asarray([len(labels), np.asarray(indices).max(axis=0)[1]+1], dtype=np.int64)
        labels  = (indices, values, shape)
        
        max_w = math.ceil(max_w/self.pool_scale) * self.pool_scale
    
        #for img in range(len(images)):
        #    image = imgs_keep[img]
        #    paddings = [[0, 0], [0, max_w-box_w], [0, 0]]
        #    image = np.pad(image, paddings, 'constant', constant_values=0)
        #    imgs_keep[img] = image
        #imgs_keep = np.stack(imgs_keep, axis=0)
        
        return imgs_keep, labels, slns_keep, max_w

    def generate(self):
        images = []
        labels = []
        while True:
            try:
                image, label = self.generateImg()
                #if cfg.NCHANNELS == 1: im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                images.append(image)
                labels.append(label)
                if len(images) == self.batch_size:
                    images, lables, slens, max_w = self.groupBatch(images, labels)
                    #string = self.decode(lables)
                    #print(string)
                    yield images, lables, slens, max_w
                    images = []
                    labels = []
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                continue

    
    def decode(self, decoded=None):
        
        decoded_indices = decoded[0]
        decoded_values  = decoded[1]
        decoded_shape   = decoded[2]
        
        decoded_strs = []
        for img in range(decoded_shape[0]):
            
            inds = np.where(decoded_indices[:, 0]==img)
            decoded_dats_img = decoded_values[inds]
            
            decoded_strs_img = [self.decode_maps[i] for i in decoded_dats_img]
            decoded_strs_img = "".join(decoded_strs_img)
            decoded_strs.append(decoded_strs_img)
         
        return decoded_strs
                
    def evaluate(self, decoded=None, labels=None):
        
        decoded_strs = self.decode(decoded)
        labels_strs  = self.decode(labels)
        #print("############")
        #print(decoded_strs)
        #print("!!!!!!!!!!!!")
        #print(labels_strs)
        assert len(decoded_strs) == len(labels_strs), "The number of the decoded and the number of the labels are mismatched"
        
        acc = []
        for s in range(len(decoded_strs)):
            if decoded_strs[s] == labels_strs[s]:
                acc.append(1.0)
            else:
                acc.append(0.0)
        
        mean_acc = np.mean(acc)
        return mean_acc
"""