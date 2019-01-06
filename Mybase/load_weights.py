import h5py
import pickle
import numpy as np

def load_weights():
    #fff = h5py.File('Mybase/mask_rcnn_coco.h5','r')   #打开h5文件  
    #print(list(f.keys()))
    fff = np.load('Mybase/Model/vgg16.npz')
    #print(fff.files)
    mydict = {}
    mydict['global_step:0'] = 0
    '''
    dset = fff['conv1']
    a = dset['conv1']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    '''
    w = fff['conv1_1/W'].transpose(2, 3, 1, 0)
    b = fff['conv1_1/b']
    mydict['generator/layers_module1_0/conv_relu1_1/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_0/conv_relu1_1/conv1_0/biases:0']  = b
    
    w = fff['conv1_2/W'].transpose(2, 3, 1, 0)
    b = fff['conv1_2/b']
    mydict['generator/layers_module1_0/conv_relu1_2/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_0/conv_relu1_2/conv1_0/biases:0']  = b
    
    w = fff['conv2_1/W'].transpose(2, 3, 1, 0)
    b = fff['conv2_1/b']
    mydict['generator/layers_module1_0/conv_relu1_4/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_0/conv_relu1_4/conv1_0/biases:0']  = b
    
    w = fff['conv2_2/W'].transpose(2, 3, 1, 0)
    b = fff['conv2_2/b']
    mydict['generator/layers_module1_0/conv_relu1_5/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_0/conv_relu1_5/conv1_0/biases:0']  = b
    
    w = fff['conv3_1/W'].transpose(2, 3, 1, 0)
    b = fff['conv3_1/b']
    mydict['generator/layers_module1_0/conv_relu1_7/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_0/conv_relu1_7/conv1_0/biases:0']  = b
    
    w = fff['conv3_2/W'].transpose(2, 3, 1, 0)
    b = fff['conv3_2/b']
    mydict['generator/layers_module1_0/conv_relu1_8/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_0/conv_relu1_8/conv1_0/biases:0']  = b
    
    w = fff['conv3_3/W'].transpose(2, 3, 1, 0)
    b = fff['conv3_3/b']
    mydict['generator/layers_module1_0/conv_relu1_9/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_0/conv_relu1_9/conv1_0/biases:0']  = b
    
    w = fff['conv4_1/W'].transpose(2, 3, 1, 0)
    b = fff['conv4_1/b']
    mydict['generator/layers_module1_1/conv_relu1_0/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_1/conv_relu1_0/conv1_0/biases:0']  = b
    
    w = fff['conv4_2/W'].transpose(2, 3, 1, 0)
    b = fff['conv4_2/b']
    mydict['generator/layers_module1_1/conv_relu1_1/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_1/conv_relu1_1/conv1_0/biases:0']  = b
    
    w = fff['conv4_3/W'].transpose(2, 3, 1, 0)
    b = fff['conv4_3/b']
    mydict['generator/layers_module1_1/conv_relu1_2/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_1/conv_relu1_2/conv1_0/biases:0']  = b
    
    w = fff['conv5_1/W'].transpose(2, 3, 1, 0)
    b = fff['conv5_1/b']
    mydict['generator/layers_module1_2/conv_relu1_0/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_2/conv_relu1_0/conv1_0/biases:0']  = b
    
    w = fff['conv5_2/W'].transpose(2, 3, 1, 0)
    b = fff['conv5_2/b']
    mydict['generator/layers_module1_2/conv_relu1_1/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_2/conv_relu1_1/conv1_0/biases:0']  = b
    
    w = fff['conv5_3/W'].transpose(2, 3, 1, 0)
    b = fff['conv5_3/b']
    mydict['generator/layers_module1_2/conv_relu1_2/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_2/conv_relu1_2/conv1_0/biases:0']  = b
    
    w = fff['fc6/W'].reshape(4096,  512, 7, 7).transpose(2, 3, 1, 0)
    b = fff['fc6/b']
    mydict['generator/layers_module1_2/conv_relu_dropout1_4/conv_relu1_0/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_2/conv_relu_dropout1_4/conv_relu1_0/conv1_0/biases:0']  = b
    
    w = fff['fc7/W'].reshape(4096, 4096, 1, 1).transpose(2, 3, 1, 0)
    b = fff['fc7/b']
    mydict['generator/layers_module1_2/conv_relu_dropout1_5/conv_relu1_0/conv1_0/weights:0'] = w
    mydict['generator/layers_module1_2/conv_relu_dropout1_5/conv_relu1_0/conv1_0/biases:0']  = b
    return mydict