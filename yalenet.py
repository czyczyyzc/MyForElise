import copy
import numpy as np
import tensorflow as tf

from Mybase import layers
from Mybase.layers import *
from Mybase.layers_utils import *
from Mybase.losses import *


class YaleNet(object):
    
    def __init__(self, cls_num=10, reg=1e-4, typ=tf.float32):
        
        self.cls_num  = cls_num
        self.reg      = reg
        self.typ      = typ
        
        self.mod_tra  = True
        self.use_gan  = False
        self.msk_min  = 0.5
        
        #the resnet block setting
        self.res_set  = [( 256,  64, [1, 1], [1, 1],  3, True ),  #conv2x 256 /4     #use        56*56*64      56*56*256
                         ( 512, 128, [1, 1], [2, 2],  4, True ),  #conv3x 128 /8     #use                      28*28*512
                         (1024, 256, [1, 1], [2, 2],  6, True ),  #conv4x 64  /16    #use #23--->101 #6--->50  14*14*1024
                         (2048, 512, [1, 1], [2, 2],  3, True )]  #conv5x 32  /32    #use                      7*7*2048
        self.out_srd  = 8       #output stride
        
        self.grp_set  = [([[ 1,1,  64],[1, 1,  64],[ 1,1, 256]], [3,3], [1,1], 3, True ), #  256    64   128*128*   256
                         ([[ 2,1, 128],[1, 2, 128],[ 2,1, 512]], [3,3], [1,1], 4, True ), # 1024   256    64* 64*  1024
                         ([[ 4,1, 256],[1, 4, 256],[ 4,1,1024]], [3,3], [1,1], 6, True ), # 4096  1024    32* 32*  4096
                         ([[ 8,1, 512],[1, 8, 512],[ 8,1,2048]], [3,3], [1,1], 3, True )] #16384  4096    16* 16* 16384
        
        self.com_pams = {
            'com':   {'reg':self.reg, 'wscale':0.01, 'dtype':self.typ, 'reuse':False, 'is_train':False, 'trainable':True},
            'pad':   {'shape':[100,100], 'mode':'CONSTANT'},
            'bn':    {'axis':-1, 'decay':0.99, 'eps':1e-3},
            'relu':  {'alpha':-0.1},
            'conv':  {'number':64,    'shape':[7,7],'rate':[1,1],'stride':[1,1],'padding':'SAME','use_bias':True },
            'proj':  {'number':[4,16],'shape':[3,3],'rate':[1,1],'stride':[1,1],'padding':'SAME','use_bias':True },
            'deconv':{'number':64,    'shape':[7,7],'rate':1,    'stride':[2,2],'padding':'SAME','use_bias':True, 'out_shape':None},
            'unfold':{'stride':[2,2], 'use_crs':False},
            'max_pool':    {'shape':[3,3], 'stride':[2,2], 'padding':'VALID'},
            'group_block': {'block_setting':self.grp_set},
            'resnet_block':{'block_setting':self.res_set, 'output_stride':self.out_srd},
            'glb_pool': {'axis':  [1, 2]},
            'reshape':  {'shape': []},
            'resize':   {'shape': []},
            'expand':   {'axis':  3},
            'squeeze':  {'axis':  [1, 2]},
            'transpose':{'perm':  [0, 3, 1, 2, 4]},
            'affine':   {'dim':   10, 'use_bias':False},
            'dropout':  {'keep_p':0.5, 'shape':None},
            #'bilstm':  {'num_h': self.fet_dep//2, 'num_o': None, 'fbias': 1.0, 'tmajr': False},
            #'concat':  {'axis': 0},
            #'split':   {'axis': 0, 'number': img_num},
        }
    """
    def forward(self, imgs=None, lbls=None, mtra=None, scp=None):
        img_shp  = imgs.get_shape().as_list()
        img_num, img_hgt, img_wdh = img_shp[0], img_shp[1], img_shp[2]
        img_shp  = np.stack([img_hgt, img_wdh], axis=0)
        com_pams = copy.deepcopy(self.com_pams)
        '''
        #####################Get the first feature map!####################
        print('Get the first feature map!')
        opas    = {'op':[{'op':'conv_bn_relu1', 'loop':1, 'params':{'conv':{'stride':[2,2]}}},
                         {'op':'max_pool1',     'loop':1, 'params':{}},
                        ], 'loop':1}
        tsr_out = layers_module1(imgs, 0, com_pams, opas, mtra)
        print('')
        ###################Get the first resnet block!#####################
        print('Get the first resnet block!')
        fet_lst = []
        opas    = {'op':[{'op':'resnet_block2', 'loop':1, 'params':{}}], 'loop':1}
        fet_lst.extend(layers_module1(tsr_out, 1, com_pams, opas, mtra))
        assert len(fet_lst) == 4, 'The resnet block is worng!'
        tsr_out = fet_lst[-1]
        print('')
        '''
        #####################Get the first feature map!####################
        print('Get the first feature map!')
        opas    = {'op':[{'op':'conv_bn_relu1', 'loop':2, 'params':{'conv':{'stride':[2,2]}}}], 'loop':1}
        tsr_out = layers_module1(imgs,    0, com_pams, opas, mtra)
        print('')
        ###################Get the first resnet block!#####################
        print('Get the first resnet block!')
        opas    = {'op':[{'op':'group_block1',  'loop':1, 'params':{}}], 'loop':1}
        fet_lst = layers_module1(tsr_out, 1, com_pams, opas, mtra)
        tsr_out = fet_lst[-1]
        opas    = {'op':[{'op':'unfold1',       'loop':1, 'params':{}}], 'loop':1}
        tsr_out = layers_module1(tsr_out, 2, com_pams, opas, mtra)
        print('')
        
        ################Get the image classification results!##############
        print('Get the image classification results!')
        com_pams['conv'] = {'number':self.cls_num,'shape':[1,1],'rate':[1,1],'stride':[1,1],'padding':'SAME','use_bias':True}
        opas = {'op':[{'op':'glb_pool1', 'loop':1, 'params':{}},
                      {'op':'conv1',     'loop':1, 'params':{}},
                      {'op':'squeeze1',  'loop':1, 'params':{}},
                     ], 'loop':1}
        scrs     = layers_module1(tsr_out, 99, com_pams, opas, mtra) #class scores
        
        acc_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(scrs, lbls, k=1), tf.float32))
        acc_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(scrs, lbls, k=5), tf.float32))
        accs     = tf.stack([acc_top1, acc_top5], axis=0)
        los_dat  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbls, logits=scrs))
        los_reg  = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        los      = los_dat + los_reg
        loss     = tf.stack([los, los_dat, los_reg], axis=0)
        return loss, accs
    """
    def forward(self, imgs=None, gmks=None, mtra=None, scp=None):
        
        img_shp  = imgs.get_shape().as_list()
        img_num, img_hgt, img_wdh = img_shp[0], img_shp[1], img_shp[2]
        img_shp  = np.stack([img_hgt, img_wdh], axis=0)
        com_pams = copy.deepcopy(self.com_pams)
        
        #####################Get the first feature map!####################
        print('Get the first feature map!')
        opas     = {'op':[{'op':'conv_bn_relu1',   'loop':1, 'params':{'conv':{'stride':[2,2]}}},
                          {'op':'conv_bn_relu1',   'loop':1, 'params':{}},
                         ], 'loop':1}
        tsr_out  = layers_module1(imgs,    0, com_pams, opas, mtra)
        print('')
        ###################Get the first resnet block!#####################
        print('Get the first resnet block!')
        opas     = {'op':[{'op':'group_block1',    'loop':1, 'params':{}}], 'loop':1}
        fet_lst  = layers_module1(tsr_out, 1, com_pams, opas, mtra) #(N, 4, 4, 262144)
        tsr_out  = fet_lst[-1]
        opas     = {'op':[{'op':'unfold1',         'loop':4, 'params':{}},
                          #{'op':'deconv_bn_relu1', 'loop':1, 'params':{'deconv':{'out_shape':[img_hgt, img_wdh]}}},
                          {'op':'resize1',        'loop':1, 'params':{'resize':{'shape':[img_hgt, img_wdh]}}},
                          {'op':'conv1',           'loop':1, 'params':{'conv':{'number':self.cls_num}}},
                         ], 'loop':1}
        msks_pst = layers_module1(tsr_out, 2, com_pams, opas, mtra) #(N, 256, 256, 64)
        print('')
        
        msks_pst_= tf.nn.softmax(msks_pst, axis=-1)                 #(N, H, W, C)
        msks_pre = gmks
        msks_pre_= tf.one_hot(msks_pre, depth=self.cls_num, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)
        msks     = tf.argmax(msks_pst_, axis=-1, output_type=tf.int32)
        
        if self.mod_tra: 
            los_dat   = self.loss_seg(msks_pst, msks_pre)
            #los_dat  = self.loss_seg(msks_pst_, msks_pre_)
            los_reg   = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            los       = los_dat + los_reg
            loss      = tf.stack([los, los_dat, los_reg], axis=0)
            accs      = self.accs_seg(msks, gmks)
            return loss, accs, msks
        else:
            return msks
    
    def loss_seg(self, msks_pst=None, msks_pre=None):
        msks_los     = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=msks_pre, logits=msks_pst) #(N, H, W)
        '''
        fgd_idxs     = tf.where(msks_pre>=1)
        msks_los_fgd = tf.reduce_mean(tf.gather_nd(msks_los, fgd_idxs))
        bgd_idxs     = tf.where(tf.equal(msks_pre, 0))
        msks_los_bgd = tf.reduce_mean(tf.gather_nd(msks_los, bgd_idxs))
        msks_los     = msks_los_fgd * 5.0 + msks_los_bgd * 1.0
        '''
        msks_los     = tf.reduce_sum(msks_los, axis=[1,2])
        msks_los     = tf.reduce_mean(msks_los)
        return msks_los
    '''
    def loss_seg(self, msks_pst=None, msks_pre=None):
        smooth   = 1
        msks_int = msks_pst * msks_pre                                        #(N, H, W, C)
        msks_int = tf.reduce_sum(msks_int, axis=[1,2])                        #(N, C)
        msks_pst = tf.reduce_sum(msks_pst, axis=[1,2])                        #(N, C)
        msks_pre = tf.reduce_sum(msks_pre, axis=[1,2])                        #(N, C)
        msks_los = (2.0 * msks_int + smooth) / (msks_pst + msks_pre + smooth) #(N, C)
        msks_los = 1.0 - msks_los                                             #(N, C)
        msks_los = tf.reduce_sum(msks_los, axis=1)                            #(N)
        msks_los = tf.reduce_mean(msks_los, axis=0)                           #()
        return msks_los
    '''
    def accs_seg_img(self, msks_pst=None, msks_pre=None):
        idxs         = tf.where((msks_pre>=0)&(msks_pre<self.cls_num))
        msks_pre     = tf.gather_nd(msks_pre, idxs)
        msks_pst     = tf.gather_nd(msks_pst, idxs)
        msk          = msks_pre * self.cls_num + msks_pst #一个真值对应一个预测值
        msk_cnt      = tf.bincount(msk, weights=None, minlength=self.cls_num*self.cls_num, maxlength=None, dtype=tf.float32)
        msk_cnt      = tf.reshape(msk_cnt, [self.cls_num, self.cls_num])
        pre_cls      = tf.reduce_sum(msk_cnt, axis=0)
        pst_cls      = tf.reduce_sum(msk_cnt, axis=1)
        pst_cls_tru  = tf.linalg.diag_part(msk_cnt)
        accs         = tf.stack([pre_cls, pst_cls, pst_cls_tru], axis=0) #(3, 21)
        return accs
    
    def accs_seg(self, msks_pst=None, msks_pre=None):
        #msks_pst --> (N, H, W)
        img_num = get_shape(msks_pst)[0]       # N
        elems   = [msks_pst, msks_pre]
        accs    = tf.map_fn(lambda x: self.accs_seg_img(x[0], x[1]), elems, dtype=tf.float32, \
                            parallel_iterations=img_num, back_prop=False, swap_memory=False, infer_shape=True)
        return accs #(N, 3, C)
    
    def accs_seg_py(self, accs=None):
        pre_cls     = accs[:, 0, :]               #(N, C)
        pst_cls     = accs[:, 1, :]               #(N, C)
        pst_cls_tru = accs[:, 2, :]               #(N, C)
        pre_cls     = np.sum(pre_cls,     axis=0) #(C)
        pst_cls     = np.sum(pst_cls,     axis=0) #(C)
        pst_cls_tru = np.sum(pst_cls_tru, axis=0) #(C)
        iou_int     = pst_cls_tru
        iou_uin     = pst_cls + pre_cls - pst_cls_tru
        iou_cls     = iou_int / iou_uin
        iou_avg     = np.nanmean(iou_cls)
        accs        = np.concatenate([[iou_avg], iou_cls], axis=0)
        return accs
    '''
    def accs_seg(self, msks_pst=None, msks_pre=None):
        iou_avg, opa0 = tf.metrics.mean_iou(labels=msks_pre, predictions=msks_pst, \
                                            num_classes=self.cls_num, weights=None, \
                                            metrics_collections=None, \
                                            updates_collections=None)
        iou_cls, opa1 = tf.metrics.mean_per_class_accuracy(labels=msks_pre, predictions=msks_pst, \
                                                           num_classes=self.cls_num, weights=None, \
                                                           metrics_collections=None, \
                                                           updates_collections=None)
        tf.add_to_collection('upd_opas', opa0)
        tf.add_to_collection('upd_opas', opa1)
        accs = tf.stack([iou_avg, iou_cls], axis=0)
        accs = tf.expand_dims(accs, axis=0)
        return accs
    
    def accs_seg(self, msks_pst=None, msks_pre=None):
        
        smooth   = 1e-8
        msks_int = msks_pst * msks_pre                                        #(N, H, W, C)
        msks_int = tf.reduce_sum(msks_int, axis=[1,2])                        #(N, C)
        msks_pst = tf.reduce_sum(msks_pst, axis=[1,2])                        #(N, C)
        msks_pre = tf.reduce_sum(msks_pre, axis=[1,2])                        #(N, C)
        msks_los = 2.0 * msks_int / (msks_pst + msks_pre + smooth)            #(N, C)
        return msks_acc
    '''