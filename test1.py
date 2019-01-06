import time
import numpy as np
import tensorflow as tf
from yalenet import YaleNet
from Mybase.solver import Solver

"""
def test():
    
    mdl = YaleNet(cls_num=1000, reg=1e-4, typ=tf.float32)
    sov = Solver(mdl,
                 opm_cfg={
                     'lr_base':      0.005,
                     'decay_rule':  'fixed',
                     #'decay_rule':  'exponential',
                     'decay_rate':   0.5,
                     'decay_step':   50,
                     'staircase':    False,
                     #'optim_rule':  'adam',
                     'optim_rule':  'momentum',
                     'momentum':     0.9,
                     'use_nesterov': True
                 },
                 gpu_lst     =    '0',
                 bat_siz     =     50,
                 tra_num     =   2000,
                 val_num     =    100,
                 epc_num     = 200000,
                 min_que_tra =  10000,
                 min_que_val =   1000,
                 prt_ena     =   True,
                 itr_per_prt =     20,
                 tst_num     =   None,
                 tst_shw     =   True,
                 tst_sav     =   True,
                 mdl_nam     = 'model.ckpt',
                 mdl_dir     = 'Mybase/Model',
                 log_dir     = 'Mybase/logdata',
                 dat_dir     = 'Mybase/datasets',
                 mov_ave_dca = 0.99)
    print('TRAINING...')
    sov.train()
    '''
    print('TESTING...')
    sov.test()
    sov.display_detections()
    #sov.show_loss_acc()
    '''
"""
def test():
    
    mdl = YaleNet(cls_num=21, reg=1e-4, typ=tf.float32)
    sov = Solver(mdl,
                 opm_cfg={
                     'lr_base':      1e-5,
                     'decay_rule':  'fixed',
                     #'decay_rule':  'exponential',
                     'decay_rate':   0.5,
                     'decay_step':   50,
                     'staircase':    False,
                     #'optim_rule':  'adam',
                     'optim_rule':  'momentum',
                     'momentum':     0.9,
                     'use_nesterov': True
                 },
                 gpu_lst     = '0,1,2,3',
                 bat_siz     =      4,
                 tra_num     =   2000,
                 val_num     =    100,
                 epc_num     = 200000,
                 min_que_tra =   4000,
                 min_que_val =    200,
                 prt_ena     =   True,
                 itr_per_prt =     20,
                 tst_num     =   None,
                 tst_shw     =   True,
                 tst_sav     =   True,
                 mdl_nam     = 'model.ckpt',
                 mdl_dir     = 'Mybase/Model',
                 log_dir     = 'Mybase/logdata',
                 dat_dir     = 'Mybase/datasets',
                 mov_ave_dca = 0.99)
    print('TRAINING...')
    sov.train()
    '''
    print('TESTING...')
    #sov.test()
    sov.display_detections()
    #sov.show_loss_acc()
    '''

test()
