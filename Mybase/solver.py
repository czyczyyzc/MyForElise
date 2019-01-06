import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from .load_weights import *
from .optim_utils import *
from .yale_utils.make_image import *

def get_data(fid):
    try:  
            a = pickle.load(fid)
            return 1, a
    except EOFError:  
            return 0, 0

def get_all_data(fid):
    data = []
    while(True):
        sig, dat = get_data(fid)
        if(sig == 0): break
        else:
            data.append(dat)
    return data

class Solver(object):
    
    def __init__(self, mdl, **kwargs):
        
        self.mdl         = mdl
        self.opm_cfg     = kwargs.pop('opm_cfg',       {})
        self.gpu_lst     = kwargs.pop('gpu_lst',      '0')
        self.gpu_num     = len(self.gpu_lst.split(','))
        self.mdl_dev     = '/cpu:%d' if self.gpu_num == 0 else '/gpu:%d'
        self.MDL_DEV     = 'CPU_%d'  if self.gpu_num == 0 else 'GPU_%d'
        self.gpu_num     = 1         if self.gpu_num == 0 else self.gpu_num
        self.bat_siz     = kwargs.pop('bat_siz',        2)
        self.bat_siz_all = self.bat_siz * self.gpu_num
        self.tra_num     = kwargs.pop('tra_num',     8000)
        self.val_num     = kwargs.pop('val_num',       80)
        self.epc_num     = kwargs.pop('epc_num',       10) 
        self.min_que_tra = kwargs.pop('min_que_tra', 5000)
        self.min_que_val = kwargs.pop('min_que_val', 1000)
        self.prt_ena     = kwargs.pop('prt_ena',     True)
        self.itr_per_prt = kwargs.pop('itr_per_prt',   20)
        self.tst_num     = kwargs.pop('tst_num',     None)
        self.tst_shw     = kwargs.pop('tst_shw',     True)
        self.tst_sav     = kwargs.pop('tst_sav',     True)
        self.mdl_nam     = kwargs.pop('mdl_nam',    'model.ckpt'     )
        self.mdl_dir     = kwargs.pop('mdl_dir',    'Mybase/Model'   )
        self.log_dir     = kwargs.pop('log_dir',    'Mybase/logdata' )
        self.dat_dir     = kwargs.pop('dat_dir',    'Mybase/datasets')
        self.mov_ave_dca = kwargs.pop('mov_ave_dca', 0.99)
        self.dat_dir_tra = self.dat_dir + '/train'
        self.dat_dir_val = self.dat_dir + '/val'
        self.dat_dir_tst = self.dat_dir + '/test'
        self.dat_dir_rst = self.dat_dir + '/result'
        self.log_dir_tra = self.log_dir + '/train'
        self.log_dir_val = self.log_dir + '/val'
        self.log_dir_tst = self.log_dir + '/test'
        
        os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_lst
        
        if len(kwargs) > 0:
            extra = ', '.join('%s' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)
    """
    ###############################For CLASSIFY################################
    def _train_step(self, mtra=None, mtst=None, glb_stp=None):
        #将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上
        with tf.device('/cpu:0'):
            self.mdl.mod_tra = True
            
            GI_tra = GeneratorForImageNet(True,  self.dat_dir_tra, self.bat_siz, self.epc_num, \
                                          self.min_que_tra, self.gpu_lst, 32)
            GI_val = GeneratorForImageNet(False, self.dat_dir_val, self.bat_siz, self.epc_num, \
                                          self.min_que_val, self.gpu_lst,  1)
            imgs_lst_tra, lbls_lst_tra = GI_tra.get_input()
            imgs_lst_val, lbls_lst_val = GI_val.get_input()
            
            #with tf.name_scope('input_image'):
            #    tf.summary.image('input', X, 10)
            
            self.opm_cfg['decay_step'] =  self.opm_cfg['decay_step'] * self.tra_num
            tra_stp, lrn_rat = update_rule(self.opm_cfg, glb_stp)

            grds_lst = []
            loss_lst = []
            accs_lst = []
            for i in range(self.gpu_num):
                with tf.device(self.mdl_dev % i):
                    with tf.name_scope(self.MDL_DEV % i) as scp:
                        imgs_tra = GI_tra.preprocessing1(imgs_lst_tra[i])
                        imgs_val = GI_val.preprocessing1(imgs_lst_val[i])
                        lbls_tra = lbls_lst_tra[i]
                        lbls_val = lbls_lst_val[i]
                        imgs     = tf.cond(mtst, lambda: imgs_val, lambda: imgs_tra, strict=True)
                        lbls     = tf.cond(mtst, lambda: lbls_val, lambda: lbls_tra, strict=True)
                        loss, accs = \
                            self.mdl.forward(imgs, lbls, mtra=mtra, scp=scp)
                        #在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数
                        #注意tf.name_scope函数并不会影响tf.get_variable的命名空间
                        tf.get_variable_scope().reuse_variables()
                        #使用当前GPU计算所有变量的梯度
                        grds = tra_stp.compute_gradients(loss[0])
                        #print(grds)
                grds_lst.append(grds)
                loss_lst.append(loss)
                accs_lst.append(accs)
            '''
            with tf.variable_scope('average',  reuse = tf.AUTO_REUSE):
                mov_ave    = tf.train.ExponentialMovingAverage(self.mov_ave_dca, glb_stp)
                mov_ave_op = mov_ave.apply(tf.trainable_variables())
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mov_ave_op)
            '''
            with tf.variable_scope('optimize', reuse = tf.AUTO_REUSE):
                grds     = average_gradients(grds_lst)
                upd_opas = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(upd_opas):
                    tra_opa = tra_stp.apply_gradients(grds, global_step=glb_stp)

            loss = tf.stack(loss_lst, axis=0)
            accs = tf.stack(accs_lst, axis=0)
            #tf.summary.scalar('loss', loss)
            #tf.summary.scalar('acc', acc)
            #for grad, var in grads:
            #    if grad is not None:
            #        tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)
            #for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
        return tra_opa, lrn_rat, loss, accs
    """
    ###############################For Segmentation################################
    def _train_step(self, mtra=None, mtst=None, glb_stp=None):
        #将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上
        with tf.device('/cpu:0'):
            self.mdl.mod_tra = True
            
            GV_tra = GeneratorForVOC(True,  self.dat_dir_tra, self.bat_siz, self.epc_num, self.min_que_tra, self.gpu_lst, 2)
            GV_val = GeneratorForVOC(False, self.dat_dir_val, self.bat_siz, self.epc_num, self.min_que_val, self.gpu_lst, 1)
            imgs_lst_tra, gbxs_lst_tra, gmk_inss_lst_tra, gmk_sems_lst_tra, \
                gbx_nums_lst_tra, img_hgts_lst_tra, img_wdhs_lst_tra = GV_tra.get_input()
            imgs_lst_val, gbxs_lst_val, gmk_inss_lst_val, gmk_sems_lst_val, \
                gbx_nums_lst_val, img_hgts_lst_val, img_wdhs_lst_val = GV_val.get_input()
            
            #with tf.name_scope('input_image'):
            #    tf.summary.image('input', X, 10)
            self.opm_cfg['decay_step'] = self.opm_cfg['decay_step'] * self.tra_num
            tra_stp, lrn_rat = update_rule(self.opm_cfg, glb_stp)
            
            loss_lst  = []
            accs_lst  = []
            #msks_lst = []
            grds_lst  = []
            for i in range(self.gpu_num):
                with tf.device(self.mdl_dev % i):
                    with tf.name_scope(self.MDL_DEV % i) as scp:
                        imgs_tra, _, _, gmks_tra, _, _ = \
                            GV_tra.preprocessing1(imgs_lst_tra[i], gbxs_lst_tra[i], gmk_inss_lst_tra[i], gmk_sems_lst_tra[i], \
                                                  gbx_nums_lst_tra[i], img_hgts_lst_tra[i], img_wdhs_lst_tra[i])
                        imgs_val, _, _, gmks_val, _, _ = \
                            GV_val.preprocessing1(imgs_lst_val[i], gbxs_lst_val[i], gmk_inss_lst_val[i], gmk_sems_lst_val[i], \
                                                  gbx_nums_lst_val[i], img_hgts_lst_val[i], img_wdhs_lst_val[i])
                        imgs = tf.cond(mtst, lambda: imgs_val, lambda: imgs_tra, strict=True)
                        gmks = tf.cond(mtst, lambda: gmks_val, lambda: gmks_tra, strict=True)
                        loss, accs, msks = \
                            self.mdl.forward(imgs, gmks, mtra, scp)
                        #在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数
                        #注意tf.name_scope函数并不会影响tf.get_variable的命名空间
                        tf.get_variable_scope().reuse_variables()
                        #使用当前GPU计算所有变量的梯度
                        vars_lst = tf.trainable_variables()
                        #vars_lst= [v for v in vars_lst if 'generator/' in v.name]
                        grds     = tra_stp.compute_gradients(loss[0], var_list=vars_lst)
                        #print(grds)
                grds_lst.append(grds)
                loss_lst.append(loss)
                accs_lst.append(accs)
                #msks_lst.append(msks)
            '''
            with tf.variable_scope('average',  reuse = tf.AUTO_REUSE):
                mov_ave    = tf.train.ExponentialMovingAverage(self.mov_ave_dca, glb_stp)
                mov_ave_op = mov_ave.apply(tf.trainable_variables())
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mov_ave_op)
            '''
            upd_opas = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(upd_opas):
                with tf.variable_scope('optimize', reuse = tf.AUTO_REUSE):
                    grds    = average_gradients(grds_lst)
                    tra_opa = tra_stp.apply_gradients(grds, global_step=glb_stp)
            
            loss = tf.stack(loss_lst, axis=0)
            accs = tf.concat(accs_lst, axis=0)
            msks = None
            #msks= tf.concat(msks_lst, axis=0)
            #tf.summary.scalar('loss', loss)
            #tf.summary.scalar('acc', acc)
            #for grad, var in grads:
            #    if grad is not None:
            #        tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)
            #for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
        return tra_opa, lrn_rat, loss, accs, msks
    
    ###############################For Segmentation################################
    def _test_step(self):
        
        with tf.device("/cpu:0"):
            self.mdl.mod_tra = False
            
            mtra = tf.constant(False, dtype=tf.bool)
            GV   = GeneratorForVOC(False, self.dat_dir_tst, self.bat_siz, self.epc_num, self.min_que_tra, self.gpu_lst, None)
            imgs_lst, gbxs_lst, gmk_inss_lst, gmk_sems_lst, gbx_nums_lst, img_hgts_lst, img_wdhs_lst, img_nams_lst = GV.get_input2()
            
            msks_lst     = []
            img_wdws_lst = []
            for i in range(self.gpu_num):
                with tf.device(self.mdl_dev % i):
                    with tf.name_scope(self.MDL_DEV % i) as scp:
                        imgs, _, _, gmks, _, img_wdws = \
                            GV.preprocessing1(imgs_lst[i], gbxs_lst[i], gmk_inss_lst[i], gmk_sems_lst[i], \
                                              gbx_nums_lst[i], img_hgts_lst[i], img_wdhs_lst[i])
                        msks = self.mdl.forward(imgs, gmks, mtra, scp)
                        #在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数
                        #注意tf.name_scope函数并不会影响tf.get_variable的命名空间
                        tf.get_variable_scope().reuse_variables()
                msks_lst    .append(msks)
                img_wdws_lst.append(img_wdws)
            msks     = tf.concat(msks_lst,     axis=0) #(N, H, W)
            img_wdws = tf.concat(img_wdws_lst, axis=0)
            img_hgts = tf.concat(img_hgts_lst, axis=0)
            img_wdhs = tf.concat(img_wdhs_lst, axis=0)
            img_nams = tf.concat(img_nams_lst, axis=0)
        return msks, img_wdws, img_hgts, img_wdhs, img_nams
    
    def concat(self, sess=None, fetches=None, feed_dict=None, itr_num=None):
        rsts_lst = [[] for _ in range(len(fetches))]
        itr_cnt  = 0
        try:
            while True:
                rsts = sess.run(fetches, feed_dict=feed_dict)
                for i, rst in enumerate(rsts):
                    rsts_lst[i].append(rst)
                itr_cnt = itr_cnt + 1
                if itr_num != None and itr_cnt >= itr_num:
                    break
        except tf.errors.OutOfRangeError:
            print('Have reached the end of the dataset!')
        for i, rst in enumerate(rsts_lst):
            rsts_lst[i] = np.concatenate(rst, axis=0)
        return rsts_lst
    
    def merge(self, rsts=None, rst_nums=None):
        rst_imxs = []
        rsts_lst = [[] for _ in range(len(rsts))]
        for i, rst_num in enumerate(rst_nums): #batch
            rst_imxs.extend([i]*rst_num)
            for j, rst in enumerate(rsts):     #tensors
                rsts_lst[j].append(rst[i][:rst_num])
        rst_imxs = np.asarray(rst_imxs, dtype=np.int32)
        for i, rst in enumerate(rsts_lst):
            rsts_lst[i] = np.concatenate(rst, axis=0)
        return rsts_lst, rst_imxs
    """
    #####################################For CLASSIFY#####################################
    def train(self):
        
        tf.reset_default_graph()
        
        mtra     = tf.placeholder(dtype=tf.bool, name='train')
        mtst     = tf.placeholder(dtype=tf.bool, name='test' )
        glb_stp  = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        tra_opa, lrn_rat, loss, accs = self._train_step(mtra, mtst, glb_stp)

        #var     = tf.trainable_variables() #tf.global_variables()
        #var     = [v for v in var if 'layers_module1_0/' in v.name or 'layers_module1_1/' in v.name]
        #var     = [v for v in var if 'average/' not in v.name and 'optimize/' not in v.name]
        #var_ave = tf.train.ExponentialMovingAverage(self.mv_ave_decay, glb_stp)
        #var     = var_ave.variables_to_restore()
        #saver   = tf.train.Saver(var)
                
        #tf.summary.scalar('loss', loss)
        #summary_op   = tf.summary.merge_all()
        #summary_loss = tf.summary.merge(loss)
        #writer       = tf.summary.FileWriter(LOG_PATH, sess.graph, flush_secs=5) #tf.get_default_graph()    
        #gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={'CPU': 2}, \
        #                              inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)
        config        = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            #coord   = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            ckpt    = tf.train.get_checkpoint_state(self.mdl_dir)
            saver   = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            with open(os.path.join(self.log_dir_tra, 'accs'), 'ab') as fid_tra_accs, \
                 open(os.path.join(self.log_dir_tra, 'loss'), 'ab') as fid_tra_loss, \
                 open(os.path.join(self.log_dir_val, 'accs'), 'ab') as fid_val_accs, \
                 open(os.path.join(self.log_dir_val, 'loss'), 'ab') as fid_val_loss:
                
                try:
                    tra_cnt      = 0
                    epc_cnt      = 0
                    tra_loss_lst = []
                    while True:
                        #_, summary, loss1, = sess.run([train_op, summary_op, loss], feed_dict = {mtrain:True})
                        #writer.add_summary(summary, global_step=glb_stp.eval())
                        _, tra_loss = sess.run([tra_opa, loss], feed_dict={mtra:True, mtst:False})
                        tra_loss_lst.append(tra_loss)
                        tra_cnt = tra_cnt + 1
                        
                        if self.prt_ena and tra_cnt % self.itr_per_prt == 0:
                            tra_loss_lst = np.concatenate(tra_loss_lst, axis=0)
                            tra_loss     = np.mean(tra_loss_lst, axis=0)
                            tra_loss_lst = []
                            print('(Iteration %d) losses: %s' % (tra_cnt, str(tra_loss)))
                        
                        if tra_cnt % self.tra_num == 0:
                            epc_cnt   = epc_cnt + 1
                            saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)
                            
                            fetches   = [accs, loss]
                            feed_dict = {mtra:False, mtst:False}
                            tra_accs, tra_loss = self.concat(sess, fetches, feed_dict, self.val_num)
                            fetches   = [accs, loss]
                            feed_dict = {mtra:False, mtst:True }
                            val_accs, val_loss = self.concat(sess, fetches, feed_dict, self.val_num)
                            tra_accs  = np.mean(tra_accs, axis=0)
                            val_accs  = np.mean(val_accs, axis=0)
                            tra_loss  = np.mean(tra_loss, axis=0)
                            val_loss  = np.mean(val_loss, axis=0)

                            pickle.dump(tra_accs, fid_tra_accs, pickle.HIGHEST_PROTOCOL)
                            pickle.dump(val_accs, fid_val_accs, pickle.HIGHEST_PROTOCOL)
                            pickle.dump(tra_loss, fid_tra_loss, pickle.HIGHEST_PROTOCOL)
                            pickle.dump(val_loss, fid_val_loss, pickle.HIGHEST_PROTOCOL)
                            
                            if self.prt_ena:
                                print('(Epoch %d) lrn_rate: %f \n tra_accs: %s \n val_accs: %s \n tra_loss: %s \n val_loss: %s' \
                                      % (epc_cnt, lrn_rat.eval(), str(tra_accs), str(val_accs), str(tra_loss), str(val_loss)))      
                except tf.errors.OutOfRangeError:
                    print('Training is over!')
                #coord.request_stop()
                #coord.join(threads)
                    
    """
    #####################################For Segmentation##################################
    def train(self):
        
        tf.reset_default_graph()
        mtra    = tf.placeholder(dtype=tf.bool, name='train')
        mtst    = tf.placeholder(dtype=tf.bool, name='test' )
        glb_stp = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        tra_opa, lrn_rat, loss, accs, msks = self._train_step(mtra, mtst, glb_stp)
        
        #with tf.device(self.mdl_dev % 0):
            
        #tf.summary.scalar('loss', loss)
        #summary_op   = tf.summary.merge_all()
        #summary_loss = tf.summary.merge(loss)
        #writer       = tf.summary.FileWriter(LOG_PATH, sess.graph, flush_secs=5) #tf.get_default_graph()    
        #gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={'CPU': 2}, \
        #                              inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)
        config        = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            saver = tf.train.Saver()
            ckpt  = tf.train.get_checkpoint_state(self.mdl_dir)
            if ckpt and ckpt.model_checkpoint_path:
                var     = tf.global_variables()  #tf.global_variables()tf.trainable_variables()
                #var    = [v for v in var if 'average/' not in v.name and 'optimize/' not in v.name]
                #var    = [v for v in var if 'group_block1_0_0' in v.name or 'group_block1_0_1' in v.name \
                #          or 'group_block1_0_2' in v.name or 'group_block1_0_3' in v.name]
                #var_ave= tf.train.ExponentialMovingAverage(self.mv_ave_decay, glb_stp)
                #var    = var_ave.variables_to_restore()
                saver   = tf.train.Saver(var)
                '''
                mydict = load_weights()
                mykeys = mydict.keys()
                for i, v in enumerate(var):
                    if v.name in mykeys:
                        sess.run(tf.assign(v, mydict[v.name], validate_shape=True, use_locking=True))
                    else:
                        print(v.name)
                saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)
                return
                '''
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver = tf.train.Saver()
            
            with open(os.path.join(self.log_dir_tra, 'accs'), 'ab') as fid_tra_accs, \
                 open(os.path.join(self.log_dir_tra, 'loss'), 'ab') as fid_tra_loss, \
                 open(os.path.join(self.log_dir_val, 'accs'), 'ab') as fid_val_accs, \
                 open(os.path.join(self.log_dir_val, 'loss'), 'ab') as fid_val_loss:
                
                try:
                    tra_cnt      = 0
                    epc_cnt      = 0
                    tra_loss_lst = []
                    while True:
                        #_, summary, loss1, = sess.run([train_op, summary_op, loss], feed_dict = {mtrain:True})
                        #writer.add_summary(summary, global_step=glb_stp.eval())
                        _, tra_loss = sess.run([tra_opa, loss], feed_dict={mtra:True, mtst:False})
                        tra_loss_lst.append(tra_loss)
                        tra_cnt = tra_cnt + 1
                        
                        if self.prt_ena and tra_cnt % self.itr_per_prt == 0:
                            tra_loss_lst = np.concatenate(tra_loss_lst, axis=0)
                            tra_loss     = np.mean(tra_loss_lst, axis=0)
                            tra_loss_lst = []
                            print('(Iteration %d) losses: %s' % (tra_cnt, str(tra_loss)))

                        if tra_cnt % self.tra_num == 0:
                            epc_cnt   = epc_cnt + 1
                            saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)

                            fetches   = [accs, loss]
                            feed_dict = {mtra:False, mtst:False}
                            tra_accs, tra_loss = self.concat(sess, fetches, feed_dict, self.val_num)

                            fetches   = [accs, loss]
                            feed_dict = {mtra:False, mtst:True }
                            val_accs, val_loss = self.concat(sess, fetches, feed_dict, self.val_num)
                            tra_accs  = self.mdl.accs_seg_py(tra_accs)
                            val_accs  = self.mdl.accs_seg_py(val_accs)
                            tra_loss  = np.mean(tra_loss, axis=0)
                            val_loss  = np.mean(val_loss, axis=0)

                            pickle.dump(tra_accs, fid_tra_accs, pickle.HIGHEST_PROTOCOL)
                            pickle.dump(val_accs, fid_val_accs, pickle.HIGHEST_PROTOCOL)
                            pickle.dump(tra_loss, fid_tra_loss, pickle.HIGHEST_PROTOCOL)
                            pickle.dump(val_loss, fid_val_loss, pickle.HIGHEST_PROTOCOL)

                            if self.prt_ena:
                                print('(Epoch %d) lrn_rate: %f\n tra_accs: %s\n val_accs: %s\n tra_loss: %s\n val_loss: %s\n ' \
                                      % (epc_cnt, lrn_rat.eval(), \
                                         str(tra_accs), str(val_accs), str(tra_loss), str(val_loss)))
                except tf.errors.OutOfRangeError:
                    print('Training is over!')
    
    ###############################For Segmentation################################
    def test(self):
        
        GV = GeneratorForVOC()
        
        tf.reset_default_graph()
        glb_stp = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
        msks, img_wdws, img_hgts, img_wdhs, img_nams = self._test_step()

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        #config      = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        #config      = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={"CPU": 2}, \
        #                             inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)
        config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            saver = tf.train.Saver()
            ckpt  = tf.train.get_checkpoint_state(self.mdl_dir)
            if ckpt and ckpt.model_checkpoint_path:
                #var     = tf.trainable_variables() #tf.global_variables()
                #var_ave = tf.train.ExponentialMovingAverage(self.mv_ave_decay, glb_stp)
                #var     = var_ave.variables_to_restore()
                #saver   = tf.train.Saver(var)
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver    = tf.train.Saver()
            else:
                print("No checkpoint file found!")
                return
            
            with open(os.path.join(self.log_dir_tst, "imgs"), 'wb') as fid_tst_imgs, \
                 open(os.path.join(self.log_dir_tst, "msks"), 'wb') as fid_tst_msks:
                
                fetches    = [msks, img_wdws, img_hgts, img_wdhs, img_nams]
                feed_dict  = {}
                [msks_kep, img_wdws_kep, img_hgts_kep, img_wdhs_kep, img_nams_kep] = self.concat(sess, fetches, feed_dict)
                
                for i, img_nam_tmp in enumerate(img_nams_kep):
                    msks_tmp    = msks_kep[i]
                    img_wdw_tmp = img_wdws_kep[i]
                    img_hgt_tmp = img_hgts_kep[i]
                    img_wdh_tmp = img_wdhs_kep[i]
                    img_nam_tmp = bytes.decode(img_nam_tmp)
                    _, _, _, msks_tmp = \
                        GV.recover_instances(None, None, None, msks_tmp, img_wdw_tmp, img_hgt_tmp, img_wdh_tmp)
                    pickle.dump(img_nam_tmp, fid_tst_imgs, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(msks_tmp,    fid_tst_msks, pickle.HIGHEST_PROTOCOL)
    
    ###############################For Segmentation################################
    def display_detections(self):
        
        GV = GeneratorForVOC(dat_dir=self.dat_dir_tst, tst_shw=self.tst_shw, tst_sav=self.tst_sav)
        with open(os.path.join(self.log_dir_tst, "imgs"), 'rb') as fid_tst_imgs, \
             open(os.path.join(self.log_dir_tst, "msks"), 'rb') as fid_tst_msks:
            
            while True:
                try:  
                    img_nam = pickle.load(fid_tst_imgs)
                    msks    = pickle.load(fid_tst_msks)
                    #print(msks.shape)
                    img_fil = os.path.join(self.dat_dir_tst, img_nam)
                    img     = cv2.imread(img_fil)
                    #print(img.shape)
                    if type(img) != np.ndarray:
                        print("Failed to find image %s" %(img_fil))
                        continue
                    img_hgt, img_wdh = img.shape[0], img.shape[1]
                    if img.size == img_hgt * img_wdh:
                        print ('Gray Image %s' %(img_fil))
                        img_zro = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                        img_zro[:, :, :] = img[:, :, np.newaxis]
                        img = img_zro
                    assert img.size == img_wdh * img_hgt * 3, '%s' % img_nam
                    img = img[:, :, ::-1]
                    GV.display_instances(img, None, None, None, None, msks, img_nam)
                except EOFError:
                    return
                
    def show_loss_acc(self):

        with open(os.path.join(LOG_PATH1, 'loss'), 'rb') as fid_train_loss, \
             open(os.path.join(LOG_PATH1, 'mAP'), 'rb') as fid_train_mAP, \
             open(os.path.join(LOG_PATH2, 'mAP'), 'rb') as fid_val_mAP:
                    
            loss_history      = get_all_data(fid_train_loss)
            train_acc_history = get_all_data(fid_train_mAP)
            val_acc_history   = get_all_data(fid_val_mAP)

            plt.figure(1)

            plt.subplot(2, 1, 1)
            plt.title('Training loss')
            plt.xlabel('Iteration')

            plt.subplot(2, 1, 2)
            plt.title('accuracy')
            plt.xlabel('Epoch')
            
            #plt.subplot(3, 1, 3)
            #plt.title('Validation accuracy')
            #plt.xlabel('Epoch')
            
            plt.subplot(2, 1, 1)
            plt.plot(loss_history, 'o')

            plt.subplot(2, 1, 2)
            plt.plot(train_acc_history, '-o', label='train_acc')
            plt.plot(val_acc_history, '-o', label='val_acc')

            for i in [1, 2]:
                plt.subplot(2, 1, i)
                plt.legend(loc='upper center', ncol=4)

                plt.gcf().set_size_inches(15, 15)
            
            plt.show()
