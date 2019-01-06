    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.cls_num)
        hist = np.bincount(self.cls_num*label_true[mask].astype(int)+label_pred[mask], minlength=self.cls_num**2).reshape(self.cls_num, self.cls_num)
        return hist
    
    def accs_seg_img_py(self, label_pred, label_true):
        
        hist    = self._fast_hist(label_pred.flatten(), label_true.flatten())
        acc     = np.diag(hist).sum() /  hist.sum()
        acc_cls = np.diag(hist)       /  hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu      = np.diag(hist)       / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq    = hist.sum(axis=1)    / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        accs    = np.stack([mean_iu, acc, acc_cls, fwavacc], axis=0)
        accs    = accs.astype(dtype=np.float32, copy=False)
        return accs
    
    def accs_seg_img(self, label_pred, label_true):
        accs    = tf.py_func(self.accs_seg_img_py, [label_pred, label_true], tf.float32)
        return accs





















msks_pst_= tf.nn.softmax(msks_pst, axis=-1)               #(N, H, W, C)
        msks_pre = gmks
        msks     = msks_pst_
        
        if self.mod_tra: 
            los_dat   = self.loss_seg(msks_pst, msks_pre)
            los_reg   = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            los       = los_dat + los_reg
            loss      = tf.stack([los, los_dat, los_reg], axis=0)
            
            msks_pst0 = tf.argmax(msks_pst_, axis=-1, output_type=tf.int32)      #(N, H, W) class
            msks_pst1 = tf.reduce_max(msks_pst_, axis=-1)                        #(N, H, W) probs
            msks_pst1 = msks_pst1 >= self.msk_min                                #(N, H, W)
            msks_pst1 = tf.cast(msks_pst1, dtype=tf.int32)                       #(N, H, W)
            msks_pst  = msks_pst0 * msks_pst1                                    #(N, H, W)
            accs      = self.accs_seg(msks_pst, msks_pre)
            return loss, accs, msks
        else:
            return msks



def fold1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in) #[N, H, W, C]
    
    with tf.variable_scope('fold1_'+str(layer)) as scope:
        tensor_in  = tf.reshape(tensor_in, [x_shape[0], x_shape[1]//2, 2, x_shape[2]//2, 2, x_shape[3]])
        tensor_in  = tf.transpose(tensor_in, [0, 1, 3, 2, 4, 5])
        tensor_out = tf.reshape(tensor_in, [x_shape[0], x_shape[1]//2, x_shape[2]//2, x_shape[3]*4])
        print_activations(tensor_out)
    return tensor_out

def unfold1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in) #[N, H, W, C]
    
    with tf.variable_scope('unfold1_'+str(layer)) as scope:
        tensor_in  = tf.reshape(tensor_in, [x_shape[0], x_shape[1], x_shape[2], 2, 2, x_shape[3]//4])
        tensor_in  = tf.transpose(tensor_in, [0, 1, 3, 2, 4, 5])
        tensor_out = tf.reshape(tensor_in, [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//4])
        print_activations(tensor_out)
    return tensor_out




            coord   = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            coord.request_stop()
            coord.join(threads)


        imgs_lst = []
        lbls_lst = []
        if self.fil_num >= self.gpu_num:
            fil_pat  = os.path.join(self.dat_dir, 'imagenet', '*.tfrecord')
            dataset  = tf.data.Dataset.list_files(file_pattern=fil_pat, shuffle=True, seed=None)
            #dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
            #                             cycle_length=self.num_readers, block_length=1, num_parallel_calls=1)
            dataset  = dataset.apply(tf.data.experimental.\
                                     parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                                                         cycle_length=self.num_readers, block_length=1, sloppy=True))
        else:
            fil_nam  = glob.glob(os.path.join(self.dat_dir, 'imagenet', '*.tfrecord'))
            dataset  = tf.data.TFRecordDataset(fil_nam, compression_type='ZLIB', num_parallel_reads=self.num_readers)
        
        for i in range(self.gpu_num):
            dat_sha  = dataset.shard(num_shards=self.gpu_num, index=i)
            if self.fil_num >= self.gpu_num:

                #dat_sha= dat_sha.shuffle(buffer_size=self.num_readers, seed=None, reshuffle_each_iteration=True)

            dat_sha  = dat_sha.prefetch(buffer_size=self.bat_siz)
            dat_sha  = dat_sha.map(parse_function, num_parallel_calls=self.num_threads)
            dat_sha  = dat_sha.apply(tf.data.experimental.\
                                     shuffle_and_repeat(buffer_size=self.capacity, count=self.epc_num, seed=None))
            dat_sha  = dat_sha.batch(batch_size=self.bat_siz, drop_remainder=True)
            #dat_sha = dat_sha.apply(tf.data.experimental.\
            #                        map_and_batch(parse_function, batch_size=self.bat_siz, num_parallel_batches=None, \
            #                                      drop_remainder=True, num_parallel_calls=self.num_threads))
            #dat_sha = dat_sha.cache(filename=os.path.join(self.dat_dir, 'cache'))
            dat_sha  = dat_sha.prefetch(buffer_size=1)
            #dat_sha = dat_sha.apply(tf.data.experimental.prefetch_to_device(self.mdl_dev%i, buffer_size=1))
            iterator = dat_sha.make_one_shot_iterator()
            example  = iterator.get_next()
            imgs_lst.append(example['image/image'])
            lbls_lst.append(example['label/label'])
        return imgs_lst, lbls_lst   

            dataset  = dataset.apply(tf.data.experimental.\
                                 parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'), \
                                                     cycle_length=self.num_readers, block_length=1, sloppy=True, \
                                                     buffer_output_elements=self.bat_siz_all//self.num_readers, \
                                                     prefetch_input_elements=None))







#random_uniform

    '''
    def get_input(self):
        #创建文件列表，并通过文件列表创建输入文件队列。
        #在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中
        #文件列表应该包含所有提供训练数据的TFRecord文件
        filename = os.path.join(self.dat_dir, 'cifar', '*.tfrecord')
        files    = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(files, shuffle=True, capacity=1000)

        #解析TFRecord文件里的数据
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)

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
    
        capacity = self.min_after_dequeue + 3 * self.bat_siz
        #tf.train.shuffle_batch_join
        imgs, lbls = tf.train.shuffle_batch(
            tensors=[img, lbl], batch_size=self.bat_siz, \
            num_threads=self.num_threads, capacity=capacity, min_after_dequeue=self.min_after_dequeue)
        return imgs, lbls
    '''

def get_input(self):
        #创建文件列表，并通过文件列表创建输入文件队列。
        #在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中
        #文件列表应该包含所有提供训练数据的TFRecord文件
        filename = os.path.join(self.dat_dir, "*.tfrecord")
        files    = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(files, shuffle=True, capacity=1000)

        #解析TFRecord文件里的数据
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/img_id':        tf.FixedLenFeature([], tf.int64 ),
                'image/image':         tf.FixedLenFeature([], tf.string),
                'image/height':        tf.FixedLenFeature([], tf.int64 ),
                'image/width':         tf.FixedLenFeature([], tf.int64 ),

                'label/num_instances': tf.FixedLenFeature([], tf.int64 ),
                'label/gt_masks':      tf.FixedLenFeature([], tf.string),
                'label/gt_boxes':      tf.FixedLenFeature([], tf.string),
            }
        )
        
        img_idx = tf.cast(features['image/img_id'],        tf.int32)
        img_hgt = tf.cast(features['image/height'],        tf.int32)
        img_wdh = tf.cast(features['image/width'],         tf.int32)
        gbx_num = tf.cast(features['label/num_instances'], tf.int32)

        img  = tf.decode_raw(features['image/image'],    tf.uint8  )
        gbxs = tf.decode_raw(features['label/gt_boxes'], tf.float32)
        gmks = tf.decode_raw(features['label/gt_masks'], tf.uint8  )
        img  = tf.reshape(img,  [img_hgt, img_wdh, 3])
        gbxs = tf.reshape(gbxs, [gbx_num, 5])
        gmks = tf.reshape(gmks, [gbx_num, img_hgt, img_wdh])
        
        img, gbxs, gmks, img_wdw, img_hgt_, img_wdh_ = self.preprocessing(img, gbxs, gmks)
        
        gbx_num  = tf.shape(gbxs)[0]
        paddings = [[0, self.max_num-gbx_num], [0, 0]]
        gbxs     = tf.pad(gbxs, paddings, "CONSTANT")
        paddings = [[0, self.max_num-gbx_num], [0, 0], [0, 0]]
        gmks     = tf.pad(gmks, paddings, "CONSTANT")
        img      = tf.reshape(img,  [self.img_siz_max, self.img_siz_max, 3])
        gbxs     = tf.reshape(gbxs, [self.max_num, 5])
        gmks     = tf.reshape(gmks, [self.max_num]+self.box_msk_siz)
        capacity = self.min_after_dequeue + 3 * self.bat_siz
        #tf.train.shuffle_batch_join
        imgs, gbxs, gmks, gbx_nums, img_wdws, img_hgts_, img_wdhs_ = tf.train.shuffle_batch(
            tensors=[img, gbxs, gmks, gbx_num, img_wdw, img_hgt_, img_wdh_], batch_size=self.bat_siz, \
            num_threads=self.num_threads, capacity=capacity, min_after_dequeue=self.min_after_dequeue)
        return imgs, gbxs, gmks, gbx_nums, img_wdws, img_hgts_, img_wdhs_


def fold1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    stride  = params['fold']['stride']   #[[2, 2], [2, 2]] 
    use_crs = params['fold']['use_crs']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    if isinstance(stride[0],   int):
        stride    = [stride]
    #stride = stride[::-1]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    
    with tf.variable_scope('fold1_'+str(layer)) as scope:
        
        num_srds    = len(stride)
        hgt_srds    = [srd[0] for srd in stride]
        wdh_srds    = [srd[1] for srd in stride]
        hws_srds    = reduce(lambda x,y: x+y, stride  )
        hgt_srd_all = reduce(lambda x,y: x*y, hgt_srds)
        wdh_srd_all = reduce(lambda x,y: x*y, wdh_srds)
        hws_srd_all = hgt_srd_all * wdh_srd_all
        
        hgt_dims    = [           2 + i for i in range(num_srds)]
        wdh_dims    = [num_srds + 3 + i for i in range(num_srds)]
        hws_dims    = [[hgt_dims[i], wdh_dims[i]] for i in range(num_srds)]
        hws_dims    = reduce(lambda x,y: x+y, hws_dims)
        
        new_num     = x_shape[3]  * hws_srd_all
        new_hgt     = x_shape[1] // hgt_srd_all
        new_wdh     = x_shape[2] // wdh_srd_all
        old_hgt     = new_hgt     * hgt_srd_all
        old_wdh     = new_wdh     * wdh_srd_all
        
        if old_hgt != x_shape[1] or old_wdh != x_shape[2]:
            tensor_in = tensor_in[:, :old_hgt, :old_wdh, :]
            #x_shape  = get_shape(tensor_in)
        tensor_in   = tf.reshape(tensor_in, [x_shape[0], new_hgt] + hgt_srds + [new_wdh] + wdh_srds + [x_shape[3]])
        tensor_in   = tf.transpose(tensor_in, [0, 1, 2+num_srds] + hws_dims + [3+2*num_srds])
        
        if use_crs:
            for srd in stride:
                assert srd[0] == srd[1] == 2, 'Invalid stride for cross position!'
            indices = np.arange(hws_srd_all)
            indices = np.reshape(indices, [4 for _ in range(len(stride))])
            for i in range(len(stride)):
                indices = np.take(indices, [0,3,1,2], axis=i)
            indices   = np.reshape(indices, [-1])
            tensor_in = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, hws_srd_all, x_shape[3]])
            tensor_in = tf.gather(tensor_in, indices, axis=3)
        
        tensor_out = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, new_num])
        print_activations(tensor_out)
    return tensor_out


def unfold1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    stride  = params['unfold']['stride']
    use_crs = params['unfold']['use_crs']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    if isinstance(stride[0],   int):
        stride    = [stride]
    #stride = stride[::-1]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    
    with tf.variable_scope('unfold1_'+str(layer)) as scope:
        
        num_srds    = len(stride)
        hgt_srds    = [srd[0] for srd in stride]
        wdh_srds    = [srd[1] for srd in stride]
        hws_srds    = reduce(lambda x,y: x+y, stride  )
        hgt_srd_all = reduce(lambda x,y: x*y, hgt_srds)
        wdh_srd_all = reduce(lambda x,y: x*y, wdh_srds)
        hws_srd_all = hgt_srd_all * wdh_srd_all
        
        hgt_dims    = [3 + 2 * i for i in range(num_srds)]
        wdh_dims    = [4 + 2 * i for i in range(num_srds)]
        
        new_num     = x_shape[3] // hws_srd_all
        new_hgt     = x_shape[1]  * hgt_srd_all
        new_wdh     = x_shape[2]  * wdh_srd_all
        old_num     = new_num     * hws_srd_all
        
        if old_num != x_shape[3]:
            tensor_in = tensor_in[:, :, :, :old_num]
            #x_shape  = get_shape(tensor_in)
        
        if use_crs:
            for srd in stride:
                assert srd[0] == srd[1] == 2, 'Invalid stride for cross position!'
            indices = np.arange(hws_srd_all)
            indices = np.reshape(indices, [4 for _ in range(len(stride))])
            for i in range(len(stride)):
                indices = np.take(indices, [0,2,3,1], axis=i)
            indices   = np.reshape(indices, [-1])
            tensor_in = tf.reshape(tensor_in, x_shape[0:3] + [hws_srd_all] + [new_num])
            tensor_in = tf.gather(tensor_in, indices, axis=3)
        
        tensor_in   = tf.reshape(tensor_in, x_shape[0:3] + hws_srds + [new_num])
        tensor_in   = tf.transpose(tensor_in, [0,1] + hgt_dims + [2] + wdh_dims + [3+2*num_srds])
        tensor_out  = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, new_num])
        print_activations(tensor_out)
    return tensor_out


def attn1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['attn']['number']
    shape     = params['attn']['shape']
    rate      = params['attn']['rate']
    stride    = params['attn']['stride']
    padding   = params['attn']['padding']
    use_bias  = params['attn']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)                                      #[N, H, W, M, C]
    m_shape = [shape[i]+(shape[i]-1)*(rate[i]-1) for i in range(2)]     #[h, w]
    
    shape   = shape + x_shape[3:] + number                              #[h, w, M, C, M', C']
    shape_q = [shape[0]*shape[1]*shape[2]*shape[3], shape[4]*shape[5]]  #[h*w*M*C, M'*C']
    shape_k = shape[0:4] + [shape[5]]                                   #[h, w, M, C, C']
    with tf.variable_scope('attn1_'+str(layer)) as scope:
        
        weights = tf.get_variable(name='weights', shape=shape, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)                  #(h, w, M, C, M', C')
        weight_q = tf.reshape(weights, shape_q)                         #(h*w*M*C, M'*C')
        weight_k = tf.reduce_sum(weights, axis=4)                       #(h, w, M, C, C')
        
        if use_bias:
            biases = tf.get_variable(name='biases', shape=number, dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable) #(M', C')
            
        if padding == 'SAME':
            new_hgt     = int(np.ceil(x_shape[1] / stride[0]))
            new_wdh     = int(np.ceil(x_shape[2] / stride[1]))
            pad_hgt_all = (new_hgt - 1) * stride[0] + m_shape[0] - x_shape[1]
            pad_wdh_all = (new_wdh - 1) * stride[1] + m_shape[1] - x_shape[2]
            pad_top     = pad_hgt_all // 2
            pad_btm     = pad_hgt_all - pad_top
            pad_lft     = pad_wdh_all // 2
            pad_rgt     = pad_wdh_all - pad_lft
            paddings    = [[0, 0], [pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0], [0, 0]]
            tensor_in   = tf.pad(tensor_in, paddings, mode='CONSTANT', constant_values=0)
            x_shape     = get_shape(tensor_in)     #[N, H, W, M, C]
        elif padding == 'VALID':
            new_hgt     = int(np.ceil((x_shape[1] - m_shape[0] + 1) / stride[0]))
            new_wdh     = int(np.ceil((x_shape[2] - m_shape[1] + 1) / stride[1]))
        else:
            raise ValueError('Invalid padding method!')

        y_shape    = [x_shape[0], new_hgt, new_wdh] + number
        tensor_out = tf.TensorArray(dtype=tf.float32, size=y_shape[1]*y_shape[2], dynamic_size=False, clear_after_read=True, \
                                    tensor_array_name=None, handle=None, flow=None, infer_shape=True, \
                                    element_shape=[y_shape[0]]+number, colocate_with_first_write_call=True) #(H*W, N, M', C')

        def cond(i, tensor_out):
            c = tf.less(i, y_shape[1]*y_shape[2])
            return c

        def body(i, tensor_out):
            ymn  = i  // y_shape[2] * stride[0]
            xmn  = i  %  y_shape[2] * stride[1]
            ymx  = ymn + m_shape[0]
            xmx  = xmn + m_shape[1]
            fetx = tensor_in[:, ymn:ymx:rate[0], xmn:xmx:rate[1], :, :]       #(N, h, w, M, C)
            fett = tf.reshape(fetx, [y_shape[0], -1])                         #(N, h*w*M*C)
            fetq = tf.matmul(fett, weight_q)                                  #(N, M'*C') (N, h*w*M*C) (h*w*M*C, M'*C')
            fetq = tf.reshape(fetq, [y_shape[0]]+number)                      #(N, M', C')
            fett = tf.transpose(fetx, [1, 2, 3, 0, 4])                        #(h, w, M, N, C)
            fetk = tf.matmul(fett, weight_k)                                  #(h, w, M, N, C') (h, w, M, N, C) (h, w, M, C, C')
            fetk = tf.transpose(fetk, [3, 0, 1, 2, 4])                        #(N, h, w, M, C')
            fetk = tf.reshape(fetk, [y_shape[0], -1, number[1]])              #(N, h*w*M, C')
            atts = tf.matmul(fetq, fetk, transpose_b=True)                    #(N, M', h*w*M)
            atts = atts / np.sqrt(number[1])                                  #(N, M', h*w*M)
            atts = tf.nn.softmax(atts, axis=-1)                               #(N, M', h*w*M)
            fetk = tf.matmul(atts, fetk)                                      #(N, M', C') (N, M', h*w*M) (N, h*w*M, C')
            fetq = fetq + fetk                                                #(N, M', C')
            fetq = fetq + biases if use_bias else fetq                        #(N, M', C')
            tensor_out = tensor_out.write(i, fetq)                            #(H'*W', N, M', C')
            return [i+1, tensor_out]
        
        i = tf.constant(0)
        [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                        parallel_iterations=y_shape[1]*y_shape[2], back_prop=True, swap_memory=False)
        tensor_out = tensor_out.stack()                                       #(H'*W', N, M', C')
        tensor_out = tf.transpose(tensor_out, [1, 0, 2, 3])                   #(N, H'*W', M', C')
        tensor_out = tf.reshape(tensor_out, y_shape)                          #(N, H', W', M', C')
        print_activations(tensor_out)
    return tensor_out


def proj1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    number    = params['proj']['number']   #[b, r, c']
    shape     = params['proj']['shape']
    rate      = params['proj']['rate']
    stride    = params['proj']['stride']
    padding   = params['proj']['padding']
    use_bias  = params['proj']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope('proj1_'+str(layer)) as scope:
        
        x_shape        = get_shape(tensor_in)                                                    #[N, H, W, C]
        tensor_in      = tf.reshape(tensor_in, x_shape[:3]+number[0:2]+\
                                              [x_shape[3]//number[0]//number[1]])                #(N, H, W, b, r, c)
        tensor_in      = tf.transpose(tensor_in, [0, 3, 4, 1, 2, 5])                             #(N, b, r, H, W, c)
        x_shape        = get_shape(tensor_in)                                                    #[N, b, r, H, W, c]
        y_shape        = x_shape[:5] + [number[2]]                                               #[N, b, r, H, W, c']
        
        tensor_in      = tf.reshape(tensor_in, [x_shape[0]*x_shape[1]*x_shape[2]]+x_shape[3:6])  #(N*b*r, H, W, c)
        params['conv'] = {'number':number[2], 'shape':shape, 'rate':rate, \
                          'stride':stride, 'padding':padding, 'use_bias':use_bias}
        tensor_out     = conv1(tensor_in, 0, params, mtrain)                                     #(N*b*r, H, W, c')
        
        tensor_out     = tf.reshape(tensor_out, y_shape)                                         #[N, b, r, H, W, c']
        tensor_out     = tf.transpose(tensor_out, [0, 3, 4, 1, 2, 5])                            #(N, H, W, b, r, c')
        y_shape        = get_shape(tensor_out)                                                   #[N, H, W, b, r, c']
        tensor_out     = tf.reshape(tensor_out, y_shape[0:3]+[y_shape[3]*y_shape[4]*y_shape[5]]) #(N, H, W, b*r*c')
        #tf.summary.histogram('proj', tensor_out)
        print_activations(tensor_out)
    return tensor_out


def proj_bn1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    params['proj']['use_bias'] = False
    
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('proj_bn1_'+str(layer)) as scope:
        proj       = proj1(tensor_in, 0, params, mtrain)
        tensor_out = batchnorm1(proj, 0, params, mtrain)
    return tensor_out


def proj_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('proj_relu1_'+str(layer)) as scope:
        proj       = proj1(tensor_in, 0, params, mtrain) 
        tensor_out = relu1(proj, 0, params, mtrain) 
    return tensor_out


def proj_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('proj_bn_relu1_'+str(layer)) as scope:
        bn         = proj_bn1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain)
    return tensor_out


def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    use_fold = params['group_unit']['use_fold']
    number   = params['group_unit']['number'] #[[b, r, c], [b, r, c], [b, r, c]]
    shape    = params['group_unit']['shape']
    rate     = params['group_unit']['rate']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:
        
        if use_fold:
            params['fold'] = {'stride':[[2,2]], 'use_crs':False}
            tensor_in      = fold1(tensor_in, 0, params, mtrain)
        
        params['proj'] = {'number':number[0], 'shape':[1,1], 'rate':[1,1], 'stride':[1,1], \
                          'padding':'VALID', 'use_bias':False}
        residual       = proj_bn_relu1(tensor_in, 0, params, mtrain)
        
        params['proj'] = {'number':number[1], 'shape':shape, 'rate':rate,  'stride':[1,1], \
                          'padding':'SAME',  'use_bias':False}
        residual       = proj_bn_relu1(residual,  1, params, mtrain)
        
        params['proj'] = {'number':number[2], 'shape':[1,1], 'rate':[1,1], 'stride':[1,1], \
                          'padding':'VALID', 'use_bias':False}
        residual       = proj_bn1(residual,  0, params, mtrain)
        
        tensor_out     = tensor_in + residual
        tensor_out     = relu1(tensor_out, 0, params, mtrain)
    return tensor_out


def group_block1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    block_setting = params['group_block']['block_setting']

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    tensor_out    = tensor_in
    out_list      = []
    for i, block in enumerate(block_setting):
        
        number, shape, rate, unit_number, unit_trainable = block
        params['com']['trainable'] = unit_trainable
        
        with tf.variable_scope('group_block1_'+str(layer)+'_'+str(i)) as scope:
            
            for j in range(unit_number):
                if j == 0: #the first unit in the block
                    params['group_unit'] = {'use_fold':True ,'number':number, 'shape':shape, 'rate':rate}
                else:      #identity mapping
                    params['group_unit'] = {'use_fold':False,'number':number, 'shape':shape, 'rate':rate}
                tensor_out = group_unit1(tensor_out, j, params, mtrain)
        out_list.append(tensor_out)
    return out_list















#the group block setting
        #depth_bottle, depth_output, shape, rate, unit_number, unit_trainable
        #不管输入的特征有多么弱，我们认为它也应该产生一个对输出的完整描述
        #尽管低级特征对低级属性的描述更强，但是它也应该有对更高级属性的完整描述能力
        #况且在层数增加的过程中，特征的低级属性会被弱化，而高级属性不断被增强
        
        #256    #1    * 256 -->    1 * 64 | #64
        #1024   #4    * 256 -->    4 * 64 | #256
        #4096   #16   * 256 -->   16 * 64 | #1024  #1  * 1024 -->  1 * 256 | #256
        #16384  #64   * 256 -->   64 * 64 | #4096  #4  * 1024 -->  4 * 256 | #1024
        #65536  #256  * 256 -->  256 * 64 | #16384 #16 * 1024 --> 16 * 256 | #4096  #1 * 4096 --> 1 * 1024 | #1024
        #262144 #1024 * 256 --> 1024 * 64 | #65536 #64 * 1024 --> 64 * 256 | #16384 #4 * 4096 --> 4 * 1024 | #4096
        
        self.grp_set  = [([[[1, 64]],         [[1, 256]],        ], [2,2], [1,1], 3, True ), 
                         ([[[4, 64]],         [[4, 256]],        ], [2,2], [1,1], 4, True ), 
                         ([[[16,64],[1,256]], [[1,1024],[16,256]]], [2,2], [1,1], 6, True ), 
                         ([[[64,64],[4,256]], [[4,1024],[64,256]]], [2,2], [1,1], 3, True )]



def proj1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    number    = params['proj']['number']   #[b, c']
    shape     = params['proj']['shape']
    rate      = params['proj']['rate']
    stride    = params['proj']['stride']
    padding   = params['proj']['padding']
    use_bias  = params['proj']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope('proj1_'+str(layer)) as scope:
        
        x_shape        = get_shape(tensor_in)                                                   #[N, H, W, C]
        tensor_in      = tf.reshape(tensor_in, x_shape[:3]+[number[0],x_shape[3]//number[0]])   #(N, H, W, b, c)
        tensor_in      = tf.transpose(tensor_in, [0, 3, 1, 2, 4])                               #(N, b, H, W, c)
        x_shape        = get_shape(tensor_in)                                                   #[N, b, H, W, c]
        y_shape        = x_shape[:4] + [number[1]]                                              #[N, b, H, W, c']
        
        tensor_in      = tf.reshape(tensor_in, [x_shape[0]*x_shape[1]]+x_shape[2:4]+\
                                               [x_shape[4]])                                    #(N*b, H, W, c)
        params['conv'] = {'number':number[1], 'shape':shape, 'rate':rate, \
                          'stride':stride, 'padding':padding, 'use_bias':use_bias}
        tensor_out     = conv1(tensor_in, 0, params, mtrain)                                    #(N*b, H, W, c')
        
        tensor_out     = tf.reshape(tensor_out, y_shape)                                        #[N, b, H, W, c']
        tensor_out     = tf.transpose(tensor_out, [0, 2, 3, 1, 4])                              #(N, H, W, b, c')
        y_shape        = get_shape(tensor_out)                                                  #[N, H, W, b, c']
        tensor_out     = tf.reshape(tensor_out, y_shape[0:3]+[y_shape[3]*y_shape[4]])           #(N, H, W, b*c')
        #tf.summary.histogram('proj', tensor_out)
        print_activations(tensor_out)
    return tensor_out



def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    use_fold = params['group_unit']['use_fold']
    number   = params['group_unit']['number'] #[[[b, c'], [b, c'], [b, c']], [[b, c'], [b, c'], [b, c']], c"]
    shape    = params['group_unit']['shape']
    rate     = params['group_unit']['rate']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:
        
        if use_fold:
            params['fold'] = {'stride':[[2,2]], 'use_crs':False}
            tensor_in      = fold1(tensor_in, 0, params, mtrain)
        
        x_shape  = get_shape(tensor_in)               #[N, H, W, C]
        shrink   = [tensor_in]
        #维度收缩
        #不管输入的特征有多么弱，我们认为它也应该产生一个对输出的完整描述
        #尽管低级特征对低级属性的描述更强，但是它也应该有对更高级属性的完整描述能力
        #况且在层数增加的过程中，特征的低级属性会被弱化，而高级属性不断被增强
        residual = tensor_in
        for i, depth in enumerate(number[0]):      #[b, c']
            params['proj'] = {'number':depth, 'shape':[1,1], 'rate':[1,1], 'stride':[1,1], \
                              'padding':'VALID', 'use_bias':False}
            residual       = proj_bn_relu1(residual, 0+i, params, mtrain)
            shrink.append(residual)
        shrink = shrink[:-1]
        shrink = shrink[::-1]
        #全局关联
        r_shape        = get_shape(residual)
        params['conv'] = {'number':r_shape[3], 'shape':shape, 'rate':rate, 'stride':[1,1], 'padding':'SAME'}
        residual       = conv_bn_relu1(residual, 0, params, mtrain)
        #维度伸展
        for i, depth in enumerate(number[1]):      #[b, c']
            params['proj'] = {'number':depth, 'shape':[1,1], 'rate':[1,1], 'stride':[1,1], \
                              'padding':'VALID', 'use_bias':False}
            residual       = proj_bn1(residual, 0+i, params, mtrain)
            residual       = residual + shrink[i]
            residual       = relu1(residual, 0+i, params, mtrain)
        tensor_out = residual
    return tensor_out




#64   | #256    #1    * 256 -->    1 * 64 | #64
        #128  | #1024   #4    * 256 -->    4 * 64 | #256
        #256  | #4096   #16   * 256 -->   16 * 64 | #1024  #2   * 512 -->   2 * 128 | #256
        #512  | #16384  #64   * 256 -->   64 * 64 | #4096  #8   * 512 -->   8 * 128 | #1024
        #1024 | #65536  #256  * 256 -->  256 * 64 | #16384 #32  * 512 -->  32 * 128 | #4096  #4  * 1024 -->  4 * 256 | #1024
        #2048 | #262144 #1024 * 256 --> 1024 * 64 | #65536 #128 * 512 --> 128 * 128 | #16384 #16 * 1024 --> 16 * 256 | #4096
        
        self.grp_set  = [([[[1, 64]],         [[1,256]],           64], [3,3], [1,1], 3, True ), 
                         ([[[4, 64]],         [[4,256]],          128], [3,3], [1,1], 4, True ), 
                         ([[[16,64],[2,128]], [[2,512],[16,256]], 256], [3,3], [1,1], 6, True ), 
                         ([[[64,64],[8,128]], [[8,512],[64,256]], 512], [3,3], [1,1], 3, True )]



def attn1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['attn']['number']
    shape     = params['attn']['shape']
    rate      = params['attn']['rate']
    stride    = params['attn']['stride']
    padding   = params['attn']['padding']
    use_bias  = params['attn']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)               #[N, H, W, M, C]
    m_shape = [shape[i]+(shape[i]-1)*(rate[i]-1) for i in range(2)]
    
    q_shape = shape + x_shape[3:] + number       #[h, w, M, C, M', C']
    k_shape = shape + x_shape[3:] + [number[1]]  #[1, h, w, M, C,  C']
    q_shape = [reduce(lambda x,y: x*y, q_shape[0:4]), q_shape[4]*q_shape[5]] #[h*w*M*C, M'*C']
    with tf.variable_scope('attn1_'+str(layer)) as scope:
        
        weight_q = tf.get_variable(name='weight_q', shape=q_shape, dtype=dtype, \
                                   #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                   #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                   regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                   trainable=trainable) #(h*w*M*C, M'*C')
        
        weight_k = tf.get_variable(name='weight_k', shape=k_shape, dtype=dtype, \
                                   #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                   #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                   regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                   trainable=trainable) #(h, w, M, C, C')
        
        if use_bias:
            biase_q = tf.get_variable(name='biase_q', shape=q_shape[-1], dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable)
            
            biase_k = tf.get_variable(name='biase_k', shape=k_shape[-1], dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable)
            
            
        def attn_img(tensor_in):
            
            x_shape = get_shape(tensor_in)     #[H, W, M, C]
            
            if padding == 'SAME':
                new_hgt     = int(np.ceil(x_shape[1] / stride[0]))
                new_wdh     = int(np.ceil(x_shape[2] / stride[1]))
                pad_hgt_all = (new_hgt - 1) * stride[0] + m_shape[0] - x_shape[1]
                pad_wdh_all = (new_wdh - 1) * stride[1] + m_shape[1] - x_shape[2]
                pad_top     = pad_hgt_all // 2
                pad_btm     = pad_hgt_all - pad_top
                pad_lft     = pad_wdh_all // 2
                pad_rgt     = pad_wdh_all - pad_lft
                paddings    = [[pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0], [0, 0]]
                tensor_in   = tf.pad(tensor_in, paddings, mode='CONSTANT', constant_values=0)
                x_shape     = get_shape(tensor_in)     #[N, H, W, M, C]
            elif padding == 'VALID':
                new_hgt     = int(np.ceil((x_shape[1] - m_shape[0] + 1) / stride[0]))
                new_wdh     = int(np.ceil((x_shape[2] - m_shape[1] + 1) / stride[1]))
            else:
                raise ValueError('Invalid padding method!')
            
            y_shape    = [x_shape[0], new_hgt, new_wdh] + number
            tensor_out = tf.TensorArray(dtype=tf.float32, size=y_shape[0]*y_shape[1], dynamic_size=False, clear_after_read=True, \
                                        tensor_array_name=None, handle=None, flow=None, infer_shape=True, \
                                        element_shape=number, colocate_with_first_write_call=True) #(H*W, M', C')

            def cond(i, tensor_out):
                c = tf.less(i, y_shape[1]*y_shape[2])
                return c

            def body(i, tensor_out):
                ymn  = i  // y_shape[2] * stride[0]
                xmn  = i  %  y_shape[2] * stride[1]
                ymx  = ymn + m_shape[0]
                xmx  = xmn + m_shape[1]
                fetx = tensor_in[:, ymn:ymx:rate[0], xmn:xmx:rate[1], :, :]       #(N, h, w, M, C)
                fett = tf.reshape(fetx, [y_shape[0], -1])                         #(N, h*w*M*C)
                fetq = tf.matmul(fett, weight_q)                                  #(N, M'*C') (N, h*w*M*C) (h*w*M*C, M'*C')
                fetq = fetq + biase_q if use_bias else fetq                       #(N, M'*C')
                fetq = tf.reshape(fetq, [y_shape[0]]+number)                      #(N, M', C')
                
                fett = tf.expand_dims(fetx, axis=3)                               #(N, h, w, M, C)
                
                fetk = tf.matmul(fett, weight_k)                                  #(h, w, M, 1, C') (h, w, M, 1, C) (h, w, M, C, C')
                fetk = fetk + biase_k if use_bias else fetk                       #(h, w, M, 1, C')
                fetk = tf.reshape(fetk, [-1, number[1]])                          #(h*w*M, C')
                atts = tf.matmul(fetq, fetk, transpose_b=True)                    #(M', h*w*M)
                atts = atts / np.sqrt(number[1])                                  #(M', h*w*M)
                atts = tf.nn.softmax(atts, axis=-1)                               #(M', h*w*M)
                fetk = tf.matmul(atts, fetk)                                      #(M', C') (M', h*w*M) (h*w*M, C')
                fetq = fetq + fetk                                                #(M', C')
                tensor_out = tensor_out.write(i, fetq)                            #(H'*W', M', C')
                return [i+1, tensor_out]
            
            #pra_itrs = max(y_shape[0] * y_shape[1] // 64, 16)
            i = tf.constant(0)
            [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                            parallel_iterations=10, back_prop=True, swap_memory=False)
            tensor_out = tensor_out.stack()                                           #(H'*W', M', C')
            tensor_out = tf.reshape(tensor_out, y_shape)                              #(H', W', M', C')
            return tensor_out
        
        tensor_out = tf.map_fn(attn_img, tensor_in, dtype=tf.float32, parallel_iterations=x_shape[0], \
                               back_prop=True, swap_memory=False, infer_shape=True)    #(N, H', W', M', C')
        print_activations(tensor_out)
    return tensor_out




def proj1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M*C]
    '''
    number    = params['proj']['number']
    shape     = params['proj']['shape']
    rate      = params['proj']['rate']
    stride    = params['proj']['stride']
    padding   = params['proj']['padding']
    use_bias  = params['proj']['use_bias']
    use_attn  = params['proj']['use_attn']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope('proj1_'+str(layer)) as scope:
        
        x_shape   = get_shape(tensor_in)                                                        #[N, H, W, M*C]
        tensor_in = tf.reshape(tensor_in, x_shape[:3]+number[0])                                #(N, H, W, b, r, m, c)
        tensor_in = tf.transpose(tensor_in, [0, 4, 1, 2, 3, 5, 6])                              #(N, r, H, W, b, m, c)
        x_shape   = get_shape(tensor_in)                                                        #[N, r, H, W, b, m, c]
        y_shape   = x_shape[:4] + number[1]                                                     #[N, r, H, W, b', m', c']
        
        if use_attn:
            tensor_in      = tf.reshape(tensor_in, [x_shape[0]*x_shape[1]]+x_shape[2:4]+\
                                                   [x_shape[4]*x_shape[5], x_shape[6]])         #(N*r, H, W, b*m, c)
            params['attn'] = {'number':[number[1][0]*number[1][1], number[1][2]], 'shape':shape, 'rate':rate, \
                              'stride':stride, 'padding':padding, 'use_bias':use_bias}
            tensor_out     = attn1(tensor_in, 0, params, mtrain)                                #(N*r, H, W, b'*m', c')
        else:
            tensor_in      = tf.reshape(tensor_in, [x_shape[0]*x_shape[1]]+x_shape[2:4]+\
                                                   [x_shape[4]*x_shape[5]*x_shape[6]])          #(N*r, H, W, b'*m'*c')
            params['conv'] = {'number':number[1][0]*number[1][1]*number[1][2],    'shape':shape, 'rate':rate, \
                              'stride':stride, 'padding':padding, 'use_bias':use_bias}
            tensor_out     = conv1(tensor_in, 0, params, mtrain)                                #(N*r, H, W, b'*m'*c')
        
        tensor_out = tf.reshape(tensor_out, y_shape)                                            #(N, r, H, W, b', m', c')
        tensor_out = tf.transpose(tensor_out, [0, 2, 3, 4, 1, 5, 6])                            #(N, H, W, b', r, m', c')
        y_shape    = get_shape(tensor_out)                                                      #[N, H, W, b', r, m', c']
        tensor_out = tf.reshape(tensor_out, y_shape[0:3] + \
                                            [y_shape[3]*y_shape[4]*y_shape[5]*y_shape[6]])      #(N, H, W, b'*r*m'*c')
        #tf.summary.histogram('proj', tensor_out)
        print_activations(tensor_out)
    return tensor_out









self.grp_set  = [([[[ 4,1,1,64],[1,1,64]], [[1,1,1,64],[1,1,64]], [[1,1,1,64],[ 4,1,64]]], 3, True ),  # 4
                         ([[[ 8,2,1,64],[2,1,64]], [[1,2,2,64],[1,2,64]], [[2,2,1,64],[ 8,1,64]]], 4, True ),  # 16
                         ([[[16,4,1,64],[4,1,64]], [[1,4,4,64],[1,4,64]], [[4,4,1,64],[16,1,64]]], 6, True ),  # 64
                         ([[[32,8,1,64],[8,1,64]], [[1,8,8,64],[1,8,64]], [[8,8,1,64],[32,1,64]]], 3, True )]  # 256


#the group block setting
        self.grp_set  = [([[[ 4,1,1,64],[1,1,64]], [[1,1,1,64],[1,1,64]], [[1,1,1,64],[ 4,1,64]]], 3, True ),  # 4
                         ([[[ 8,2,1,64],[2,1,64]], [[1,2,2,64],[1,2,64]], [[2,2,1,64],[ 8,1,64]]], 4, True ),  # 16
                         ([[[16,4,1,64],[4,1,64]], [[1,4,4,64],[1,4,64]], [[4,4,1,64],[16,1,64]]], 6, True ),  # 64
                         ([[[32,8,1,64],[8,1,64]], [[1,8,8,64],[1,8,64]], [[8,8,1,64],[32,1,64]]], 3, True )]  # 256





def attn1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['attn']['number']
    shape     = params['attn']['shape']
    rate      = params['attn']['rate']
    stride    = params['attn']['stride']
    padding   = params['attn']['padding']
    use_bias  = params['attn']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)               #[N, H, W, M, C]
    m_shape = [shape[i]+(shape[i]-1)*(rate[i]-1) for i in range(2)]
    
    shape   = shape + x_shape[3:] + number       #[h, w, M, C, M', C']
    shape_q = [reduce(lambda x,y: x*y, shape[0:4]), shape[4]*shape[5]] #[h*w*M*C, M'*C']
    shape_k = shape[0:4] + [shape[5]]            #[h, w, M, C, C']
    with tf.variable_scope('attn1_'+str(layer)) as scope:
        
        weights = tf.get_variable(name='weights', shape=shape, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)     #(h, w, M, C, M', C')
        weight_q = tf.reshape(weights, shape_q)            #(h*w*M*C, M'*C')
        weight_k = tf.reduce_sum(weights, axis=4)          #(h, w, M, C, C')
        
        if use_bias:
            biases = tf.get_variable(name='biases', shape=number, dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable) #(M', C')
            
        if padding == 'SAME':
            new_hgt     = int(np.ceil(x_shape[1] / stride[0]))
            new_wdh     = int(np.ceil(x_shape[2] / stride[1]))
            pad_hgt_all = (new_hgt - 1) * stride[0] + m_shape[0] - x_shape[1]
            pad_wdh_all = (new_wdh - 1) * stride[1] + m_shape[1] - x_shape[2]
            pad_top     = pad_hgt_all // 2
            pad_btm     = pad_hgt_all - pad_top
            pad_lft     = pad_wdh_all // 2
            pad_rgt     = pad_wdh_all - pad_lft
            paddings    = [[0, 0], [pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0], [0, 0]]
            tensor_in   = tf.pad(tensor_in, paddings, mode='CONSTANT', constant_values=0)
            x_shape     = get_shape(tensor_in)     #[N, H, W, M, C]
        elif padding == 'VALID':
            new_hgt     = int(np.ceil((x_shape[1] - m_shape[0] + 1) / stride[0]))
            new_wdh     = int(np.ceil((x_shape[2] - m_shape[1] + 1) / stride[1]))
        else:
            raise ValueError('Invalid padding method!')

        y_shape    = [x_shape[0], new_hgt, new_wdh] + number
        tensor_out = tf.TensorArray(dtype=tf.float32, size=y_shape[1]*y_shape[2], dynamic_size=False, clear_after_read=True, \
                                    tensor_array_name=None, handle=None, flow=None, infer_shape=True, \
                                    element_shape=[y_shape[0]]+number, colocate_with_first_write_call=True) #(H*W, N, M', C')

        def cond(i, tensor_out):
            c = tf.less(i, y_shape[1]*y_shape[2])
            return c

        def body(i, tensor_out):
            ymn  = i  // y_shape[2] * stride[0]
            xmn  = i  %  y_shape[2] * stride[1]
            ymx  = ymn + m_shape[0]
            xmx  = xmn + m_shape[1]
            fetx = tensor_in[:, ymn:ymx:rate[0], xmn:xmx:rate[1], :, :]       #(N, h, w, M, C)
            fett = tf.reshape(fetx, [y_shape[0], -1])                         #(N, h*w*M*C)
            fetq = tf.matmul(fett, weight_q)                                  #(N, M'*C') (N, h*w*M*C) (h*w*M*C, M'*C')
            fetq = tf.reshape(fetq, [y_shape[0]]+number)                      #(N, M', C')
            fett = tf.transpose(fetx, [1, 2, 3, 0, 4])                        #(h, w, M, N, C)
            fetk = tf.matmul(fett, weight_k)                                  #(h, w, M, N, C') (h, w, M, N, C) (h, w, M, C, C')
            fetk = tf.transpose(fetk, [3, 0, 1, 2, 4])                        #(N, h, w, M, C')
            fetk = tf.reshape(fetk, [y_shape[0], -1, number[1]])              #(N, h*w*M, C')
            atts = tf.matmul(fetq, fetk, transpose_b=True)                    #(N, M', h*w*M)
            atts = atts / np.sqrt(number[1])                                  #(N, M', h*w*M)
            atts = tf.nn.softmax(atts, axis=-1)                               #(N, M', h*w*M)
            fetk = tf.matmul(atts, fetk)                                      #(N, M', C') (N, M', h*w*M) (N, h*w*M, C')
            fetq = fetq + fetk                                                #(N, M', C')
            fetq = fetq + biases if use_bias else fetq                        #(N, M', C')
            tensor_out = tensor_out.write(i, fetq)                            #(H'*W', N, M', C')
            return [i+1, tensor_out]
        
        i = tf.constant(0)
        [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                        parallel_iterations=y_shape[1]*y_shape[2], back_prop=True, swap_memory=False)
        tensor_out = tensor_out.stack()                                       #(H'*W', N, M', C')
        tensor_out = tf.transpose(tensor_out, [1, 0, 2, 3])                   #(N, H'*W', M', C')
        tensor_out = tf.reshape(tensor_out, y_shape)                          #(N, H', W', M', C')
        print_activations(tensor_out)
    return tensor_out




def group_block1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    block_setting = params['group_block']['block_setting']

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    tensor_out    = tensor_in
    out_list      = []
    for i, block in enumerate(block_setting):
        
        num_output, num_bottle, rate, stride, use_attn, use_drop, kep_prob, unit_number, unit_trainable = block
        params['com']['trainable'] = unit_trainable
        
        with tf.variable_scope('group_block1_'+str(layer)+'_'+str(i)) as scope:
            
            for j in range(unit_number):
                if j == 0: #the first unit in the block
                    params['group_unit'] = {'num_output':num_output, 'num_bottle':num_bottle, 'rate':rate, 'stride':stride, \
                                            'use_attn':use_attn, 'use_drop':use_drop, 'kep_prob':kep_prob}
                else:      #identity mapping
                    params['group_unit'] = {'num_output':num_output, 'num_bottle':num_bottle, 'rate':rate, 'stride':[1, 1], \
                                            'use_attn':use_attn, 'use_drop':use_drop, 'kep_prob':kep_prob}
                tensor_out = group_unit1(tensor_out, j, params, mtrain)
        out_list.append(tensor_out)
    return out_list




def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M*C]
    1.丢弃或融合特征采用压缩通道中空间位置相邻的特征，而不是像普通CNN那样保留通道而丢弃空间特征。
    2.CNN中丢弃或融合的特征未加以选择，未重视空间上的分布关系；CNN丢失和融合特征的时机不对，由于CNN对通道是完全连接的，
      因此为了增加空间上的关联范围，必须提前使用池化或者步长卷积以丢失未成熟关联的空间特征。
    3.向量神经元的长度应该由希望该向量神经元具有的表达能力的大小决定的，特征丢弃或融合的数量应该由所能用的参数量的大小决定。
    4.压缩和膨胀是针对冗余特征进行的，而特征的连接与组合是针对具有差异性的特征进行的，这正是CNN的要点所在。但是这里的冗余只是针对形状而言的，
      物体的不同的位置、形状细节恰恰存储在这些冗余特征里面，所以对冗余特征不能进行丢失，而只能进行压缩和膨胀。
    '''
    number0    = params['group_unit']['number0'] #[r, b, m, c, b', m', c']
    number1    = params['group_unit']['number1'] #[r, b, m, c, b', m', c']
    number2    = params['group_unit']['number2'] #[r, b, m, c, b', m', c']
    rate       = params['group_unit']['rate']
    stride     = params['group_unit']['stride']
    use_attn   = params['group_unit']['use_attn']
    use_drop   = params['group_unit']['use_drop']
    kep_prob   = params['group_unit']['kep_prob']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]      #(None, 256, 256, 1, 64)
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:
        
        if np.any(np.asarray(stride) > 1):
            params['fold'] = {'stride':stride, 'use_crs':True}
            tensor_in      = fold1(tensor_in, 0, params, mtrain)
        
        x_shape            = get_shape(tensor_in)
        
        if x_shape[3]*x_shape[4] != num_output[0]*num_output[1]*num_output[2]:
            params['proj'] = {'number':num_output, 'shape':[1,1], 'rate':rate[0]+[1,1], 'stride':[1,1,1,1], \
                              'padding':'VALID', 'use_bias':False, 'use_attn':False}
            shortcut       = proj_bn1(tensor_in, 0, params, mtrain)
        elif x_shape[3] != num_output[0]*num_output[1] or x_shape[4] != num_output[2]:
            shortcut       = tf.reshape(tensor_in, x_shape[0:3]+[num_output[0]*num_output[1], num_output[2]])
        else:
            shortcut       = tensor_in
        
        params['proj'] = {'number':num_bottle, 'shape':[1,1], 'rate':rate[0]+[1,1], 'stride':[1,1,1,1], \
                          'padding':'VALID', 'use_bias':False, 'use_attn':False}
        residual       = proj_bn_relu1(tensor_in,  0, params, mtrain)
        
        params['proj'] = {'number':num_bottle, 'shape':[3,3], 'rate':rate[1]+[1,1], 'stride':[1,1,1,1], \
                          'padding':'SAME',  'use_bias':False, 'use_attn':use_attn}
        residual       = proj_bn_relu1(residual,   1, params, mtrain)
        
        params['proj'] = {'number':num_output, 'shape':[1,1], 'rate':rate[0]+[1,1], 'stride':[1,1,1,1], \
                          'padding':'VALID', 'use_bias':False, 'use_attn':False}
        residual       = proj_bn1(residual, 1, params, mtrain)
        
        tensor_out     = residual + shortcut
        tensor_out     = relu1(tensor_out,  0, params, mtrain)
        if use_drop:
            y_shape    = get_shape(tensor_out)  #[N, H, W, M, C]
            params['dropout'] = {'keep_p':kep_prob, 'shape':y_shape[0:4] + [1]}
            tensor_out = dropout1(tensor_out,  0, params, mtrain)
    return tensor_out








def fold1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    stride  = params['fold']['stride']   #[[2, 2], [2, 2]] 
    use_crs = params['fold']['use_crs']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    if isinstance(stride[0],   int):
        stride    = [stride]
    #stride = stride[::-1]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    
    with tf.variable_scope('fold1_'+str(layer)) as scope:
        
        num_srds    = len(stride)
        hgt_srds    = [srd[0] for srd in stride]
        wdh_srds    = [srd[1] for srd in stride]
        hws_srds    = reduce(lambda x,y: x+y, stride  )
        hgt_srd_all = reduce(lambda x,y: x*y, hgt_srds)
        wdh_srd_all = reduce(lambda x,y: x*y, wdh_srds)
        hws_srd_all = hgt_srd_all * wdh_srd_all
        
        hgt_dims    = [           2 + i for i in range(num_srds)]
        wdh_dims    = [num_srds + 3 + i for i in range(num_srds)]
        hws_dims    = [[hgt_dims[i], wdh_dims[i]] for i in range(num_srds)]
        hws_dims    = reduce(lambda x,y: x+y, hws_dims)
        
        new_num     = x_shape[3]  * hws_srd_all
        new_hgt     = x_shape[1] // hgt_srd_all
        new_wdh     = x_shape[2] // wdh_srd_all
        old_hgt     = new_hgt     * hgt_srd_all
        old_wdh     = new_wdh     * wdh_srd_all
        
        if old_hgt != x_shape[1] or old_wdh != x_shape[2]:
            tensor_in = tensor_in[:, :old_hgt, :old_wdh, :, :]
            #x_shape  = get_shape(tensor_in)
        tensor_in   = tf.reshape(tensor_in, [x_shape[0], new_hgt] + hgt_srds + [new_wdh] + wdh_srds + x_shape[3:])
        tensor_in   = tf.transpose(tensor_in, [0, 1, 2+num_srds] + hws_dims + [3+2*num_srds, 4+2*num_srds])
        
        if use_crs:
            for srd in stride:
                assert srd[0] == srd[1] == 2, 'Invalid stride for cross position!'
            indices = np.arange(hws_srd_all)
            indices = np.reshape(indices, [4 for _ in range(len(stride))])
            for i in range(len(stride)):
                indices = np.take(indices, [0,3,1,2], axis=i)
            indices   = np.reshape(indices, [-1])
            tensor_in = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, hws_srd_all] + x_shape[3:])
            tensor_in = tf.gather(tensor_in, indices, axis=3)
        
        tensor_out = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, new_num, x_shape[4]])
        print_activations(tensor_out)
    return tensor_out


def unfold1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    stride  = params['unfold']['stride']
    use_crs = params['unfold']['use_crs']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    if isinstance(stride[0],   int):
        stride    = [stride]
    #stride = stride[::-1]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    
    with tf.variable_scope('unfold1_'+str(layer)) as scope:
        
        num_srds    = len(stride)
        hgt_srds    = [srd[0] for srd in stride]
        wdh_srds    = [srd[1] for srd in stride]
        hws_srds    = reduce(lambda x,y: x+y, stride  )
        hgt_srd_all = reduce(lambda x,y: x*y, hgt_srds)
        wdh_srd_all = reduce(lambda x,y: x*y, wdh_srds)
        hws_srd_all = hgt_srd_all * wdh_srd_all
        
        hgt_dims    = [3 + 2 * i for i in range(num_srds)]
        wdh_dims    = [4 + 2 * i for i in range(num_srds)]
        
        new_num     = x_shape[3] // hws_srd_all
        new_hgt     = x_shape[1]  * hgt_srd_all
        new_wdh     = x_shape[2]  * wdh_srd_all
        old_num     = new_num     * hws_srd_all
        
        if old_num != x_shape[3]:
            tensor_in = tensor_in[:, :, :, :old_num, :]
            #x_shape  = get_shape(tensor_in)
        
        if use_crs:
            for srd in stride:
                assert srd[0] == srd[1] == 2, 'Invalid stride for cross position!'
            indices = np.arange(hws_srd_all)
            indices = np.reshape(indices, [4 for _ in range(len(stride))])
            for i in range(len(stride)):
                indices = np.take(indices, [0,2,3,1], axis=i)
            indices   = np.reshape(indices, [-1])
            tensor_in = tf.reshape(tensor_in, x_shape[0:3] + [hws_srd_all] + [new_num, x_shape[4]])
            tensor_in = tf.gather(tensor_in, indices, axis=3)
        
        tensor_in   = tf.reshape(tensor_in, x_shape[0:3] + hws_srds + [new_num, x_shape[4]])
        tensor_in   = tf.transpose(tensor_in, [0,1] + hgt_dims + [2] + wdh_dims + [3+2*num_srds, 4+2*num_srds])
        tensor_out  = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, new_num, x_shape[4]])
        print_activations(tensor_out)
    return tensor_out



def proj_relu_dropout1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)                   #[N, H, W, M, C]
    params['dropout']['shape'] = x_shape[0:4] + [1]  #[N, H, W, M, 1]
    
    with tf.variable_scope('proj_relu_dropout1_'+str(layer)) as scope:
        relu       = proj_relu1(tensor_in, 0, params, mtrain) 
        tensor_out = dropout1(relu, 0, params, mtrain)
    return tensor_out

def proj_bn_relu_dropout1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)                   #[N, H, W, M, C]
    params['dropout']['shape'] = x_shape[0:4] + [1]  #[N, H, W, M, 1]
    
    with tf.variable_scope('proj_relu_dropout1_'+str(layer)) as scope:
        relu       = proj_bn_relu1(tensor_in, 0, params, mtrain) 
        tensor_out = dropout1(relu, 0, params, mtrain)
    return tensor_out












def conv6(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['conv']['number']        #[4, 64]
    shape     = params['conv']['shape']         #[3,  3]
    rate      = params['conv']['rate']          #[1,  1]
    stride    = params['conv']['stride']        #[1,  1]
    padding   = params['conv']['padding']
    use_bias  = params['conv']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape = tensor_in.get_shape().as_list()
    x_shape  = get_shape(tensor_in)             #[N, H, W, M, C]
    shape    = shape + [x_shape[3]*x_shape[4], number[0]*number[1]]
    stride   = [1, stride[0], stride[1], 1]
    rate     = [1,   rate[0],   rate[1], 1]
    
    with tf.variable_scope('conv6_'+str(layer), reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=shape, dtype=dtype, \
                                 #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                 #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
                                 regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                 trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name='biases', shape=[number[0]*number[1]], dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        
        tensor_in  = tf.reshape(tensor_in, x_shape[0:3]+[x_shape[3]*x_shape[4]])
        conv       = tf.nn.conv2d(tensor_in, kernel, stride, padding=padding, dilations=rate)
        
        if use_bias:
            tensor_out = tf.nn.bias_add(conv, biases)
        else:
            tensor_out = conv
        
        y_shape    = get_shape(tensor_out)      #[N, H', W', M'*C']
        tensor_out = tf.reshape(tensor_out, y_shape[0:3]+number)
        #tf.summary.histogram('conv', tensor_out)
        print_activations(tensor_out)
    return tensor_out


def conv_bn6(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    params['conv']['use_bias'] = False
    
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('conv_bn6_'+str(layer)) as scope:
        conv       = conv6(tensor_in, 0, params, mtrain)
        y_shape    = get_shape(conv)
        conv       = tf.reshape(conv, y_shape[0:3]+[y_shape[3]*y_shape[4]])
        bn         = batchnorm1(conv, 0, params, mtrain)
        tensor_out = tf.reshape(bn,   y_shape)
        print_activations(tensor_out)
    return tensor_out


def conv_relu6(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('conv_relu6_'+str(layer)) as scope:
        conv       = conv6(tensor_in, 0, params, mtrain)
        tensor_out = relu1(conv,      0, params, mtrain)
    return tensor_out


def conv_bn_relu6(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('conv_bn_relu6_'+str(layer)) as scope:
        bn         = conv_bn6(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain)
    return tensor_out 





def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    1.减少或融合特征采用压缩通道中空间位置相邻的特征，而不是像普通CNN那样保留通道而丢弃空间特征。
    2.CNN中丢失和融合的特征未加以选择，未重视空间上的分布关系；CNN丢失和融合特征的时机不对，由于CNN对通道是完全连接的，
      因此为了增加空间上的关联范围，必须提前使用池化或者步长卷积以丢失未成熟关联的空间特征。
    3.向量神经元的长度应该由希望该向量神经元具有的表达能力的大小决定的。
    '''
    num_output = params['group_unit']['num_output']  #[4, 64]
    num_bottle = params['group_unit']['num_bottle']  #[4, 16]
    rate       = params['group_unit']['rate']
    stride     = params['group_unit']['stride']
    use_attn   = params['group_unit']['use_attn']
    use_drop   = params['group_unit']['use_drop']
    kep_prob   = params['group_unit']['kep_prob']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]      #(None, 256, 256, 1, 64)
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:
        
        if np.any(np.asarray(stride) > 1):
            params['fold'] = {'stride':stride, 'use_crs':True}
            tensor_in      = fold1(tensor_in, 0, params, mtrain)
        
        x_shape            = get_shape(tensor_in)
        
        if x_shape[3] * x_shape[4] != num_output[0] * num_output[1]:
            params['proj'] = {'number':num_output, 'shape':[1,1], 'rate':rate[0]+[1,1], 'stride':[1,1,1,1], \
                              'padding':'VALID', 'use_bias':False, 'use_attn':False}
            shortcut       = proj_bn1(tensor_in, 0, params, mtrain)
        elif x_shape[3] != num_output[0] or x_shape[4] != num_output[1]:
            shortcut       = tf.reshape(tensor_in, x_shape[0:3]+num_output)
        else:
            shortcut       = tensor_in
        
        params['proj'] = {'number':num_bottle, 'shape':[1,1], 'rate':rate[0]+[1,1], 'stride':[1,1,1,1], \
                          'padding':'VALID', 'use_bias':False, 'use_attn':False}
        residual       = proj_bn_relu1(tensor_in,  0, params, mtrain)
        
        params['proj'] = {'number':num_bottle, 'shape':[3,3], 'rate':rate[1]+[1,1], 'stride':[1,1,1,1], \
                          'padding':'SAME',  'use_bias':False, 'use_attn':use_attn}
        residual       = proj_bn_relu1(residual,   1, params, mtrain)
        
        params['proj'] = {'number':num_output, 'shape':[1,1], 'rate':rate[0]+[1,1], 'stride':[1,1,1,1], \
                          'padding':'VALID', 'use_bias':False, 'use_attn':False}
        residual       = proj_bn1(residual, 1, params, mtrain)
        
        tensor_out     = residual + shortcut
        tensor_out     = relu1(tensor_out,  0, params, mtrain)
        if use_drop:
            y_shape    = get_shape(tensor_out)  #[N, H, W, M, C]
            params['dropout'] = {'keep_p':kep_prob, 'shape':y_shape[0:4] + [1]}
            tensor_out = dropout1(tensor_out,  0, params, mtrain)
    return tensor_out









def attn1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['attn']['number']
    shape     = params['attn']['shape']
    rate      = params['attn']['rate']
    stride    = params['attn']['stride']
    padding   = params['attn']['padding']
    use_bias  = params['attn']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)               #[N, H, W, M, C]
    m_shape = [shape[i]+(shape[i]-1)*(rate[i]-1) for i in range(2)]
    
    q_shape = shape + x_shape[3:] + number       #[h, w, M, C, M', C']
    k_shape = shape + x_shape[3:] + [number[1]]  #[h, w, M, C, C']
    q_shape = [reduce(lambda x,y: x*y, q_shape[0:4]), q_shape[4]*q_shape[5]] #[h*w*M*C, M'*C']
    with tf.variable_scope('attn1_'+str(layer)) as scope:
        
        weight_q = tf.get_variable(name='weight_q', shape=q_shape, dtype=dtype, \
                                   #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                   #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                   regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                   trainable=trainable)    #(h*w*M*C, M'*C')
        
        weight_k = tf.get_variable(name='weight_k', shape=k_shape, dtype=dtype, \
                                   #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                   #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                   regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                   trainable=trainable)    #(h, w, M, C, C')
        
        if use_bias:
            biase_q = tf.get_variable(name='biase_q', shape=q_shape[-1], dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable) #(M'*C')
            
            biase_k = tf.get_variable(name='biase_k', shape=k_shape[:3]+[1,k_shape[-1]], dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable) #(h, w, M, 1, C')
            
        if padding == 'SAME':
            new_hgt     = int(np.ceil(x_shape[1] / stride[0]))
            new_wdh     = int(np.ceil(x_shape[2] / stride[1]))
            pad_hgt_all = (new_hgt - 1) * stride[0] + m_shape[0] - x_shape[1]
            pad_wdh_all = (new_wdh - 1) * stride[1] + m_shape[1] - x_shape[2]
            pad_top     = pad_hgt_all // 2
            pad_btm     = pad_hgt_all - pad_top
            pad_lft     = pad_wdh_all // 2
            pad_rgt     = pad_wdh_all - pad_lft
            paddings    = [[0, 0], [pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0], [0, 0]]
            tensor_in   = tf.pad(tensor_in, paddings, mode='CONSTANT', constant_values=0)
            x_shape     = get_shape(tensor_in)     #[N, H, W, M, C]
        elif padding == 'VALID':
            new_hgt     = int(np.ceil((x_shape[1] - m_shape[0] + 1) / stride[0]))
            new_wdh     = int(np.ceil((x_shape[2] - m_shape[1] + 1) / stride[1]))
        else:
            raise ValueError('Invalid padding method!')

        y_shape    = [x_shape[0], new_hgt, new_wdh] + number
        tensor_out = tf.TensorArray(dtype=tf.float32, size=y_shape[1]*y_shape[2], dynamic_size=False, clear_after_read=True, \
                                    tensor_array_name=None, handle=None, flow=None, infer_shape=True, \
                                    element_shape=[y_shape[0]]+number, colocate_with_first_write_call=True) #(H*W, N, M', C')

        def cond(i, tensor_out):
            c = tf.less(i, y_shape[1]*y_shape[2])
            return c

        def body(i, tensor_out):
            ymn  = i  // y_shape[2] * stride[0]
            xmn  = i  %  y_shape[2] * stride[1]
            ymx  = ymn + m_shape[0]
            xmx  = xmn + m_shape[1]
            fetx = tensor_in[:, ymn:ymx:rate[0], xmn:xmx:rate[1], :, :]       #(N, h, w, M, C)
            fett = tf.reshape(fetx, [y_shape[0], -1])                         #(N, h*w*M*C)
            fetq = tf.matmul(fett, weight_q)                                  #(N, M'*C') (N, h*w*M*C) (h*w*M*C, M'*C')
            fetq = fetq + biase_q if use_bias else fetq                       #(N, M'*C')
            fetq = tf.reshape(fetq, [y_shape[0]]+number)                      #(N, M', C')
            fett = tf.transpose(fetx, [1, 2, 3, 0, 4])                        #(h, w, M, N, C)
            fetk = tf.matmul(fett, weight_k)                                  #(h, w, M, N, C') (h, w, M, N, C) (h, w, M, C, C')
            fetk = fetk + biase_k if use_bias else fetk                       #(h, w, M, N, C')
            fetk = tf.transpose(fetk, [3, 0, 1, 2, 4])                        #(N, h, w, M, C')
            fetk = tf.reshape(fetk, [y_shape[0], -1, number[1]])              #(N, h*w*M, C')
            atts = tf.matmul(fetq, fetk, transpose_b=True)                    #(N, M', h*w*M)
            atts = atts / np.sqrt(number[1])                                  #(N, M', h*w*M)
            atts = tf.nn.softmax(atts, axis=-1)                               #(N, M', h*w*M)
            fetk = tf.matmul(atts, fetk)                                      #(N, M', C') (N, M', h*w*M) (N, h*w*M, C')
            fetq = fetq + fetk                                                #(N, M', C')
            tensor_out = tensor_out.write(i, fetq)                            #(H'*W', N, M', C')
            return [i+1, tensor_out]
        
        i = tf.constant(0)
        [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                        parallel_iterations=y_shape[1]*y_shape[2], back_prop=True, swap_memory=False)
        tensor_out = tensor_out.stack()                                       #(H'*W', N, M', C')
        tensor_out = tf.transpose(tensor_out, [1, 0, 2, 3])                   #(N, H'*W', M', C')
        tensor_out = tf.reshape(tensor_out, y_shape)                          #(N, H', W', M', C')
        print_activations(tensor_out)
    return tensor_out











def attn1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['attn']['number']
    shape     = params['attn']['shape']
    rate      = params['attn']['rate']
    stride    = params['attn']['stride']
    padding   = params['attn']['padding']
    use_bias  = params['attn']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)               #[N, H, W, M, C]
    m_shape = [shape[i]+(shape[i]-1)*(rate[i]-1) for i in range(2)]
    
    q_shape = shape + x_shape[3:] + number       #[h, w, M, C, M', C']
    k_shape = shape + x_shape[3:] + [number[1]]  #[h, w, M, C, C']
    q_shape = [reduce(lambda x,y: x*y, q_shape[0:4]), q_shape[4]*q_shape[5]] #[h*w*M*C, M'*C']
    with tf.variable_scope('attn1_'+str(layer)) as scope:
        
        weight_q = tf.get_variable(name='weight_q', shape=q_shape, dtype=dtype, \
                                   #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                   #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                   regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                   trainable=trainable) #(h*w*M*C, M'*C')
        
        weight_k = tf.get_variable(name='weight_k', shape=k_shape, dtype=dtype, \
                                   #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                   #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                   regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                   trainable=trainable) #(h, w, M, C, C')
        
        if use_bias:
            biase_q = tf.get_variable(name='biase_q', shape=q_shape[-1], dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable)
            
            biase_k = tf.get_variable(name='biase_k', shape=k_shape[-1], dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable)
            
            
        def attn_img(tensor_in):
            
            x_shape = get_shape(tensor_in)     #[H, W, M, C]
            
            if padding == 'SAME':
                new_hgt     = int(np.ceil(x_shape[0] / stride[0]))
                new_wdh     = int(np.ceil(x_shape[1] / stride[1]))
                pad_hgt_all = (new_hgt - 1) * stride[0] + m_shape[0] - x_shape[0]
                pad_wdh_all = (new_wdh - 1) * stride[1] + m_shape[1] - x_shape[1]
                pad_top     = pad_hgt_all // 2
                pad_btm     = pad_hgt_all - pad_top
                pad_lft     = pad_wdh_all // 2
                pad_rgt     = pad_wdh_all - pad_lft
                paddings    = [[pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0], [0, 0]]
                tensor_in   = tf.pad(tensor_in, paddings, mode='CONSTANT', constant_values=0)
                x_shape     = get_shape(tensor_in)     #[H, W, M, C]
            elif padding == 'VALID':
                new_hgt     = int(np.ceil((x_shape[0] - m_shape[0] + 1) / stride[0]))
                new_wdh     = int(np.ceil((x_shape[1] - m_shape[1] + 1) / stride[1]))
            else:
                raise ValueError('Invalid padding method!')
            
            y_shape    = [new_hgt, new_wdh] + number
            tensor_out = tf.TensorArray(dtype=tf.float32, size=y_shape[0]*y_shape[1], dynamic_size=False, clear_after_read=True, \
                                        tensor_array_name=None, handle=None, flow=None, infer_shape=True, \
                                        element_shape=number, colocate_with_first_write_call=True) #(H*W, M', C')

            def cond(i, tensor_out):
                c = tf.less(i, y_shape[0]*y_shape[1])
                return c

            def body(i, tensor_out):
                ymn  = i  // y_shape[1] * stride[0]
                xmn  = i  %  y_shape[1] * stride[1]
                ymx  = ymn + m_shape[0]
                xmx  = xmn + m_shape[1]
                fetx = tensor_in[ymn:ymx:rate[0], xmn:xmx:rate[1], :, :]          #(h, w, M, C)
                fett = tf.reshape(fetx, [1, -1])                                  #(1, h*w*M*C)
                fetq = tf.matmul(fett, weight_q)                                  #(1, M'*C') (1, h*w*M*C) (h*w*M*C, M'*C')
                fetq = fetq + biase_q if use_bias else fetq                       #(1, M'*C')
                fetq = tf.reshape(fetq, number)                                   #(M', C')
                fett = tf.expand_dims(fetx, axis=3)                               #(h, w, M, 1, C)
                fetk = tf.matmul(fett, weight_k)                                  #(h, w, M, 1, C') (h, w, M, 1, C) (h, w, M, C, C')
                fetk = fetk + biase_k if use_bias else fetk                       #(h, w, M, 1, C')
                fetk = tf.reshape(fetk, [-1, number[1]])                          #(h*w*M, C')
                atts = tf.matmul(fetq, fetk, transpose_b=True)                    #(M', h*w*M)
                atts = atts / np.sqrt(number[1])                                  #(M', h*w*M)
                atts = tf.nn.softmax(atts, axis=-1)                               #(M', h*w*M)
                fetk = tf.matmul(atts, fetk)                                      #(M', C') (M', h*w*M) (h*w*M, C')
                fetq = fetq + fetk                                                #(M', C')
                tensor_out = tensor_out.write(i, fetq)                            #(H'*W', M', C')
                return [i+1, tensor_out]
            
            #pra_itrs = max(y_shape[0] * y_shape[1] // 64, 16)
            i = tf.constant(0)
            [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                            parallel_iterations=10, back_prop=True, swap_memory=False)
            tensor_out = tensor_out.stack()                                           #(H'*W', M', C')
            tensor_out = tf.reshape(tensor_out, y_shape)                              #(H', W', M', C')
            return tensor_out
        
        tensor_out = tf.map_fn(attn_img, tensor_in, dtype=tf.float32, parallel_iterations=10, \
                               back_prop=True, swap_memory=False, infer_shape=True)   #(N, H', W', M', C')
        print_activations(tensor_out)
    return tensor_out


def proj1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['proj']['number']    #[4, 64]
    shape     = params['proj']['shape']     #[3,  3]
    rate      = params['proj']['rate']      #[b, 2, 1, 1]
    stride    = params['proj']['stride']    #[2, 2, 1, 1]
    padding   = params['proj']['padding']
    use_bias  = params['proj']['use_bias']
    use_attn  = params['proj']['use_attn']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope('proj1_'+str(layer), reuse=reuse) as scope:
        
        if np.any(np.asarray(stride[0:2]) > 1):
            params['fold'] = {'stride':stride[0:2], 'use_crs':True}
            tensor_in      = fold1(tensor_in, 0, params, mtrain)
        
        x_shape   = get_shape(tensor_in)                                                        #[N, H, W, M, C]
        
        #block之内的联系紧密，block之外的联系松散
        tensor_in = tf.reshape(tensor_in, x_shape[:3]+[rate[0],x_shape[3]//rate[0],x_shape[4]]) #(N, H, W, b, M', C)
        tensor_in = tf.transpose(tensor_in, [3, 0, 1, 2, 4, 5])                                 #(b, N, H, W, M', C)
        x_shape   = get_shape(tensor_in)                                                        #[b, N, H, W, M', C]
        
        #根据通道上的膨胀率，再次对向量神经元进行划分
        tensor_in = tf.reshape(tensor_in, x_shape[:4]+[x_shape[4]//rate[1],rate[1],x_shape[5]]) #(b, N, H, W, M", r, C)
        tensor_in = tf.transpose(tensor_in, [0, 5, 1, 2, 3, 4, 6])                              #(b, r, N, H, W, M", C)
        x_shape   = get_shape(tensor_in)                                                        #[b, r, N, H, W, M", C]
        y_shape   = x_shape[0:5] + number                                                       #[b, r, N, H, W, M_, C_]
        tensor_in = tf.reshape(tensor_in, [x_shape[0]*x_shape[1]]+x_shape[2:])                  #(b*r,  N, H, W, M_, C_)
        
        if use_attn:
            params['attn'] = {'number':number, 'shape':shape, 'rate':rate[2:], 'stride':stride[2:], \
                              'padding':padding, 'use_bias':use_bias}
            tensor_out = tf.map_fn(lambda x: attn1(x, 0, params, None), tensor_in, dtype=tf.float32, \
                                   parallel_iterations=10, \
                                   back_prop=True, swap_memory=False, infer_shape=True)         #(b*r,  N, H, W, M_, C_)
        else:
            params['conv'] = {'number':number, 'shape':shape, 'rate':rate[2:], 'stride':stride[2:], \
                              'padding':padding, 'use_bias':use_bias}
            tensor_out = tf.map_fn(lambda x: conv6(x, 0, params, None), tensor_in, dtype=tf.float32, \
                                   parallel_iterations=10, \
                                   back_prop=True, swap_memory=False, infer_shape=True)         #(b*r,  N, H, W, M_, C_)
        
        tensor_out = tf.reshape(tensor_out, y_shape)                                            #(b, r, N, H, W, M_, C_)
        tensor_out = tf.transpose(tensor_out, [2, 3, 4, 0, 5, 1, 6])                            #(N, H, W, b, M_, r, C_)
        y_shape    = get_shape(tensor_out)                                                      #[N, H, W, b, M_, r, C_]
        tensor_out = tf.reshape(tensor_out, y_shape[0:3] + \
                                            [y_shape[3]*y_shape[4]*y_shape[5], y_shape[6]])     #(N, H, W, M*, C_)
        #tf.summary.histogram('proj', tensor_out)
        print_activations(tensor_out)
    return tensor_out












def attn1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['attn']['number']
    shape     = params['attn']['shape']
    rate      = params['attn']['rate']
    stride    = params['attn']['stride']
    padding   = params['attn']['padding']
    use_bias  = params['attn']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)             #[N, H, W, M, C]
    m_shape = [shape[i]+(shape[i]-1)*(rate[i]-1) for i in range(2)]
    shape   = shape + x_shape[3:] + number     #[h, w, M, C, M', C']
    
    with tf.variable_scope('attn1_'+str(layer)) as scope:
        
        weights = tf.get_variable(name='weights', shape=shape, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name='biases', shape=number, dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        
        def attn_img(tensor_in):
            
            x_shape = get_shape(tensor_in)     #[H, W, M, C]
            
            if padding == 'SAME':
                new_hgt     = int(np.ceil(x_shape[0] / stride[0]))
                new_wdh     = int(np.ceil(x_shape[1] / stride[1]))
                pad_hgt_all = (new_hgt - 1) * stride[0] + m_shape[0] - x_shape[0]
                pad_wdh_all = (new_wdh - 1) * stride[1] + m_shape[1] - x_shape[1]
                pad_top     = pad_hgt_all // 2
                pad_btm     = pad_hgt_all - pad_top
                pad_lft     = pad_wdh_all // 2
                pad_rgt     = pad_wdh_all - pad_lft
                paddings    = [[pad_top, pad_btm], [pad_lft, pad_rgt], [0, 0], [0, 0]]
                tensor_in   = tf.pad(tensor_in, paddings, mode='CONSTANT', constant_values=0)
                x_shape     = get_shape(tensor_in)     #[H, W, M, C]
            elif padding == 'VALID':
                new_hgt     = int(np.ceil((x_shape[0] - m_shape[0] + 1) / stride[0]))
                new_wdh     = int(np.ceil((x_shape[1] - m_shape[1] + 1) / stride[1]))
            else:
                raise ValueError('Invalid padding method!')
            
            y_shape    = [new_hgt, new_wdh] + number
            tensor_out = tf.TensorArray(dtype=tf.float32, size=y_shape[0]*y_shape[1], dynamic_size=False, clear_after_read=True, \
                                        tensor_array_name=None, handle=None, flow=None, infer_shape=True, \
                                        element_shape=number, colocate_with_first_write_call=True) #(H*W, M', C')

            def cond(i, tensor_out):
                c = tf.less(i, y_shape[0]*y_shape[1])
                return c

            def body(i, tensor_out):
                ymn  = i  // y_shape[1] * stride[0]
                xmn  = i  %  y_shape[1] * stride[1]
                ymx  = ymn + m_shape[0]
                xmx  = xmn + m_shape[1]
                fetx = tensor_in[ymn:ymx:rate[0], xmn:xmx:rate[1], :, :]              #(h, w, M, C) (h, w, M, C, M', C')
                fety = tf.einsum('ijkl,ijklmn->mn',   fetx, weights)                  #(M', C')
                fetz = tf.einsum('ijkl,ijklmn->ijkn', fetx, weights)                  #(h, w, M, C') 
                '''
                fetx = tf.reshape(fetx, [1, -1])                                      #(1, h*w*M*C)
                wgts = tf.reshape(weights, [-1, number[0]*number[1]])                 #(h*w*M*C, M'*C')
                fety = tf.matmul(fetx, wgts)                                          #(1, M'*C')
                fety = tf.reshape(fety, number)                                       #(M', C')
                '''
                #每个向量神经元C都预测出了M'个向量神经元C'，取这M'个向量神经元C'的均值C'，作为向量神经元C预测值
                #该预测值C'会比输入C更偏向于合理的输出，从而在和所有输入预测的输出做相似性度量时，会得到更明确的相似性值
                
                atts = tf.einsum('ijkn,mn->ijkm',     fetz, fety)                     #(h, w, M, M')
                #softmax
                #wgts= weights * atts[:, :, :, tf.newaxis, :, tf.newaxis]             #(h, w, M, C, M', C')
                fety = tf.einsum('ijkl,ijklmn->mn',   fetx, wgts) + fety              #(M', C')
                if use_bias:
                    fety = fety + biases
                tensor_out = tensor_out.write(i, fety)                                #(H'*W', M', C')
                return [i+1, tensor_out]
            
            #pra_itrs = max(y_shape[0] * y_shape[1] // 64, 16)
            
            i = tf.constant(0)
            [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                            parallel_iterations=10, back_prop=True, swap_memory=True)
            tensor_out = tensor_out.stack()                                           #(H'*W', M', C')
            tensor_out = tf.reshape(tensor_out, y_shape)                              #(H', W', M', C')
            return tensor_out
        
        tensor_out = tf.map_fn(attn_img, tensor_in, dtype=tf.float32, parallel_iterations=10, \
                               back_prop=True, swap_memory=True, infer_shape=True)    #(N, H', W', M', C')
        print_activations(tensor_out)
    return tensor_out


















def attn1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['attn']['number']
    shape     = params['attn']['shape']
    rate      = params['attn']['rate']
    stride    = params['attn']['stride']
    padding   = params['attn']['padding']
    use_bias  = params['attn']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)             #[N, H, W, M, C]
    m_shape = [shape[i]+(shape[i]-1)*(rate[i]-1) for i in range(2)]
    shape   = shape + x_shape[3:] + number     #[h, w, M, C, M', C']
    
    with tf.variable_scope('attn1_'+str(layer)) as scope:
        
        weights = tf.get_variable(name='weights', shape=shape, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name='biases', shape=number, dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        
        def attn_img(tensor_in):
            
            if padding == 'SAME':
                tf.pad
            
            x_shape = get_shape(tensor_in)     #[H, W, M, C]
            
            if padding == 'VALID':
                y_shape = [int(np.ceil((x_shape[i]-shape[i]+1)/stride[i])) for i in range(2)] + number
                crd_sta = [m_shape[i]//2 for i in range(2)]
                crd_end = [crd_sta[i]+(y_shape[i]-1)*stride[i] for i in range(2)]
            else:
                y_shape = [int(np.ceil(x_shape[i]/stride[i])) for i in range(2)] + number
                crd_sta = [0, 0]
                crd_end = 

            tensor_out = tf.TensorArray(dtype=tf.float32, size=y_shape[0]*y_shape[1], dynamic_size=False, clear_after_read=True, \
                                        tensor_array_name=None, handle=None, flow=None, infer_shape=True, \
                                        element_shape=number, colocate_with_first_write_call=True) #(H*W, M', C')

            def cond(i, tensor_out):
                c = tf.less(i, y_shape[0]*y_shape[1])
                return c

            def body(i, tensor_out):
                ycd   = i  // y_shape[1] * stride[0] + crd_sta[0]
                xcd   = i  %  y_shape[1] * stride[1] + crd_sta[1]
                crd   = tf.stack([ycd, xcd], axis=0)                                  #(2)
                ymn   = ycd - ((shape[0] - 1) // 2) * rate[0]
                xmn   = xcd - ((shape[1] - 1) // 2) * rate[1]
                '''
                ycds  = tf.concat([[ymn], tf.tile([rate[0]], [shape[0]-1])], axis=0)
                xcds  = tf.concat([[xmn], tf.tile([rate[1]], [shape[1]-1])], axis=0)
                ycds  = tf.cumsum(ycds, axis=0, exclusive=False, reverse=False)
                xcds  = tf.cumsum(xcds, axis=0, exclusive=False, reverse=False)
                yixs  = tf.where(tf.logical_and(ycds>=0, ycds<x_shape[0]))[:, 0]
                ycds  = tf.gather(ycds, yixs)
                xixs  = tf.where(tf.logical_and(xcds>=0, xcds<x_shape[1]))[:, 0]
                xcds  = tf.gather(xcds, xixs)
                ycds  = tf.tile(ycds[:, tf.newaxis], [1, tf.shape(xcds)[0]])          #(h, w)
                xcds  = tf.tile(xcds[tf.newaxis, :], [tf.shape(ycds)[0], 1])          #(h, w)
                crds  = tf.stack([ycds, xcds], axis=-1)                               #(h, w, 2)
                yixs  = tf.tile(yixs[:, tf.newaxis], [1, tf.shape(xixs)[0]])          #(h, w)
                xixs  = tf.tile(xixs[tf.newaxis, :], [tf.shape(yixs)[0], 1])          #(h, w)
                idxs  = tf.stack([yixs, xixs], axis=-1)                               #(h, w, 2)
                wgts  = tf.gather_nd(weights,   idxs)                                 #(h, w, M, C, M', C')
                fetx  = tf.gather_nd(tensor_in, crds)                                 #(h, w, M, C)
                fety  = tf.einsum('ijkl,ijklmn->mn',   fetx, wgts)                    #(M', C')
                #每个向量神经元C都预测出了M'个向量神经元C'，取这M'个向量神经元C'的均值C'，作为向量神经元C预测值
                #该预测值C'会比输入C更偏向于合理的输出，从而在和所有输入预测的输出做相似性度量时，会得到更明确的相似性值
                fetz  = tf.einsum('ijkl,ijklmn->ijkn', fetx, wgts)                    #(h, w, M, C') 
                atts  = tf.einsum('ijkn,mn->ijkm',     fetz, fety)                    #(h, w, M, M')
                wgts  = wgts * atts[:, :, :, tf.newaxis, :, tf.newaxis]               #(h, w, M, C, M', C')
                fety  = tf.einsum('ijkl,ijklmn->mn',   fetx, wgts) + fety             #(M', C')
                fety  = fety + biases if use_bias else fety
                tensor_out = tensor_out.write(i, fety)                                #(H'*W', M', C')
                '''
                tensor_out = tensor_out.write(i, tensor_in[crd[0], crd[1]])
                return [i+1, tensor_out]
            
            #pra_itrs = max(y_shape[0] * y_shape[1] // 64, 16)
            
            i = tf.constant(0)
            [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                            parallel_iterations=1, back_prop=True, swap_memory=True)
            tensor_out = tensor_out.stack()                                           #(H'*W', M', C')
            tensor_out = tf.reshape(tensor_out, y_shape)                              #(H', W', M', C')
            return tensor_out
        
        tensor_out = tf.map_fn(attn_img, tensor_in, dtype=tf.float32, parallel_iterations=1, \
                               back_prop=True, swap_memory=True, infer_shape=True)    #(N, H', W', M', C')
        print_activations(tensor_out)
    return tensor_out













def fold1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    stride  = params['fold']['stride']   #[[2, 2], [2, 2]] 
    use_crs = params['fold']['use_crs']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    if isinstance(stride[0],   int):
        stride    = [stride]
    #stride = stride[::-1]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    
    with tf.variable_scope('fold1_'+str(layer)) as scope:
        
        num_srds    = len(stride)
        hgt_srds    = [srd[0] for srd in stride]
        wdh_srds    = [srd[1] for srd in stride]
        hgt_srd_all = reduce(lambda x,y: x*y, hgt_srds)
        wdh_srd_all = reduce(lambda x,y: x*y, wdh_srds)
        hws_srd_all = hgt_srd_all * wdh_srd_all
        
        hgt_dims    = [           2 + i for i in range(num_srds)]
        wdh_dims    = [num_srds + 3 + i for i in range(num_srds)]
        hws_dims    = [[hgt_dims[i], wdh_dims[i]] for i in range(num_srds)]
        hws_dims    = reduce(lambda x,y: x+y, hws_dims)
        
        new_num     = x_shape[3]  * hws_srd_all
        new_hgt     = x_shape[1] // hgt_srd_all
        new_wdh     = x_shape[2] // wdh_srd_all
        old_hgt     = new_hgt     * hgt_srd_all
        old_wdh     = new_wdh     * wdh_srd_all
        
        if old_hgt != x_shape[1] or old_wdh != x_shape[2]:
            tensor_in = tensor_in[:, :old_hgt, :old_wdh, :, :]
            #x_shape  = get_shape(tensor_in)
        tensor_in   = tf.reshape(tensor_in, [x_shape[0], new_hgt] + hgt_srds + [new_wdh] + wdh_srds + x_shape[3:])
        tensor_in   = tf.transpose(tensor_in, [0, 1, 2+num_srds] + hws_dims + [3+2*num_srds, 4+2*num_srds])
        if use_crs:
            indices   = [np.array([1,2,4,3]) for srd in stride for i in range(srd[0]*srd[1]//4)]
            
            for srd in stride:
                
                idxs = np.arange(srd[0]*srd[1])
                hsds = [2 for _ in range(srd[0]//2)]
                wsds = [2 for _ in range(srd[1]//2)]
                idxs = np.reshape(idxs, hsds+wsds)
                hdms = [          i for i in range(srd[0]//2)]
                wdms = [srd[0]//2+i for i in range(srd[1]//2)]
                dims = []
                leh0 = srd[0] * srd[1]
                idx0 = np.arange(leh0)
                leh1 = leh0 // 4  * 4
                idx1 = idx0[:leh1]
                hwsd = [[2, 2] for _ in range(srd[0]//2) for _ in range(srd[1]//2)]
                idx1 = np.reshape(idx1, hwsd)
                
                
                wsrd = 
            
            tensor_in = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, hws_srd_all] + x_shape[3:])
            tensor_in = tf.gather(tensor_in,  idxs, axis=3)
        tensor_out  = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, new_num, x_shape[4]])
        print_activations(tensor_out)
    return tensor_out


def unfold1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    stride = params['unfold']['stride']

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    if isinstance(stride[0],   int):
        stride    = [stride]
    #stride = stride[::-1]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    
    with tf.variable_scope('unfold1_'+str(layer)) as scope:
        
        num_srds    = len(stride)
        hgt_srds    = [srd[0] for srd in stride]
        wdh_srds    = [srd[1] for srd in stride]
        hws_srds    = reduce(lambda x,y: x+y, stride  )
        hgt_srd_all = reduce(lambda x,y: x*y, hgt_srds)
        wdh_srd_all = reduce(lambda x,y: x*y, wdh_srds)
        hws_srd_all = hgt_srd_all * wdh_srd_all
        
        hgt_dims    = [3 + 2 * i for i in range(num_srds)]
        wdh_dims    = [4 + 2 * i for i in range(num_srds)]
        
        new_num     = x_shape[3] // hws_srd_all
        new_hgt     = x_shape[1]  * hgt_srd_all
        new_wdh     = x_shape[2]  * wdh_srd_all
        old_num     = new_num     * hws_srd_all
        
        if old_num != x_shape[3]:
            tensor_in = tensor_in[:, :, :, :old_num, :]
            #x_shape  = get_shape(tensor_in)
        
        tensor_in   = tf.reshape(tensor_in, x_shape[0:3] + hws_srds + [new_num, x_shape[4]])
        tensor_in   = tf.transpose(tensor_in, [0, 1] + hgt_dims + [2] + wdh_dims + [3+2*num_srds, 4+2*num_srds])
        tensor_out  = tf.reshape(tensor_in, [x_shape[0], new_hgt, new_wdh, new_num, x_shape[4]])
        print_activations(tensor_out)
    return tensor_out















def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    num_output = params['group_unit']['num_output']  #[4, 64]
    num_bottle = params['group_unit']['num_bottle']  #[4, 16]
    rate       = params['group_unit']['rate']
    stride     = params['group_unit']['stride']
    use_attn   = params['group_unit']['use_attn']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]      #(None, 256, 256, 16, 64)
    
    if isinstance(rate[0],   int):
        rate      = [rate]
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:
        
        if np.any(np.asarray(stride) > 1):
            params['fold'] = {'stride': stride}
            tensor_in      = fold1(tensor_in, 0, params, mtrain)
        
        params['proj']     = {'number':num_bottle, 'shape':[1,1], 'rate':rate[  0]+[1,1], 'stride':[1,1,1,1], \
                              'padding':'VALID', 'use_bias':False, 'use_attn':False}
        residual           = proj_bn_relu1(tensor_in,  0, params, mtrain)
        
        res_lst            = []
        for i in range(len(rate)-2):
            params['proj'] = {'number':num_bottle, 'shape':[3,3], 'rate':rate[1+i]+[1,1], 'stride':[1,1,1,1], \
                              'padding':'SAME',  'use_bias':False, 'use_attn':use_attn}
            residual       = proj_bn_relu1(residual, 1+i, params, mtrain)
        
        params['proj']     = {'number':num_output, 'shape':[1,1], 'rate':rate[ -1]+[1,1], 'stride':[1,1,1,1], \
                              'padding':'VALID', 'use_bias':False, 'use_attn':False}
        residual           = proj_bn1(residual, 0, params, mtrain)
        tensor_out         = tensor_in + residual
        tensor_out         = relu1(tensor_out,  0, params, mtrain)
    return tensor_out




def proj1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    向量神经元专用，输入形状为[N, H, W, M, C]
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['proj']['number']    #[4, 64]
    shape     = params['proj']['shape']     #[3,  3]
    rate      = params['proj']['rate']      #[b, 2, 1, 1]
    stride    = params['proj']['stride']    #[2, 2, 1, 1]
    padding   = params['proj']['padding']
    use_bias  = params['proj']['use_bias']
    use_attn  = params['proj']['use_attn']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    if use_attn:
        use_bias  = True
    
    with tf.variable_scope('proj1_'+str(layer), reuse=reuse) as scope:
        
        if np.any(np.asarray(stride[0:2]) > 1):
            params['fold']['stride'] = stride[0:2]
            tensor_in = fold1(tensor_in, 0, params, mtrain)
        
        x_shape   = get_shape(tensor_in)                                                        #[None, 256, 256, 16, 64]
        
        #block之内的联系紧密，block之外的联系松散
        tensor_in = tf.reshape(tensor_in, x_shape[:3]+[rate[0],x_shape[3]//rate[0],x_shape[4]]) #(None, 128, 128, 2, 8, 64)
        tensor_in = tf.transpose(tensor_in, [3, 0, 1, 2, 4, 5])                                 #(2, None, 128, 128, 8, 64)
        x_shape   = get_shape(tensor_in)                                                        #[2, None, 128, 128, 8, 64]
        
        #根据通道上的膨胀率，再次对向量神经元进行划分
        tensor_in = tf.reshape(tensor_in, x_shape[:4]+[x_shape[4]//rate[1],rate[1],x_shape[5]]) #(2, None, 128, 128, 4, 2, 64)
        tensor_in = tf.transpose(tensor_in, [0, 5, 1, 2, 3, 4, 6])                              #(2, 2, None, 128, 128, 4, 64)
        
        x_shape   = get_shape(tensor_in)                                                        #[2, 2, None, 128, 128, 4, 64]
        y_shape   = x_shape[0:5] + number                                                       #[2, 2, None, 128, 128, 4, 64]
        
        #reshape以便于进行卷积操作
        tsr_int = tf.reshape(tensor_in, [x_shape[0]*x_shape[1]] + x_shape[2:5] + \
                                        [x_shape[5]*x_shape[6]])                                #(2*2,  None, 128, 128, 4*64)
        tsr_shp = get_shape(tsr_int)                                                            #[2*2,  None, 128, 128, 4*64]
        
        wgt_srd = [1, stride[2], stride[3], 1]                                                  #[1, 1, 1, 1]
        wgt_rat = [1,   rate[2],   rate[3], 1]                                                  #[1, 1, 1, 1]
        wgt_shp = [tsr_shp[0]] + shape + [tsr_shp[4], y_shape[5]*y_shape[6]]                    #[2*2, 3, 3, 4*64, 4*64]
        weights = tf.get_variable(name='weights', shape=wgt_shp, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        if use_bias:
            bia_shp = [tsr_shp[0], 1, 1, 1, y_shape[5]*y_shape[6]]                              #[2*2, 1, 1, 1, 4*64]
            biases  = tf.get_variable(name='biases', shape=bia_shp, dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable)
        
        elems   = [tsr_int, weights]
        tsr_cov = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], wgt_srd, padding=padding, dilations=wgt_rat), \
                            elems, dtype=tf.float32, parallel_iterations=128, \
                            back_prop=True, swap_memory=True, infer_shape=True)                 #(2*2, None, 128, 128, 4*64)
        if use_bias:
            tsr_cov = tsr_cov + biases                                                          #(2*2, None, 128, 128, 4*64)
        
        
        if use_attn:
            elems   = [tsr_int, tsr_cov, weights]
            tsr_att = tf.map_fn(atten, elems, dtype=tf.float32, parallel_iterations=128, \
                                back_prop=True, swap_memory=True, infer_shape=True)             #(2*2, None, 128, 128, 4*64)
            tsr_att = tsr_att + biases
            tsr_out = tsr_cov + tsr_att                                                         #(2*2, None, 128, 128, 4*64)
        else:
            tsr_out = tsr_cov                                                                   #(2*2, None, 128, 128, 4*64)
        
        tensor_out = tf.reshape(tsr_out, y_shape)                                               #(2, 2, None, 128, 128, 4, 64)
        tensor_out = tf.transpose(tensor_out, [2, 3, 4, 0, 5, 1, 6])                            #(None, 128, 128, 2, 4, 2, 64)
        y_shape    = get_shape(tensor_out)                                                      #[None, 128, 128, 2, 4, 2, 64]
        tensor_out = tf.reshape(tensor_out, y_shape[0:3] + \
                                            [y_shape[3]*y_shape[4]*y_shape[5], y_shape[6]])     #(None, 128, 128, 16, 64)
        #tf.summary.histogram('proj', tensor_out)
        print_activations(tensor_out)
    return tensor_out












#############4 neurons#############(128, 128,    4, 64)
        #(128, 128,    4, 64)--[4, 64, 4, 16]--(128, 128,    4, 16) #input   [4, 16]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        '''
        #(128, 128,    4, 16)--[4, 16, 4, 16]--(128, 128,    4, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        '''
        #(128, 128,    4, 16)--[4, 16, 4, 64]--(128, 128,    4, 64) #output  [4, 64]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        
        #############16 neurons############( 64,  64,   16, 64)
        #( 64,  64,   16, 64)--[4, 64, 4, 16]--( 64,  64,   16, 16) #input   [4, 16]  [1, 1, 1, 1]  [  1,  4, 1, 1]
        '''
        #( 64,  64,   16, 16)--[4, 16, 4, 16]--( 64,  64,   16, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1,  4, 1, 1]
        #( 64,  64,   16, 16)--[4, 16, 4, 16]--( 64,  64,   16, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  1, 1, 1]
        '''
        #( 64,  64,   16, 16)--[4, 16, 4, 64]--( 64,  64,   16, 64) #output  [4, 64]  [1, 1, 1, 1]  [  4,  1, 1, 1]
        
        #############64 neurons############( 32,  32,   64, 64)
        #( 32,  32,   64, 64)--[4, 64, 4, 16]--( 32,  32,   64, 16) #input   [4, 16]  [1, 1, 1, 1]  [  1, 16, 1, 1]
        '''
        #( 32,  32,   64, 16)--[4, 16, 4, 16]--( 32,  32,   64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 16, 1, 1]
        #( 32,  32,   64, 16)--[4, 16, 4, 16]--( 32,  32,   64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  4, 1, 1]
        #( 32,  32,   64, 16)--[4, 16, 4, 16]--( 32,  32,   64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  1, 1, 1]
        '''
        #( 32,  32,   64, 16)--[4, 16, 4, 64]--( 32,  32,   64, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 16,  1, 1, 1]
        
        #############256 neurons###########( 16,  16,   256, 64)
        #( 16,  16,  256, 64)--[2, 64, 2, 16]--( 16,  16,  256, 16) #input   [4, 16]  [1, 1, 1, 1]  [  1, 64, 1, 1]
        '''
        #( 16,  16,  256, 16)--[4, 16, 4, 16]--( 16,  16,  256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 64, 1, 1]
        #( 16,  16,  256, 16)--[4, 16, 4, 16]--( 16,  16,  256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4, 16, 1, 1]
        #( 16,  16,  256, 16)--[4, 16, 4, 16]--( 16,  16,  256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  4, 1, 1]
        #( 16,  16,  256, 16)--[4, 16, 4, 16]--( 16,  16,  256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 64,  1, 1, 1]
        '''
        #( 16,  16,  256, 16)--[4, 16, 4, 64]--( 16,  16,  256, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 64,  1, 1, 1]
        
        #############1024 neurons##########(  8,   8, 1024, 64)
        #(  8,   8, 1024, 64)--[4, 16, 4, 64]--(  8,   8, 1024, 16) #input   [4, 16]  [1, 1, 1, 1]  [  1, 256, 1, 1]
        '''
        #(  8,   8, 1024, 64)--[4, 16, 4, 16]--(  8,   8, 1024, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 256, 1, 1]
        #(  8,   8, 1024, 64)--[4, 16, 4, 16]--(  8,   8, 1024, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  64, 1, 1]
        #(  8,   8, 1024, 64)--[4, 16, 4, 16]--(  8,   8, 1024, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  16, 1, 1]
        #(  8,   8, 1024, 64)--[4, 16, 4, 16]--(  8,   8, 1024, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 64,   4, 1, 1]
        #(  8,   8, 1024, 64)--[4, 16, 4, 16]--(  8,   8, 1024, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [256,   1, 1, 1]
        '''
        #(  8,   8, 1024, 16)--[4, 16, 4, 64]--(  8,   8, 1024, 64) #output  [4, 64]  [1, 1, 1, 1]  [256,   1, 1, 1]


























#############4 neurons#############(128, 128,   4, 64)
        #(256, 256,   1, 64)--[1, 64, 1, 64]--(128, 128,   4, 64) #branch  [1, 64]  [2, 2, 1, 1]  [  4,  1, 1, 1]
        #(256, 256,   1, 64)--[1, 64, 1, 16]--(128, 128,   4, 16) #input   [1, 16]  [2, 2, 1, 1]  [  4,  1, 1, 1]
        #(128, 128,   4, 16)--[4, 16, 4, 16]--(128, 128,   4, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        #(128, 128,   4, 16)--[4, 16, 4, 64]--(128, 128,   4, 64) #output  [4, 64]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        
        #(128, 128,   4, 64)--[4, 64, 4, 16]--(128, 128,   4, 16) #input   [4, 16]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        #(128, 128,   4, 16)--[4, 16, 4, 16]--(128, 128,   4, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        #(128, 128,   4, 16)--[4, 16, 4, 64]--(128, 128,   4, 64) #output  [4, 64]  [1, 1, 1, 1]  [  1,  1, 1, 1]
        
        #############16 neurons############( 64,  64,  16, 64)
        #(128, 128,   4, 64)--[2, 64, 2, 64]--( 64,  64,  16, 64) #branch  [2, 64]  [2, 2, 1, 1]  [  8,  1, 1, 1]
        #(128, 128,   4, 64)--[2, 64, 2, 16]--( 64,  64,  16, 16) #input   [2, 16]  [2, 2, 1, 1]  [  8,  1, 1, 1]
        '''
        #( 64,  64,  16, 16)--[4, 16, 4, 16]--( 64,  64,  16, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1,  4, 1, 1]
        #( 64,  64,  16, 16)--[4, 16, 4, 16]--( 64,  64,  16, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  1, 1, 1]
        '''
        #( 64,  64,  16, 16)--[4, 16, 4, 64]--( 64,  64,  16, 64) #output  [4, 64]  [1, 1, 1, 1]  [  4,  1, 1, 1]
        
        #( 64,  64,  16, 64)--[4, 64, 4, 16]--( 64,  64,  16, 16) #input   [4, 16]  [1, 1, 1, 1]  [  4,  1, 1, 1]
        '''
        #( 64,  64,  16, 16)--[4, 16, 4, 16]--( 64,  64,  16, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1,  4, 1, 1]
        #( 64,  64,  16, 16)--[4, 16, 4, 16]--( 64,  64,  16, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  1, 1, 1]
        '''
        #( 64,  64,  16, 16)--[4, 16, 4, 64]--( 64,  64,  16, 64) #output  [4, 64]  [1, 1, 1, 1]  [  4,  1, 1, 1]
        
        #############64 neurons############( 32,  32,  64, 64)
        #( 64,  64,  16, 64)--[2, 64, 2, 64]--( 32,  32,  64, 64) #branch  [2, 64]  [2, 2, 1, 1]  [ 32,  1, 1, 1]
        #( 64,  64,  16, 64)--[2, 64, 2, 16]--( 32,  32,  64, 16) #input   [2, 16]  [2, 2, 1, 1]  [ 32,  1, 1, 1]
        '''
        #( 32,  32,  64, 16)--[4, 16, 4, 16]--( 32,  32,  64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 16, 1, 1]
        #( 32,  32,  64, 16)--[4, 16, 4, 16]--( 32,  32,  64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  4, 1, 1]
        #( 32,  32,  64, 16)--[4, 16, 4, 16]--( 32,  32,  64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  1, 1, 1]
        '''
        #( 32,  32,  64, 16)--[4, 16, 4, 64]--( 32,  32,  64, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 16,  1, 1, 1]
        
        #( 32,  32,  64, 64)--[4, 64, 4, 16]--( 32,  32,  64, 16) #input   [2, 16]  [1, 1, 1, 1]  [ 16,  1, 1, 1]
        '''
        #( 32,  32,  64, 16)--[4, 16, 4, 16]--( 32,  32,  64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 16, 1, 1]
        #( 32,  32,  64, 16)--[4, 16, 4, 16]--( 32,  32,  64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  4, 1, 1]
        #( 32,  32,  64, 16)--[4, 16, 4, 16]--( 32,  32,  64, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  1, 1, 1]
        '''
        #( 32,  32,  64, 16)--[4, 16, 4, 64]--( 32,  32,  64, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 16,  1, 1, 1]
        
        #############256 neurons###########( 16,  16, 256, 64)
        #( 32,  32,  64, 64)--[2, 64, 2, 64]--( 16,  16, 256, 64) #branch  [2, 64]  [2, 2, 1, 1]  [128,  1, 1, 1]
        #( 32,  32,  64, 64)--[2, 64, 2, 16]--( 16,  16, 256, 16) #input   [2, 16]  [2, 2, 1, 1]  [128,  1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 64, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4, 16, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  4, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 64,  1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 64]--( 16,  16, 256, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 64,  1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 64, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4, 16, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  4, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 64,  1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 64]--( 16,  16, 256, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 64,  1, 1, 1]
        
        #############256 neurons###########( 16,  16, 256, 64)
        #( 32,  32,  64, 64)--[2, 64, 2, 64]--( 16,  16, 256, 64) #branch  [2, 64]  [2, 2, 1, 1]  [128,  1, 1, 1]
        #( 32,  32,  64, 64)--[2, 64, 2, 16]--( 16,  16, 256, 16) #input   [2, 16]  [2, 2, 1, 1]  [128,  1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 256, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  64, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  16, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 64,   4, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [256,   1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 64]--( 16,  16, 256, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 64,  1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  1, 256, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [  4,  64, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 16,  16, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [ 64,   4, 1, 1]
        #( 16,  16, 256, 16)--[4, 16, 4, 16]--( 16,  16, 256, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [256,   1, 1, 1]
        '''
        #( 16,  16, 256, 16)--[4, 16, 4, 64]--( 16,  16, 256, 64) #output  [4, 64]  [1, 1, 1, 1]  [ 64,  1, 1, 1]













#############4 neurons#############(128, 128,  4, 64)
        #(256, 256,  1, 64)--[1, 64, 1, 16]--(128, 128,  4, 16) #input   [1, 16]  [2, 2, 1, 1]  [4, 1, 1, 1]
        #(128, 128,  4, 16)--[4, 16, 4, 16]--(128, 128,  4, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [1, 1, 1, 1]
        #(128, 128,  4, 16)--[4, 16, 4, 64]--(128, 128,  4, 64) #output  [1, 64]  [1, 1, 1, 1]  [1, 1, 1, 1]
        
        #(128, 128,  4, 64)--[4, 64, 4, 16]--(128, 128,  4, 16) #input   [1, 16]  [1, 1, 1, 1]  [1, 1, 1, 1]
        #(128, 128,  4, 16)--[4, 16, 4, 16]--(128, 128,  4, 16) #bottle  [4, 16]  [1, 1, 1, 1]  [1, 1, 1, 1]
        #(128, 128,  4, 16)--[4, 16, 4, 64]--(128, 128,  4, 64) #output  [1, 64]  [1, 1, 1, 1]  [1, 1, 1, 1]





        #the first resnet block setting
        #(512, 512,  1, 64)--[2,  2, 1,  1]--(256, 256,  4, 64)
        #(256, 256,  4, 64)--[1, 64, 1, 32]--(256, 256,  4, 32) #branch  [1, 32]  [2, 2, 1, 1]  [4, 1, 1, 1]
        #(256, 256,  4, 64)--[1, 64, 1,  8]--(256, 256,  4,  8) #input   [1,  8]  [1, 1, 1, 1]  [4, 1, 1, 1]
        #(256, 256,  4,  8)--[4,  8, 4,  8]--(256, 256,  4,  8) #bottle  [4,  8]  [1, 1, 1, 1]  [1, 1, 1, 1]
        #(256, 256,  4,  8)--[4,  8, 4, 32]--(256, 256,  4, 32) #output  [4, 32]  [1, 1, 1, 1]  [1, 1, 1, 1]
        
        #(256, 256,  4, 32)--[4, 32, 4,  8]--(256, 256,  4,  8) #input   [1,  8]  [1, 1, 1, 1]  [1, 1, 1, 1]
        #(256, 256,  4,  8)--[4,  8, 4,  8]--(256, 256,  4,  8) #bottle  [4,  8]  [1, 1, 1, 1]  [1, 1, 1, 1]
        #(256, 256,  4,  8)--[4,  8, 4, 32]--(256, 256,  4, 32) #output  [4, 32]  [1, 1, 1, 1]  [1, 1, 1, 1]
        
        #(256, 256,  4, 32)--[2,  2, 1,  1]--(256, 256, 16, 32)
        #(256, 256, 16, 32)--[1, 64, 1, 32]--(256, 256,  4, 32) #branch  [1, 32]  [2, 2, 1, 1]  [4, 1, 1, 1]


        
        

def group_block1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    block_setting = params['group_block']['block_setting']

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    tensor_out     = tensor_in
    out_list       = []
    for i, block in enumerate(block_setting):
        
        depth_output, depth_bottle, shape, stride, rate, use_attn, unit_number, unit_trainable = block
        params['com']['trainable'] = unit_trainable
        
        with tf.variable_scope('group_block1_'+str(layer)+'_'+str(i)) as scope:
            
            for j in range(unit_number):
                if j == 0: #the first unit in the block
                    params['group_unit'] = {'depth_output':depth_output, 'depth_bottle':depth_bottle, 'use_branch':True, \
                                            'use_attn':use_attn, 'shape':shape, 'stride':stride,    'rate':rate}
                else:      #identity mapping
                    params['group_unit'] = {'depth_output':depth_output, 'depth_bottle':depth_bottle, 'use_branch':False, \
                                            'use_attn':use_attn, 'shape':shape, 'stride':[1,1,1,1], 'rate':rate}
                tensor_out = group_unit1(tensor_out, j, params, mtrain)
        out_list.append(tensor_out)
    return out_list










def group1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    length    = params['group']['length']      #向量神经元的长度[输出, 中间]
    number    = params['group']['number']      #向量神经元的个数[输出, 中间]
    shape     = params['group']['shape']       #空间和通道方向操作的向量神经元是哪些[3, 3]
    stride    = params['group']['stride']      #如何把空间特征堆叠到通道方向的 [1, 1] / [2, 2]
    rate      = params['group']['rate']        #空间和通道方向的膨胀比率，防止3*3*C全连接参数太多[b, 1, 1, 1]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)             #[None, 256, 256, 4, 64]
    
    with tf.variable_scope('group1_'+str(layer)) as scope:
        
        
                elif: number != tsr_shp[3] or length != tsr_shp[4]:
            tsr_out = tf.reshape(tsr_int, tsr_shp[0:3]+[number, length])                    #(None, 128, 128, 16, 64)
        else:
            tsr_out = tsr_int
        
        
        def densefc(tsr_int):
            with tf.variable_scope('densefc_'+str(layer)) as scope:

                
                def get_residual_img(tsr_int):
                    tsr_shp = get_shape(tsr_int)                                                    #[2*1, 128, 128, 8, 32]
                    wgt_shp = [tsr_int[0]] + shape +                                        #[16, 1, 1, 64, 64]
                    weights = tf.get_variable(name='weights', shape=wgt_shp, dtype=dtype, \         #(16, 1, 1, 64, 64)
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                              regularizer=tf.contrib.layers.l2_regularizer(reg), trainable=trainable)
                    
                    elems = [tsr_int]

                    def get_residual_blk(elems=None):


            
            
            
            
            
            
            
        

        
        shortcut = project(tensor_in, number[0], length[0], 0)                                      #(None, 128, 128, 16, 32)
        residual = project(tensor_in, number[1], length[1], 1)                                      #(None, 128, 128,  8, 32)
        residual = relu1(residual, 0, params, mtrain)                                               #(None, 128, 128,  8, 32)
        

        
        if x_shape[1]*x_shape[2]*x_shape[5] != number[0]:
            new_num = number[0] // x_shape[1] // x_shape[2]
        
        elif x_shape[6] != 
        
        
        
        
        
        
        wgt_shp   = x_shape[1:3] + shape[1:3] + [x_shape[5], length[2], number, length[2]]          #[4, 1, 3, 3, 4, 32, 4, 32]
        weights   = tf.get_variable(name='weights', shape=wgt_shp, dtype=dtype, \
                                    #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',\
                                                                                               uniform=True), 
                                    #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                    regularizer=tf.contrib.layers.l2_regularizer(reg), trainable=trainable)
        
        
        
            
        if length[2] != length[1]:
            
    
    
    
    
    
    
    shape   = np.asarray(shape)                                   #[3, 3, 4, 4, 4]
    wgt_shp = [[np.prod(shape[])] for i, n in enumerate(number)]
    
    
    
    fet_shp = [[] ]
    
    
    
    
    fet_shp0 = shape                                              #[3, 3, 4, 4, 4]
    wgt_shp0 = fet_shp0[:-1] + [fet_shp0[-1]//4]                  #[3, 3, 16, 1]
    fet_shp1 = fet_shp0[:-1] + [fet_shp0[-1]+wgt_shp0[-1]]        #[3, 3, 16, 5]
    wgt_shp1 = fet_shp1[:-1] + [fet_shp1[-1],  number[-1]]        #[3, 3, 16, 5, 4]
    fet_shp2 = fet_shp1[:-1] + [wgt_shp1[-1]]                     #[3, 3, 16,  4]
    wgt_shp2 = fet_shp2[:-1] + [number[0]]                        #[3, 3, 16, 16]
    fet_shp3 = [wgt_shp2[-1], fet_shp2[-1]]                       #(16, 4)
    
    
    with tf.variable_scope('group1_'+str(layer)) as scope:
        
        
        
        weight0 = tf.get_variable(name='weight0', shape=wgt_shp0, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)  #[3, 3, 16, 1]
        
        weight1 = tf.get_variable(name='weight1', shape=wgt_shp1, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        
        weight2 = tf.get_variable(name='weight2', shape=wgt_shp2, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        
        
        tsr_out = tf.TensorArray(dtype=tf.float32, size=height*width, dynamic_size=False, clear_after_read=True, \
                                 infer_shape=True, element_shape=[depth_input+depth_key], colocate_with_first_write_call=True)
        

        
        
        
        
        
        
        
        
        
        
        if use_branch:
            params['conv'] = {'number':depth_output, 'shape':shape, 'rate':1, 'stride':stride, 'padding':'VALID'}
            shortcut       = conv_bn1(tensor_in, 0, params, mtrain)
        else:
            shortcut       = tensor_in
        params['conv'] = {'number':depth_bottle, 'shape':shape,  'rate':1,    'stride':stride, 'padding':'VALID'}
        residual       = conv_bn_relu1(tensor_in, 0, params, mtrain)
        
        
        params['conv'] = {'number':depth_bottle, 'shape':[3, 3], 'rate':rate, 'stride':[1, 1], 'padding':'SAME' }
        residual       = conv_bn_relu1(residual,  1, params, mtrain)
        
        
        params['conv'] = {'number':depth_output, 'shape':[1, 1], 'rate':1,    'stride':[1, 1], 'padding':'VALID'}
        residual       = conv_bn1(residual, 1, params, mtrain)
        tensor_out     = relu1(shortcut+residual, 0, params, mtrain)
    return tensor_out










def proj1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    用于向量神经元的升维和降维，输入形状为[n, h, w, m, c]
    每个向量神经元肯定和自己是最相关的，所以对输入修改得越少越好，有点类似深度可分离卷积
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['proj']['number']    #[4,64]
    shape     = params['proj']['shape']     #[3, 3]
    rate      = params['proj']['rate']      #[b, 2, 1, 1]
    stride    = params['proj']['stride']    #[2, 2, 1, 1]
    padding   = params['proj']['padding']
    use_bias  = params['proj']['use_bias']
    use_attn  = params['proj']['use_attn']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)                                                      #[None, 256, 256, 16, 64]
    '''
    if number[0] == -1:
        number[0] = x_shape[3]
    if length[0] == -1:
        length[0] = x_shape[4]
    
    num_kep = x_shape[3] * x_shape[4] // length[0] // number[0]
    wgt_shp = [num_kep, shape[0], shape[1], number[0]*length[0], number[1]*length[1]]   #[16, 1, 1, 1*64, 1*64]
    bia_shp = [num_kep, 1, 1, 1, number[1]*length[1]]                                   #[16, 1, 1, 1, 1*64]
    wgt_srd = [1, stride[2], stride[3], 1]
    wgt_rat = [1,   rate[2],   rate[3], 1]
    '''
    with tf.variable_scope('proj1_'+str(layer), reuse=reuse) as scope:
        
        if np.any(stride[0:2]) > 1:
            tensor_in = tf.reshape(tensor_in, [x_shape[0], x_shape[1]//stride[0], stride[0], \
                                               x_shape[2]//stride[1], stride[1]]+x_shape[3:])   #(None, 128, 2, 128, 2, 4, 64)
            tensor_in = tf.transpose(tensor_in, [0, 1, 3, 2, 4, 5, 6])                          #(None, 128, 128, 2, 2, 4, 64)
            tensor_in = tf.reshape(tensor_in, [x_shape[0], x_shape[1]//stride[0], x_shape[2]//stride[1], \
                                               stride[0]*stride[1]*x_shape[3], x_shape[4]])     #(None, 128, 128, 16, 64)
            x_shape   = get_shape(tensor_in)                                                    #[None, 128, 128, 16, 64]
        
        #block之内的联系紧密，block之外的联系松散
        tensor_in = tf.reshape(tensor_in, x_shape[:3]+[rate[0],x_shape[3]//rate[0],x_shape[4]]) #(None, 128, 128, 2, 8, 64)
        tensor_in = tf.transpose(tensor_in, [3, 0, 1, 2, 4, 5])                                 #(2, None, 128, 128, 8, 64)
        x_shape   = get_shape(tensor_in)                                                        #[2, None, 128, 128, 8, 64]
        
        #根据通道上的膨胀率，再次对向量神经元进行划分
        tensor_in = tf.reshape(tensor_in, x_shape[:4]+[x_shape[4]//rate[1],rate[1],x_shape[5]]) #(2, None, 128, 128, 4, 2, 64)
        tensor_in = tf.transpose(tensor_in, [0, 5, 1, 2, 3, 4, 6])                              #(2, 2, None, 128, 128, 4, 64)
        x_shape   = get_shape(tensor_in)                                                        #[2, 2, None, 128, 128, 4, 64]
        tensor_in = tf.reshape(tensor_in, [rate[0]*rate[1]]+x_shape[2:])                        #(2*2,  None, 128, 128, 4, 64)
        x_shape   = get_shape(tensor_in)                                                        #[2*2,  None, 128, 128, 4, 64]
        
        wgt_shp = [x_shape[0]] + shape + x_shape[4:] + number                                   #[2*2, 3, 3, 4, 64, 4, 64]
        bia_shp = [x_shape[0]] + shape + x_shape[4:] + number                                   #[2*2, 3, 3, 4, 64, 4, 64]
        weights = tf.get_variable(name='weights', shape=wgt_shp, dtype=dtype, \                 #(2*2, 3, 3, 4, 64, 4, 64)
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name='biases', shape=bia_shp, dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        
        
        
        if use_attn:
            
        
        
        tensor_in = tf.reshape(tensor_in, x_shape[0:3]+[num_kep,number[0]*length[0]])   #(None, 128, 128, 16, 1*64)
        tensor_in = tf.transpose(tensor_in, [3, 0, 1, 2, 4])                            #(16, None, 128, 128, 1*64)
        

        
        elems = [tensor_in, kernel]
        proj  = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], wgt_srd, padding=padding, dilations=wgt_rat), \
                          elems, dtype=tf.float32, parallel_iterations=128, \
                          back_prop=True, swap_memory=True, infer_shape=True)           #(16, None, 128, 128, 1*64)
        
        if use_bias:
            tensor_out = proj + biases                                                  #(16, None, 128, 128, 1*64)
        else:
            tensor_out = proj                                                           #(16, None, 128, 128, 1*64)
        tensor_out = tf.transpose(tensor_out, [1, 2, 3, 0, 4])                          #(None, 128, 128, 16, 1*64)
        tensor_out = tf.reshape(tensor_out, x_shape[0:3]+[num_kep*number[1],length[1]]) #(None, 128, 128, 16*1, 64)
        #tf.summary.histogram('proj', tensor_out)
        print_activations(tensor_out)
    return tensor_out
    










def proj2(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    用于向量神经元的升维和降维，输入形状为[n, h, w, m, c]
    每个向量神经元肯定和自己是最相关的，所以对输入修改得越少越好，有点类似深度可分离卷积
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['proj']['number']    #[输入, 输出]
    length    = params['proj']['length']    #[输入, 输出]
    shape     = params['proj']['shape']
    rate      = params['proj']['rate']
    stride    = params['proj']['stride']
    padding   = params['proj']['padding']
    use_bias  = params['proj']['use_bias']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)                                                      #[None, 128, 128, 16, 64]
    
    if number[0] == -1:
        number[0] = x_shape[3]
    if length[0] == -1:
        length[0] = x_shape[4]
    
    num_kep = x_shape[3] * x_shape[4] // length[0] // number[0]
    wgt_shp = [num_kep, shape[0], shape[1], number[0]*length[0], number[1]*length[1]]   #[16, 1, 1, 1*64, 1*64]
    bia_shp = [num_kep, 1, 1, 1, number[1]*length[1]]                                   #[16, 1, 1, 1, 1*64]
    stride  = [1, stride[0], stride[1], 1]
    rate    = [1,   rate[0],   rate[1], 1]
    
    with tf.variable_scope('proj2_'+str(layer), reuse=reuse) as scope:
        
        tensor_in = tf.reshape(tensor_in, x_shape[0:3]+[num_kep,number[0]*length[0]])   #(None, 128, 128, 16, 1*64)
        tensor_in = tf.transpose(tensor_in, [3, 0, 1, 2, 4])                            #(16, None, 128, 128, 1*64)
        
        kernel = tf.get_variable(name='weights', shape=wgt_shp, dtype=dtype, \          #(16, 1, 1, 1*64, 1*64)
                                 #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                 #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                 regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                 trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name='biases', shape=bia_shp, dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        
        elems = [tensor_in, kernel]
        proj  = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], stride, padding=padding, dilations=rate), \
                          elems, dtype=tf.float32, parallel_iterations=128, \
                          back_prop=True, swap_memory=True, infer_shape=True)           #(16, None, 128, 128, 1*64)
        
        if use_bias:
            tensor_out = proj + biases                                                  #(16, None, 128, 128, 1*64)
        else:
            tensor_out = proj                                                           #(16, None, 128, 128, 1*64)
        tensor_out = tf.transpose(tensor_out, [1, 2, 3, 0, 4])                          #(None, 128, 128, 16, 1*64)
        tensor_out = tf.reshape(tensor_out, x_shape[0:3]+[num_kep*number[1],length[1]]) #(None, 128, 128, 16*1, 64)
        #tf.summary.histogram('proj', tensor_out)
        print_activations(tensor_out)
    return tensor_out















#获得残差特征
        residual  = tf.reshape(tensor_in, [x_shape[0]*x_shape[1]]+x_shape[2:5]+[x_shape[5]*x_shape[6]]) #[4*1, None, 128, 128, 4*64]
        r_shape   = get_shape(residual)                                                             #[4*1, None, 128, 128, 4*64]
        leh_out   = number // x_shape[0] // x_shape[1] * length[1]                                  # 4*64
        wgt_shp0  = [r_shape[0], 1, 1, r_shape[-1], leh_out]                                        #[4*1, 1, 1, 4*64, 4*64]
        weights0  = tf.get_variable(name='weights0', shape=wgt_shp0, dtype=dtype, \
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                    regularizer=tf.contrib.layers.l2_regularizer(reg), trainable=trainable)
        def get_residual(elems):
            residual, weights0 = elems
            residual = tf.nn.conv2d(residual, weights0, [1, 1, 1, 1], padding='VALID', dilations=[1, 1, 1, 1])
            residual = bn_relu1(residual, 0, params, mtrain)
            return residual
        elems     = [residual, weights0]
        residual  = tf.map_fn(get_residual, elems, dtype=tf.float32, parallel_iterations=128, \
                              back_prop=True, swap_memory=True, infer_shape=True)                   #(4*1, None, 128, 128, 4*64)
        r_shape   = get_shape(residual)                                                             #[4*1, None, 128, 128, 4*64]
        residual  = tf.reshape(residual, r_shape[:-1]+[r_shape[-1]//length[1], length[1]])          #(4*1, None, 128, 128, 4, 64)
        r_shape   = get_shape(residual)                                                             #[4*1, None, 128, 128, 4, 64]
        
        #获得旁路特征






tensor_out = tf.layers.conv2d(inputs=tensor_in, filters=number, kernel_size=shape, strides=stride, \
                                  padding=padding, data_format='channels_last', dilation_rate=rate, \
                                  activation=None, use_bias=use_bias, \
                                  kernel_initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  bias_initializer=tf.zeros_initializer(), \
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  bias_regularizer=None, activity_regularizer=None, \
                                  kernel_constraint=None, bias_constraint=None, \
                                  trainable=trainable, reuse=reuse)


def group1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    number    = params['group']['number']                         #[4, 4, 4]
    shape     = params['group']['shape']                          #[3, 3, 4, 4, 4]
    stride    = params['group']['stride']                         #[1, 1] / [2, 2]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)                                #[None, 256, 256, 64]
    shape   = np.asarray(shape)                                   #[3, 3, 4, 4, 4]
    wgt_shp = [[np.prod(shape[])] for i, n in enumerate(number)]
    
    
    
    fet_shp = [[] ]
    
    
    
    
    fet_shp0 = shape                                              #[3, 3, 4, 4, 4]
    wgt_shp0 = fet_shp0[:-1] + [fet_shp0[-1]//4]                  #[3, 3, 16, 1]
    fet_shp1 = fet_shp0[:-1] + [fet_shp0[-1]+wgt_shp0[-1]]        #[3, 3, 16, 5]
    wgt_shp1 = fet_shp1[:-1] + [fet_shp1[-1],  number[-1]]        #[3, 3, 16, 5, 4]
    fet_shp2 = fet_shp1[:-1] + [wgt_shp1[-1]]                     #[3, 3, 16,  4]
    wgt_shp2 = fet_shp2[:-1] + [number[0]]                        #[3, 3, 16, 16]
    fet_shp3 = [wgt_shp2[-1], fet_shp2[-1]]                       #(16, 4)
    
    
    with tf.variable_scope('group1_'+str(layer)) as scope:
        
        
        
        weight0 = tf.get_variable(name='weight0', shape=wgt_shp0, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)  #[3, 3, 16, 1]
        
        weight1 = tf.get_variable(name='weight1', shape=wgt_shp1, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        
        weight2 = tf.get_variable(name='weight2', shape=wgt_shp2, dtype=dtype, \
                                  #initializer=tf.initializers.truncated_normal(stddev=wscale), \
                                  initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32), \
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        
        
        tsr_out = tf.TensorArray(dtype=tf.float32, size=height*width, dynamic_size=False, clear_after_read=True, \
                                 infer_shape=True, element_shape=[depth_input+depth_key], colocate_with_first_write_call=True)
        
        def group_img(elems=None):
            
            tsr_int = elems  #(H, W, C)
            
            def cond(i, tsr_out):
                c = tf.less(i, x_shape[1]*x_shape[2])
                return c

            def body(i, tsr_out):

                ycd   = i // x_shape[2]
                xcd   = i  % x_shape[2]
                ymn   = ycd - (shape[0] - 1) // 2
                xmn   = xcd - (shape[1] - 1) // 2
                ycds  = tf.concat([[ymn], tf.tile([1], [shape[0]-1])], axis=0)
                xcds  = tf.concat([[xmn], tf.tile([1], [shape[1]-1])], axis=0)
                ycds  = tf.cumsum(ycds, axis=0, exclusive=False, reverse=False)
                xcds  = tf.cumsum(xcds, axis=0, exclusive=False, reverse=False)
                yixs  = tf.where(tf.logical_and(ycds>=0, ycds<x_shape[1]))[:, 0]
                ycds  = tf.gather(ycds, yixs)
                xixs  = tf.where(tf.logical_and(xcds>=0, xcds<x_shape[2]))[:, 0]
                xcds  = tf.gather(xcds, xixs)
                ycds  = tf.tile(ycds[:, tf.newaxis], [1, tf.shape(xcds)[0]])         #(3, 3)
                xcds  = tf.tile(xcds[tf.newaxis, :], [tf.shape(ycds)[0], 1])         #(3, 3)
                crds  = tf.concat([ycds, xcds], axis=-1)                             #(3, 3, 2)
                yixs  = tf.tile(yixs[:, tf.newaxis], [1, tf.shape(xixs)[0]])         #(3, 3)
                xixs  = tf.tile(xixs[tf.newaxis, :], [tf.shape(yixs)[0], 1])         #(3, 3)
                idxs  = tf.concat([yixs, xixs], axis=-1)                             #(3, 3, 2)
                fet0  = tf.gather_nd(tsr_int, crds)                                  #(3, 3, 64)
                fet0  = tf.reshape(fet0, fet_shp0)                                   #(3, 3, 16, 4)
                
                
                
                shp0  = get_shape(fet0)[:-1] + fet_shp0[2:]                          #[3, 3, 16, 4]
                fet0  = tf.reshape(fet0, shp0)                                       #(3, 3, 16, 4)
                wgt0  = tf.gather_nd(weight0, idxs)                                  #(3, 3, 16, 1)
                fet1  = tf.concat([fet0, wgt0])                                      #(3, 3, 16, 5)
                fet1  = tf.expand_dims(fet1, axis=-2)                                #(3, 3, 16, 1, 5)
                wgt1  = tf.gather_nd(weight1, idxs)                                  #(3, 3, 16, 5, 4)
                fet2  = tf.matmul(fet1, wgt1)                                        #(3, 3, 16, 1, 4)
                fet2  = tf.nn.relu(fet2)                                             #(3, 3, 16, 1, 4)
                fet2  = tf.transpose(fet2, perm=[4, 3, 0, 1, 2])                     #(4, 1, 3, 3, 16)
                shp2  = get_shape(fet2)                                              #[4, 1, 3, 3, 16]
                fet2  = tf.reshape(fet2, shp2[0:2]+[-1])                             #(4, 1, 3*3*16)
                
                
                wgt2  = tf.gather_nd(weight2, idxs)                                  #(3, 3, 4, 16, 16)
                fet3  = tf.matmul(fet2, wgt2)                                        #(3, 3, 4,  1, 16)
                fet3  = tf.nn.relu(fet3)                                             #(3, 3, 4,  1, 16)
                
                
                
                crd0  = tf.stack([ycd, xcd], axis=0)                                 #(2)         实际中心
                crds0 = tf.concat([ycds, xcds], axis=-1)                             #(h, w, 2)   实际坐标
                fets0 = tf.gather_nd(tensor_value, crds0)                            #(h, w, c)   实际特征
                fets3 = tf.gather_nd(tensor_key,   crds0)                            #(h, w, c')  实际特征
                crd1  = (shape - 1) // 2                                             #(2)         相对中心
                crds1 = (crds0 - crd0) // rate                                       #(h, w, 2)   相对坐标
                crds1 = crds1 + crd1                                                 #(h, w, 2)   相对坐标
                fets1 = tf.gather_nd(PE, crds1)                                      #(h, w, c)   相对特征
                #fets2= tf.concat([fets0, fets1], axis=-1)                           #(h, w, c'') 融合特征
                crd3  = crd0 - crds0[0, 0]         #crd、crds下标换成1也一样           #(2)         相对坐标
                fet3  = tf.gather_nd(fets3, crd3)                                    #(c')        相对中心
                #计算注意力
                att3  = tf.einsum('ijk,k->ij', fets3, fet3)                          #(h, w)
                att3  = tf.exp(att3 / tf.sqrt(depth_key))                            #(h, w)
                att3  = att3 / tf.reduce_sum(att3)                                   #(h, w)
                fet0  = tf.einsum('ij,ijk->k', att3, fets0)                          #(c)
                fet1  = tf.einsum('ij,ijk->k', att3, fets1)                          #(c')
                fet2  = tf.concat([fet0, fet1], axis=-1)                             #(c'')
                #fet2 = tf.einsum('ij,ijk->k', att3, fets2)                          #(c'')
                tsr_out = tsr_out.write(i, fet2)                               #(h, w, c')
                return [i+1, tsr_out]

            i = tf.constant(0)
            [i, tsr_out] = tf.while_loop(cond, body, loop_vars=[i, tsr_out], shape_invariants=None, \
                                         parallel_iterations=128, back_prop=True, swap_memory=True)
        
        
        
        
        
        
        
        
        
        
        if use_branch:
            params['conv'] = {'number':depth_output, 'shape':shape, 'rate':1, 'stride':stride, 'padding':'VALID'}
            shortcut       = conv_bn1(tensor_in, 0, params, mtrain)
        else:
            shortcut       = tensor_in
        params['conv'] = {'number':depth_bottle, 'shape':shape,  'rate':1,    'stride':stride, 'padding':'VALID'}
        residual       = conv_bn_relu1(tensor_in, 0, params, mtrain)
        
        
        params['conv'] = {'number':depth_bottle, 'shape':[3, 3], 'rate':rate, 'stride':[1, 1], 'padding':'SAME' }
        residual       = conv_bn_relu1(residual,  1, params, mtrain)
        
        
        params['conv'] = {'number':depth_output, 'shape':[1, 1], 'rate':1,    'stride':[1, 1], 'padding':'VALID'}
        residual       = conv_bn1(residual, 1, params, mtrain)
        tensor_out     = relu1(shortcut+residual, 0, params, mtrain)
    return tensor_out











def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    depth_output = params['group_unit']['depth_output']
    depth_bottle = params['group_unit']['depth_bottle']
    use_branch   = params['group_unit']['use_branch']
    shape        = params['group_unit']['shape']        #LSTM和Attenion的关联形状
    stride       = params['group_unit']['stride']       #是如何把空间特征堆叠到通道方向的
    rate         = params['group_unit']['rate']         #提取抽象形状特征时，卷积的膨胀率，防止1*1*C全连接参数太多
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape = tensor_in.get_shape().as_list()
    x_shape  = get_shape(tensor_in)
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:

        if use_branch:
            params['conv'] = {'number':depth_output, 'shape':shape, 'rate':1, 'stride':stride, 'padding':'VALID'}
            shortcut       = conv_bn1(tensor_in, 0, params, mtrain)
        else:
            shortcut       = tensor_in
        params['conv'] = {'number':depth_bottle, 'shape':shape,  'rate':1,    'stride':stride, 'padding':'VALID'}
        residual       = conv_bn_relu1(tensor_in, 0, params, mtrain)
        
        
        params['conv'] = {'number':depth_bottle, 'shape':[3, 3], 'rate':rate, 'stride':[1, 1], 'padding':'SAME' }
        residual       = conv_bn_relu1(residual,  1, params, mtrain)
        
        
        params['conv'] = {'number':depth_output, 'shape':[1, 1], 'rate':1,    'stride':[1, 1], 'padding':'VALID'}
        residual       = conv_bn1(residual, 1, params, mtrain)
        tensor_out     = relu1(shortcut+residual, 0, params, mtrain)
    return tensor_out








def group_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    number   = params['group']['number']
    multiple = params['group']['multiple']
    shape    = params['group']['shape']
    rate     = params['group']['rate']
    stride   = params['group']['stride']
    padding  = params['group']['padding']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope('group_bn_relu1_'+str(layer)) as scope:
        params['conv'] = {'number':number,  'shape':[3, 3],'rate':[1, 1],'stride':[1, 1],'padding':'SAME' }
        tensor_out     = conv_bn_relu1(tensor_in,  0, params, mtrain)
        params['conv'] = {'number':multiple,'shape':shape, 'rate':rate,  'stride':stride,'padding':padding}
        tensor_out     = conv_bn_relu2(tensor_out, 0, params, mtrain)
    return tensor_out





        if use_attn:
            gamma = tf.get_variable(name='gamma', shape=[len(shape)], dtype=dtype, \
                                    initializer=tf.constant_initializer(0.0), trainable=trainable)

            if use_attn:
                params['affine'] = {'dim': num_int[l], 'use_bias': True}
                tsr_qry = affine1(tsr_out, 3*l+0, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_key = affine1(tsr_out, 3*l+1, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_vau = affine1(tsr_out, 3*l+2, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_qry = tf.reshape(tsr_qry, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c)
                tsr_key = tf.reshape(tsr_key, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c)
                tsr_vau = tf.reshape(tsr_vau, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c   )
                qry_key = tf.matmul(tsr_qry, tsr_key, transpose_b=True)           #(N, H'*W'*C', H'*W'*C')
                qry_key = tf.nn.softmax(qry_key, axis=-1)                         #(N, H'*W'*C', H'*W'*C')
                tsr_att = tf.matmul(qry_key, tsr_vau)                             #(N, H'*W'*C', h*w*c)
                tsr_att = tf.reshape(tsr_att, [-1, num_int[l]])                   #(N*H'*W'*C', h*w*c)
                tsr_out = tsr_out + tsr_att * gamma[l]                            #(N*H'*W'*C', h*w*c)

        if use_prev and np.all(np.asarray(get_shape(tensor_out)==np.asarray(get_shape(tensor_in)))):
            tensor_out = tensor_out + tensor_in

            
            
def group1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    1.保证∏(i=1->n)xi = d(d为原本的参数维度，在这里为特征图的体积H*W*C)，从而进行分组，以进行局部全连接
    2.参数复杂度为∑(i=1->n)(xi)^2
    3.batchnorm只针对/除滤波器输出的维度外/的维度做，滤波器输出的特征为本层看重的特征
    4.本函数包含针对主干网络的注意力机制，以聚集相似特征，使滤波器在较小体积的情况下依然能够尽可能地把相似特征映射为差异特征
      这样不仅提高了网络的抽象能力和表达能力，还起到进一步增大网络感受野的作用。
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    shape     = params['group']['shape']     #[[[7, 4, 1], [7, 4, 1]], [[4, 7, 1], [4, 7, 1]]]
    use_attn  = params['group']['use_attn']
    use_prev  = params['group']['use_prev']
    use_drop  = params['group']['use_drop']
    keep_p    = params['group']['keep_p']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    tsr_out = tensor_in
    shape   = np.asarray(shape)
    num_int = np.prod(shape[:, 0], axis=1)                                       #[h1*w1*c1, h2*w2*c2]
    num_out = np.prod(shape[:, 1], axis=1)                                       #[h'*w'*c', h"*w"*c"]
    
    with tf.variable_scope('group1_'+str(layer), reuse=reuse) as scope:
        
        if use_attn:
            gamma = tf.get_variable(name='gamma', shape=[len(shape)], dtype=dtype, \
                                    initializer=tf.constant_initializer(0.0), trainable=trainable)
        
        for l in range(len(shape)):
            
            shp_int = get_shape(tsr_out)                                          #[N, H, W, C]
            bat_siz = shp_int[0]                                                  # N
            shp_int = np.asarray(shp_int[1:])                                     #[H, W, C]
            shp_out = shp_int // shape[l][0]                                      #[H', W', C']
            shp_int = np.stack([shp_out, shape[l][0]], axis=-1)                   #[[H', h], [W', w], [C', c]]
            shp_int = np.reshape(shp_int, -1)                                     #[H', h, W', w, C', c]
            shp_int = [bat_siz] + list(shp_int)                                   #[N, H', h, W', w, C', c]
            shp_idx = np.arange(1, len(shp_int)).reshape([-1, 2])                 #[[1, 2], [3, 4], [5, 6]]
            shp_idx = np.transpose(shp_idx).reshape(-1)                           #[1, 3, 5, 2, 4, 6]
            shp_idx = [ 0] + list(shp_idx)                                        #[0, 1, 3, 5, 2, 4, 6]
            shp_out = [-1] + list(shp_out)                                        #[-1, H', W', C']
            
            tsr_out = tf.reshape(tsr_out,   shp_int)                              #(N, H', h, W', w, C', c)
            tsr_out = tf.transpose(tsr_out, shp_idx)                              #(N, H', W', C', h, w, c)
            tsr_out = tf.reshape(tsr_out, [-1, num_int[l]])                       #(N*H'*W'*C', h*w*c)
            
            if use_attn:
                params['affine'] = {'dim': num_int[l], 'use_bias': True}
                tsr_qry = affine1(tsr_out, 3*l+0, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_key = affine1(tsr_out, 3*l+1, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_vau = affine1(tsr_out, 3*l+2, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_qry = tf.reshape(tsr_qry, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c)
                tsr_key = tf.reshape(tsr_key, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c)
                tsr_vau = tf.reshape(tsr_vau, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c   )
                qry_key = tf.matmul(tsr_qry, tsr_key, transpose_b=True)           #(N, H'*W'*C', H'*W'*C')
                qry_key = tf.nn.softmax(qry_key, axis=-1)                         #(N, H'*W'*C', H'*W'*C')
                tsr_att = tf.matmul(qry_key, tsr_vau)                             #(N, H'*W'*C', h*w*c)
                tsr_att = tf.reshape(tsr_att, [-1, num_int[l]])                   #(N*H'*W'*C', h*w*c)
                tsr_out = tsr_out + tsr_att * gamma[l]                            #(N*H'*W'*C', h*w*c)
                
            params['affine'] = {'dim': num_out[l], 'use_bias': False}
            tsr_out = affine_bn_relu1(tsr_out, l, params, mtrain)                 #(N*H'*W'*C', h'*w'*c')
            
            if use_drop:
                params['dropout'] = {'keep_p': keep_p, 'shape': None}
                tsr_out = dropout1(tsr_out, l, params, mtrain)                    #(N*H'*W'*C', h'*w'*c')
                
            tsr_out = tf.reshape(tsr_out, [bat_siz, -1, num_out[l]])              #(N, H'*W'*C', h'*w'*c')
            tsr_out = tf.transpose(tsr_out, [0, 2, 1])                            #(N, h'*w'*c', H'*W'*C')
            tsr_out = tf.reshape(tsr_out, shp_out)                                #(N*h'*w'*c', H', W', C')
        '''
        prt_opa = tf.print([gamma])
        with tf.control_dependencies([prt_opa]):
            tsr_out = tf.identity(tsr_out)
        '''
        #tsr_out-->(N*h'*w'*c'*h"*w"*c", H', W', C')
        shp_int  = get_shape(tensor_in)                                           #[N, H, W, C]
        bat_siz  = shp_int[0]                                                     # N
        shp_int  = np.asarray(shp_int[1:])                                        #[H, W, C]
        shape    = np.concatenate([shape[:, 1], [shp_out[1:]]], axis=0)           #[[h', w', c'], [h", w", c"], [H', W', C']]
        shp_out0 = [bat_siz] + list(np.reshape(shape,  -1))                       #[N, h', w', c', h", w", c", H', W', C']
        shp_out1 = [bat_siz] + list(np.prod(shape, axis=0))                       #[N, h'*h"*H', w'*w"*W', c'*c"*C']
        
        shp_idx  = np.arange(1, len(shp_out0))                                    #[1, 2, 3, 4, 5, 6, 7, 8, 9]
        shp_idx  = np.reshape(shp_idx, shape.shape)                               #[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        shp_idx  = np.transpose(shp_idx)                                          #[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        shp_idx  = shp_idx[:, ::-1].reshape(-1)                                   #[7, 4, 1, 8, 5, 2, 9, 6, 3]
        shp_idx  = [0] + list(shp_idx)                                            #[0, 7, 4, 1, 8, 5, 2, 9, 6, 3]
        
        tsr_out  = tf.reshape  (tsr_out, shp_out0)                                #(N, h', w', c', h", w", c", H', W', C')
        tsr_out  = tf.transpose(tsr_out, shp_idx )                                #(N, H', h", h', W', w", w', C', c", c')
        tensor_out = tf.reshape(tsr_out, shp_out1)                                #(N, H'*h"*h', W'*w"*w', C'*c"*c')
        
        if use_prev and np.all(np.asarray(get_shape(tensor_out)==np.asarray(get_shape(tensor_in)))):
            tensor_out = tensor_out + tensor_in
        print_activations(tensor_out)
    return tensor_out            




def group1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    1.保证∏(i=1->n)xi = d(d为原本的参数维度，在这里为特征图的体积H*W*C)，从而进行分组，以进行局部全连接
    2.参数复杂度为∑(i=1->n)(xi)^2
    3.batchnorm只针对/除滤波器输出的维度外/的维度做，滤波器输出的特征为本层看重的特征
    4.本函数包含针对主干网络的注意力机制，以聚集相似特征，使滤波器在较小体积的情况下依然能够尽可能地把相似特征映射为差异特征
      这样不仅提高了网络的抽象能力和表达能力，还起到进一步增大网络感受野的作用。
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    shape     = params['group']['shape']     #[[[7, 4, 1], [7, 4, 1]], [[4, 7, 1], [4, 7, 1]]]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    tsr_out = tensor_in
    shape   = np.asarray(shape)
    num_int = np.prod(shape[:, 0], axis=1)                                       #[h1*w1*c1, h2*w2*c2]
    num_out = np.prod(shape[:, 1], axis=1)                                       #[h'*w'*c', h"*w"*c"]
    
    with tf.variable_scope('group1_'+str(layer), reuse=reuse) as scope:
        
        for l in range(len(shape)):
            
            shp_int = get_shape(tsr_out)                                          #[N, H, W, C]
            bat_siz = shp_int[0]                                                  # N
            shp_int = np.asarray(shp_int[1:])                                     #[H, W, C]
            shp_out = shp_int // shape[l][0]                                      #[H', W', C']
            shp_int = np.stack([shp_out, shape[l][0]], axis=-1)                   #[[H', h], [W', w], [C', c]]
            shp_int = np.reshape(shp_int, -1)                                     #[H', h, W', w, C', c]
            shp_int = [bat_siz] + list(shp_int)                                   #[N, H', h, W', w, C', c]
            shp_idx = np.arange(1, len(shp_int)).reshape([-1, 2])                 #[[1, 2], [3, 4], [5, 6]]
            shp_idx = np.transpose(shp_idx).reshape(-1)                           #[1, 3, 5, 2, 4, 6]
            shp_idx = [ 0] + list(shp_idx)                                        #[0, 1, 3, 5, 2, 4, 6]
            shp_out = [-1] + list(shp_out)                                        #[-1, H', W', C']
            
            tsr_out = tf.reshape(tsr_out,   shp_int)                              #(N, H', h, W', w, C', c)
            tsr_out = tf.transpose(tsr_out, shp_idx)                              #(N, H', W', C', h, w, c)
            tsr_out = tf.reshape(tsr_out, [-1, num_int[l]])                       #(N*H'*W'*C', h*w*c)
            
            params['affine'] = {'dim': num_out[l], 'use_bias': False}
            tsr_out = affine_bn_relu1(tsr_out, l, params, mtrain)                 #(N*H'*W'*C', h'*w'*c')
            
            tsr_out = tf.reshape(tsr_out, [bat_siz, -1, num_out[l]])              #(N, H'*W'*C', h'*w'*c')
            tsr_out = tf.transpose(tsr_out, [0, 2, 1])                            #(N, h'*w'*c', H'*W'*C')
            tsr_out = tf.reshape(tsr_out, shp_out)                                #(N*h'*w'*c', H', W', C')
        '''
        prt_opa = tf.print([gamma])
        with tf.control_dependencies([prt_opa]):
            tsr_out = tf.identity(tsr_out)
        '''
        #tsr_out-->(N*h'*w'*c'*h"*w"*c", H', W', C')
        shp_int  = get_shape(tensor_in)                                           #[N, H, W, C]
        bat_siz  = shp_int[0]                                                     # N
        shp_int  = np.asarray(shp_int[1:])                                        #[H, W, C]
        shape    = np.concatenate([shape[:, 1], [shp_out[1:]]], axis=0)           #[[h', w', c'], [h", w", c"], [H', W', C']]
        shp_out0 = [bat_siz] + list(np.reshape(shape,  -1))                       #[N, h', w', c', h", w", c", H', W', C']
        shp_out1 = [bat_siz] + list(np.prod(shape, axis=0))                       #[N, h'*h"*H', w'*w"*W', c'*c"*C']
        
        shp_idx  = np.arange(1, len(shp_out0))                                    #[1, 2, 3, 4, 5, 6, 7, 8, 9]
        shp_idx  = np.reshape(shp_idx, shape.shape)                               #[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        shp_idx  = np.transpose(shp_idx)                                          #[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        shp_idx  = shp_idx[:, ::-1].reshape(-1)                                   #[7, 4, 1, 8, 5, 2, 9, 6, 3]
        shp_idx  = [0] + list(shp_idx)                                            #[0, 7, 4, 1, 8, 5, 2, 9, 6, 3]
        
        tsr_out  = tf.reshape  (tsr_out, shp_out0)                                #(N, h', w', c', h", w", c", H', W', C')
        tsr_out  = tf.transpose(tsr_out, shp_idx )                                #(N, H', h", h', W', w", w', C', c", c')
        tensor_out = tf.reshape(tsr_out, shp_out1)                                #(N, H'*h"*h', W'*w"*w', C'*c"*c')
        print_activations(tensor_out)
    return tensor_out


'''
def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    output_shape = params['group_unit']['output_shape'] #(H', W', C')
    bottle_shape = params['group_unit']['bottle_shape'] #(H", W", C")
    filter_shape = params['group_unit']['filter_shape'] #(h, w, c)
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape = tensor_in.get_shape().as_list() #(N, H, W, C)
    x_shape = get_shape(tensor_in)            #(N, H, W, C)
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:

        if np.any(np.asarray(x_shape[1:]) != np.asarray(output_shape)): #深度可分离卷积!!!
            number         = output_shape[-1] // x_shape[-1]
            shape          = [3, 3]
            stride         = np.asarray(x_shape[1:3]) // np.asarray(output_shape[0:2])
            params['conv'] = {'number': number, 'shape': shape, 'rate': [1, 1], 'stride': stride, 'padding': 'SAME'}
            shortcut       = conv_bn3(tensor_in, 0, params, mtrain)
        else:
            shortcut       = tensor_in
        
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(tensor_in, 0, params, mtrain)
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(residual,  1, params, mtrain)
        params['group'] = {'output_shape': output_shape, 'filter_shape': filter_shape}
        residual        = group_bn1(residual,       0, params, mtrain)
        tensor_out      = relu1(shortcut+residual,  0, params, mtrain)
    return tensor_out
'''











def group1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    1.保证∏(i=1->n)xi = d(d为原本的参数维度，在这里为特征图的体积H*W*C)，从而进行分组，以进行局部全连接
    2.参数复杂度为∑(i=1->n)(xi)^2
    3.batchnorm只针对/除滤波器输出的维度外/的维度做，滤波器输出的特征为本层看重的特征
    4.本函数包含针对主干网络的注意力机制，以聚集相似特征，使滤波器在较小体积的情况下依然能够尽可能地把相似特征映射为差异特征
      这样不仅提高了网络的抽象能力和表达能力，还起到进一步增大网络感受野的作用。
    '''
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    shape     = params['group']['shape']     #[[[7, 4, 1], [7, 4, 1]], [[4, 7, 1], [4, 7, 1]]]
    num_loop  = params['group']['num_loop']  #[2, 2]
    use_attn  = params['group']['use_attn']
    use_drop  = params['group']['use_drop']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    tsr_out = tensor_in
    shape   = np.asarray(shape)
    num_int = np.prod(shape[:, 0], axis=1)                                       #[h1*w1*c1, h2*w2*c2]
    num_out = np.prod(shape[:, 1], axis=1)                                       #[h'*w'*c', h"*w"*c"]
    
    with tf.variable_scope('group1_'+str(layer), reuse=reuse) as scope:
        
        if use_attn:
            gamma = tf.get_variable(name='gamma', shape=[len(shape)], dtype=dtype, \
                                    initializer=tf.constant_initializer(0.0), trainable=trainable)
        
        for l in range(len(shape)):
            
            shp_int = get_shape(tsr_out)                                          #[N, H, W, C]
            bat_siz = shp_int[0]                                                  # N
            shp_int = np.asarray(shp_int[1:])                                     #[H, W, C]
            shp_out = shp_int // shape[l][0]                                      #[H', W', C']
            shp_int = np.stack([shp_out, shape[l][0]], axis=-1)                   #[[H', h], [W', w], [C', c]]
            shp_int = np.reshape(shp_int, -1)                                     #[H', h, W', w, C', c]
            shp_int = [bat_siz] + list(shp_int)                                   #[N, H', h, W', w, C', c]
            shp_idx = np.arange(1, len(shp_int)).reshape([-1, 2])                 #[[1, 2], [3, 4], [5, 6]]
            shp_idx = np.transpose(shp_idx).reshape(-1)                           #[1, 3, 5, 2, 4, 6]
            shp_idx = [ 0] + list(shp_idx)                                        #[0, 1, 3, 5, 2, 4, 6]
            shp_out = [-1] + list(shp_out)                                        #[-1, H', W', C']
            
            tsr_out = tf.reshape(tsr_out,   shp_int)                              #(N, H', h, W', w, C', c)
            tsr_out = tf.transpose(tsr_out, shp_idx)                              #(N, H', W', C', h, w, c)
            tsr_out = tf.reshape(tsr_out, [-1, num_int[l]])                       #(N*H'*W'*C', h*w*c)
            
            if use_attn:
                params['affine'] = {'dim': num_int[l], 'use_bias': True}
                tsr_qry = affine1(tsr_out, 3*l+0, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_key = affine1(tsr_out, 3*l+1, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_vau = affine1(tsr_out, 3*l+2, params, mtrain)                 #(N*H'*W'*C', h*w*c)
                tsr_qry = tf.reshape(tsr_qry, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c)
                tsr_key = tf.reshape(tsr_key, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c)
                tsr_vau = tf.reshape(tsr_vau, [bat_siz, -1, num_int[l]])          #(N, H'*W'*C', h*w*c   )
                qry_key = tf.matmul(tsr_qry, tsr_key, transpose_b=True)           #(N, H'*W'*C', H'*W'*C')
                qry_key = tf.nn.softmax(qry_key, axis=-1)                         #(N, H'*W'*C', H'*W'*C')
                tsr_att = tf.matmul(qry_key, tsr_vau)                             #(N, H'*W'*C', h*w*c)
                tsr_att = tf.reshape(tsr_att, [-1, num_int[l]])                   #(N*H'*W'*C', h*w*c)
                tsr_out = tsr_out + tsr_att * gamma[l]                            #(N*H'*W'*C', h*w*c)
                
            params['affine']  = {'dim': num_out[l], 'use_bias': False}
            for i in range(num_loop[l]):
                tsr_out = affine_bn_relu1(tsr_out, num_loop[l]*l+i, params, mtrain) #(N*H'*W'*C', h'*w'*c')
                
            params['dropout'] = {'keep_p': 0.9, 'shape': None}
            if use_drop:
                tsr_out = dropout1(tsr_out, l, params, mtrain)                    #(N*H'*W'*C', h'*w'*c')
                
            tsr_out = tf.reshape(tsr_out, [bat_siz, -1, num_out[l]])              #(N, H'*W'*C', h'*w'*c')
            tsr_out = tf.transpose(tsr_out, [0, 2, 1])                            #(N, h'*w'*c', H'*W'*C')
            tsr_out = tf.reshape(tsr_out, shp_out)                                #(N*h'*w'*c', H', W', C')
        '''
        prt_opa = tf.print([gamma])
        with tf.control_dependencies([prt_opa]):
            tsr_out = tf.identity(tsr_out)
        '''
        #tsr_out-->(N*h'*w'*c'*h"*w"*c", H', W', C')
        shp_int  = get_shape(tensor_in)                                           #[N, H, W, C]
        bat_siz  = shp_int[0]                                                     # N
        shp_int  = np.asarray(shp_int[1:])                                        #[H, W, C]
        shape    = np.concatenate([shape[:, 1], [shp_out[1:]]], axis=0)           #[[h', w', c'], [h", w", c"], [H', W', C']]
        shp_out0 = [bat_siz] + list(np.reshape(shape,  -1))                       #[N, h', w', c', h", w", c", H', W', C']
        shp_out1 = [bat_siz] + list(np.prod(shape, axis=0))                       #[N, h'*h"*H', w'*w"*W', c'*c"*C']
        
        shp_idx  = np.arange(1, len(shp_out0))                                    #[1, 2, 3, 4, 5, 6, 7, 8, 9]
        shp_idx  = np.reshape(shp_idx, shape.shape)                               #[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        shp_idx  = np.transpose(shp_idx)                                          #[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        shp_idx  = shp_idx[:, ::-1].reshape(-1)                                   #[7, 4, 1, 8, 5, 2, 9, 6, 3]
        shp_idx  = [0] + list(shp_idx)                                            #[0, 7, 4, 1, 8, 5, 2, 9, 6, 3]
        
        tsr_out  = tf.reshape  (tsr_out, shp_out0)                                #(N, h', w', c', h", w", c", H', W', C')
        tsr_out  = tf.transpose(tsr_out, shp_idx )                                #(N, H', h", h', W', w", w', C', c", c')
        tsr_out  = tf.reshape  (tsr_out, shp_out1)                                #(N, H'*h"*h', W'*w"*w', C'*c"*c')
        #tsr_out = relu1(tsr_out, l, params, mtrain)                              #(N, H'*h"*h', W'*w"*w', C'*c"*c')
    return tsr_out










att_tsr = tsr_out * alpha[l]                                      #(N, H'*W'*C', h'*w'*c')
                att_key = tf.einsum('ijk, imk->ijm', att_tsr, att_tsr)            #(N, H'*W'*C', H'*W'*C')
                att_key = tf.nn.softmax(att_key, axis=-1)                         #(N, H'*W'*C', H'*W'*C')
                #att_num= tf.sqrt(tf.cast(num_out[l], dtype=tf.float32))          # sqrt(h'*w'*c')
                #att_key= tf.nn.softmax(att_key/att_num, axis=-1)                 #(N, H'*W'*C', H'*W'*C')
                tsr_att = tf.einsum('ijk, ikm->ijm', att_key, tsr_out)            #(N, H'*W'*C', h'*w'*c')
                tsr_out = tsr_out + tsr_att *  beta[l]                            #(N, H'*W'*C', h'*w'*c')



if use_attn:
            alpha = tf.get_variable(name='alpha', shape=[len(shape)], dtype=dtype, \
                                    initializer=tf.truncated_normal_initializer(stddev=wscale), trainable=trainable)
            beta  = tf.get_variable(name='beta',  shape=[len(shape)], dtype=dtype, \
                                    initializer=tf.truncated_normal_initializer(stddev=wscale), trainable=trainable)



def group1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    1.保证∏(i=1->n)xi = d(d为原本的参数维度，在这里为特征图的体积H*W*C)，从而进行分组，以进行局部全连接
    2.参数复杂度为∑(i=1->n)(xi)^2
    3.batchnorm只针对/除滤波器输出的维度外/的维度做，滤波器输出的特征为本层看重的特征
    4.本函数包含针对主干网络的注意力机制，以聚集相似特征，使滤波器在较小体积的情况下依然能够尽可能地把相似特征映射为差异特征
      这样不仅提高了网络的抽象能力和表达能力，还起到进一步增大网络感受野的作用。
    '''
    output_shape = params['group']['output_shape'] #[H', W', C']
    filter_shape = params['group']['filter_shape'] #[h, w, c]    [8, 8, 8]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)                 #[N, H, W, C]
    
    #降维时刚进入堆叠层时就降，升维时最终输出堆叠层时才升，以节省内存，在信息量不过损失的情况下，保持bottleneck连接
    out_in  = np.asarray(output_shape) / np.asarray(x_shape[1:])
    
    #保证∏(i=1->n)xi = d(d为原本的参数维度，比如空间的面积或通道的数量，从而进行分组，以进行局部全连接)，参数复杂度∑(i=1->n)(xi)^2
    t_num = []
    t_res = []
    t_add = []
    for i in range(len(filter_shape)):
        d_add = False
        d_num = np.log(x_shape[1+i]) // np.log(filter_shape[i])
        d_num = d_num.astype(dtype=np.int32, copy=False)
        d_shp = np.power(filter_shape[i], d_num)
        d_shp = d_shp.astype(dtype=np.int32, copy=False)
        d_res = x_shape[1+i] // d_shp
        d_shp = d_shp * d_res
        assert d_shp == x_shape[1+i], 'The filter_shape[%d] and x_shape[%d] do not match!' %(i, i+1)
        if d_res != 1 or x_shape[1+i] == 1:
            d_num = d_num + 1
            d_add = True
        t_num.append(d_num)
        t_res.append(d_res)
        t_add.append(d_add)
    
    #t_num=[2, 1, 1], t_res=[2, 1, 1], t_add=[True, False, True], x_shape=[N, 8, 4, 1], filter_shape=[4, 4, 1]
    #[[[4, 4, 1], [4, 4, 1]], [[2, 1, 1], [2, 1, 1]]]
    m_num = np.amax(t_num)
    #输入输出参数，本着t_shp[i][0]和t_shp[i][1]差异尽可能小的原则，这样参数利用率高
    t_shp = [] #[[[h, w, c], [h', w', c']]]*np.amax(t_num)
    for i in range(np.amax(t_num)):
        d_shp = []
        for j in range(len(filter_shape)):
            
    
    
    
    return






def group_bn1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    #保证∏(i=1->n)xi = d(d为原本的参数维度，比如空间的面积或通道的数量，从而进行分组，以进行局部全连接)，参数复杂度∑(i=1->n)(xi)^2
    #不需要rate，因为本卷积关联整个图像
    #不需要pad， 因为本卷积只利用图像中的有效像素
    #stride不由卷积控制，而由avg_pool控制，先做avg_pool，再做relu
    #batchnorm只针对batch做，不针对通道或空间，因为通道和空间本身都是特征
    #当面积或通道数较小时，没有必要分块进行全连接
    
    output_shape = params['group']['output_shape'] #[H', W', C']
    filter_shape = params['group']['filter_shape'] #[h, w, c]    [8, 8, 64]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()   #(N, H, W, C)
    x_shape = get_shape(tensor_in)
    
    #降维时刚进入堆叠层时就降，升维时最终输出堆叠层时才升，以节省内存，在信息量不过损失的情况下，保持bottleneck连接
    out_in  = np.asarray(output_shape) / np.asarray(x_shape[1:])
    assert (out_in[0]<=1 and out_in[1]<=1) or (out_in[0]>1 and out_in[1]>1), 'The space shape of output is wrong!'
    
    #保证∏(i=1->n)xi = d(d为原本的参数维度，比如空间的面积或通道的数量，从而进行分组，以进行局部全连接)，参数复杂度∑(i=1->n)(xi)^2
    #空间关联次数p_num
    p_add = False
    p_num = np.log(np.asarray(x_shape[1:3])) // np.log(np.asarray(filter_shape[0:2]))
    p_num = p_num.astype(dtype=np.int32, copy=False)
    assert p_num[0] == p_num[1], 'The space shape of filter is wrong!'
    p_num = p_num[0]
    p_shp = np.power(np.asarray(filter_shape[0:2]), p_num)
    p_shp = p_shp.astype(dtype=np.int32, copy=False)
    p_res = np.asarray(x_shape[1:3]) // p_shp
    p_shp = p_shp * p_res
    assert np.all(p_shp == np.asarray(x_shape[1:3])), 'The space shape of filter is wrong!'
    if np.any(p_res != np.array([1, 1])) or np.all(np.asarray(x_shape[1:3]) == np.array([1, 1])):
        p_num = p_num + 1
        p_add = True
    
    #通道关联次数c_num
    c_add = False
    c_num = np.log(x_shape[-1]) // np.log(filter_shape[-1])
    c_num = c_num.astype(dtype=np.int32, copy=False)
    c_shp = np.power(filter_shape[-1], c_num)
    c_shp = c_shp.astype(dtype=np.int32, copy=False)
    c_res = x_shape[-1] // c_shp
    c_shp = c_shp * c_res
    assert c_shp == x_shape[-1], 'The channel shape of filter is wrong!'
    if c_res != 1 or x_shape[-1] == 1:
        c_num = c_num + 1
        c_add = True
    
    #空间关联参数，本着pos_shp0和pos_shp1差异尽可能小的原则，这样参数利用率高
    p_shape = [] #[[8, 8, p1, p2]]*p_number，只针对空间，8*8 × p1*p2 的全连接共p_number次，使用conv2d
    for i in range(p_num):
        pos_shp = []
        #若面积增大且有p_res，则将其放到最后一层；若面积减小或相等且有p_res，则将其放到第一层
        #out_in[0]<=1 or out_in[0]>1肯定会发生，i==0和i==p_num-1也肯定会经过，p_res存在的话肯定会得到处理
        pos_shp0 = np.asarray(filter_shape[0:2])
        if (i == 0 and out_in[0] <= 1) or (i == p_num - 1 and out_in[0] > 1):
            if p_add:
                pos_shp0 = p_res
            pos_shp1 = pos_shp0 * out_in[0:2]
            pos_shp1 = pos_shp1.astype(dtype=np.int32, copy=False)
            assert np.all(pos_shp1 / pos_shp0 == out_in[0:2]), 'The space shape of output is wrong!'
        else:
            pos_shp1 = pos_shp0
        pos_shp.extend(list(pos_shp0))
        pos_shp.extend(list(pos_shp1))
        p_shape.append(pos_shp)
    
    #通道关联参数，本着chn_shp0和chn_shp1差异尽可能小的原则，这样参数利用率高
    c_shape = [] #[[16, c]]*c_number，只针对通道，16  × c 的全连接共c_number次，使用conv1d
    for i in range(c_num):
        chn_shp = []
        #若通道增多且有c_res，则将其放到最后一层；若通道减少或相等且有c_res，则将其放到第一层
        #out_in[-1]<=1 or out_in[-1]>1肯定会发生，i==0和i==c_num-1也肯定会经过，c_res存在的话肯定会得到处理
        chn_shp0 = filter_shape[-1]
        if (i == 0 and out_in[-1] <= 1) or (i == c_num -1 and out_in[-1] > 1):
            if c_add:
                chn_shp0 = c_res
            chn_shp1 = chn_shp0 * out_in[-1]
            chn_shp1 = chn_shp1.astype(dtype=np.int32, copy=False)
            assert chn_shp1 / chn_shp0 == out_in[-1], 'The channel shape of output is wrong!'
        else:
            chn_shp1 = chn_shp0
        chn_shp.append(chn_shp0)
        chn_shp.append(chn_shp1)
        c_shape.append(chn_shp)
    
    p_shp   = np.prod(p_shape, axis=0)
    p_srd   = p_shp[0:2] // p_shp[2:]
    assert np.all(p_shp[0:2] == np.asarray(x_shape[1:3])), 'The p_shape is wrong!'
    
    c_shp   = np.prod(c_shape, axis=0)
    c_srd   = c_shp[0] // c_shp[1]
    assert c_shp[0] == x_shape[-1], 'The c_shape is wrong!'
    
    def pos_group_bn1(tensor_in, layer):
        with tf.variable_scope('pos_group_bn1_'+str(layer), reuse=reuse) as scope:
            #提取空间特征
            x_shape = get_shape(tensor_in)                                            #(N, H, W, C)
            pra_num = x_shape[0] * x_shape[-1]                                        # N*C
            fet_pos = tf.transpose(tensor_in, [0, 3, 1, 2])                           #(N, C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[1], x_shape[2]])               #(N*C, H, W)
            
            def dispatch(fet_pos):
                #fet_pos --> (H, W)
                fet_shp = np.asarray(x_shape[1:3])                                    #[H, W]
                fet_pos = tf.reshape(fet_pos, [1]+x_shape[1:3]+[1])                   #(1, H, W, 1)
                for i in range(len(p_shape)):
                    pos_shp        = p_shape[i]
                    fet_shp        = fet_shp // np.asarray(pos_shp[0:2])              #[H', W']
                    params['conv'] =  {'number': pos_shp[2]*pos_shp[3], 'shape': pos_shp[0:2], 'rate': 1, 'stride': pos_shp[0:2], \
                                       'padding': 'VALID', 'use_bias': False}
                    fet_pos        = conv1(fet_pos, i, params, mtrain)                #(C"', H', W', C")
                    fet_pos        = tf.transpose(fet_pos, [0, 3, 1, 2])              #(C"', C", H', W')把已关联特征放到下层继续关联剩下的
                    fet_pos        = tf.reshape(fet_pos, [-1]+list(fet_shp)+[1])      #(C"', H', W', 1)
                return fet_pos
            
            fet_pos = tf.map_fn(dispatch, fet_pos, dtype=tf.float32, parallel_iterations=pra_num, \
                                back_prop=True, swap_memory=True, infer_shape=True)   #(N*C, C"', 1, 1, 1)
            
            fet_pos = tf.reshape(fet_pos, [-1, p_shp[2]*p_shp[3]])                    #(N*C, C"')C"'是空间特征，做BN时应该对之外的维度做
            fet_pos = batchnorm1(fet_pos, 0, params, mtrain)                          #(N*C, C"')
            shape   = [-1] + list(np.asarray(p_shape)[:, 2:].reshape(-1))             #还原空间维度
            fet_pos = tf.reshape(fet_pos, shape)                                      #(N*C, H, W)
            perm    = [0] + [x for x in range(1, 1+2*len(p_shape), 2)][::-1] + [x for x in range(2, 2+2*len(p_shape), 2)][::-1]
            fet_pos = tf.transpose(fet_pos, perm)                                     #(N*C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[-1], p_shp[2], p_shp[3]])      #(N, C, H, W)
            fet_pos = tf.transpose(fet_pos, [0, 2, 3, 1])                             #(N, H, W, C)
            print_activations(fet_pos)
        return fet_pos
    
    def chn_group_bn1(tensor_in, layer):
        with tf.variable_scope('chn_group_bn1_'+str(layer), reuse=reuse) as scope:
            #提取通道特征
            x_shape = get_shape(tensor_in)                                            #(N, H, W, C)
            pra_num = x_shape[0] * x_shape[1] * x_shape[2]                            # N*H*W
            fet_chn = tf.reshape(tensor_in, [-1, x_shape[-1]])                        #(N*H*W, C)
            
            def dispatch(fet_chn):
                #fet_chn --> (C)
                fet_shp = x_shape[-1]                                                 #[C]
                fet_chn = tf.reshape(fet_chn, [1]+x_shape[-1]+[1])                    #(1, C, 1)
                for i in range(len(c_shape)):
                    chn_shp        = c_shape[i]
                    fet_shp        = fet_shp // chn_shp[0]                            #[C']
                    params['conv'] = {'number': chn_shp[1], 'shape': chn_shp[0], 'rate': 1, 'stride': chn_shp[0], \
                                      'padding': 'VALID', 'use_bias': False}
                    fet_chn        = conv2(fet_chn, i, params, mtrain)                #(C"', C', C")
                    fet_chn        = tf.transpose(fet_chn, [0, 2, 1])                 #(C"', C", C')把已关联特征放到下层继续关联剩下的
                    fet_chn        = tf.reshape(fet_chn, [-1, fet_shp, 1])            #(C"', C',  1)
                return fet_chn
            
            fet_chn = tf.map_fn(dispatch, fet_chn, dtype=tf.float32, parallel_iterations=pra_num, \
                                back_prop=True, swap_memory=True, infer_shape=True)   #(N*H*W, C"', 1, 1)
            
            fet_chn = tf.reshape(fet_chn, [-1, c_shp[1]])                             #(N*H*W, C"')C"'是通道特征，做BN时应该对之外的维度做
            fet_chn = batchnorm1(fet_chn, 1, params, mtrain)                          #(N*H*W, C"')
            shape   = [-1] + list(np.asarray(c_shape)[:, 1:].reshape(-1))             #还原通道维度
            fet_chn = tf.reshape(fet_chn, shape)                                      #(N*H*W, C)
            perm    = [0] + [x for x in range(1, 1+len(c_shape), 1)][::-1]
            fet_chn = tf.transpose(fet_chn, perm)                                     #(N*H*W, C)
            fet_chn = tf.reshape(fet_chn, [-1, x_shape[1], x_shape[2], c_shp[1]])     #(N, H, W, C)
            print_activations(fet_chn)
        return fet_chn
    
    with tf.variable_scope('group_bn1_'+str(layer), reuse=reuse) as scope:
        fet_pos0   = pos_group_bn1(tensor_in, 0)
        fet_chn0   = chn_group_bn1(tensor_in, 0)
        fet_pos1   = pos_group_bn1(fet_chn0,  1)
        fet_chn1   = chn_group_bn1(fet_pos0,  1)
        tensor_out = fet_pos1 + fet_chn1
        print_activations(tensor_out)
    return tensor_out


def group_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('group_bn_relu1_'+str(layer)) as scope:
        bn         = group_bn1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain) 
    return tensor_out


def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    output_shape = params['group_unit']['output_shape'] #(H', W', C')
    bottle_shape = params['group_unit']['bottle_shape'] #(H", W", C")
    filter_shape = params['group_unit']['filter_shape'] #(h, w, c)
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape = tensor_in.get_shape().as_list() #(N, H, W, C)
    x_shape = get_shape(tensor_in)            #(N, H, W, C)
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:

        if np.any(np.asarray(x_shape[1:]) != np.asarray(output_shape)): #深度可分离卷积!!!
            number         = output_shape[-1] // x_shape[-1]
            shape          = [3, 3]
            stride         = np.asarray(x_shape[1:3]) // np.asarray(output_shape[0:2])
            params['conv'] = {'number': number, 'shape': shape, 'rate': [1, 1], 'stride': stride, 'padding': 'SAME'}
            shortcut       = conv_bn3(tensor_in, 0, params, mtrain)
        else:
            shortcut       = tensor_in
        
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(tensor_in, 0, params, mtrain)
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(residual,  1, params, mtrain)
        params['group'] = {'output_shape': output_shape, 'filter_shape': filter_shape}
        residual        = group_bn1(residual,       0, params, mtrain)
        tensor_out      = relu1(shortcut+residual,  0, params, mtrain)
    return tensor_out
    
    
def group_block1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    block_setting = params['group_block']['block_setting']

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    tensor_out    = tensor_in
    out_list      = []
    for i, block in enumerate(block_setting):
        
        output_shape, bottle_shape, filter_shape, unit_number, unit_trainable = block
        params['com']['trainable'] = unit_trainable
        with tf.variable_scope('group_block1_'+str(layer)+'_'+str(i)) as scope:
            for j in range(unit_number):
                params['group_unit'] = {'output_shape':output_shape, 'bottle_shape':bottle_shape, 'filter_shape':filter_shape}
                tensor_out           = group_unit1(tensor_out, j, params, mtrain)
        out_list.append(tensor_out)
    return out_list


def atten1(tensor_in=None, layer=0, params=None, mtrain=None):
    #对空间关联，group_bn在独立的通道上关联空间，attention在所有的通道上关联空间
    #对通道关联，group_bn在独立的空间上关联通道，attention在所有的空间上关联通道
    #因此，attention是group_bn的集大成者
    #attention作用的层数要把握好，空间面积和通道数都不宜过大过小
    #
    output_shape = params['atten']['output_shape'] #[H', W', C']
    attent_shape = params['atten']['attent_shape'] #[h, w, c]    [8, 8, 64]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()   #(N, H, W, C)
    x_shape = get_shape(tensor_in)
    
    #降维时刚进入堆叠层时就降，升维时最终输出堆叠层时才升，以节省内存，在信息量不过损失的情况下，保持bottleneck连接
    out_in  = np.asarray(output_shape) / np.asarray(x_shape[1:])
    assert (out_in[0]<=1 and out_in[1]<=1) or (out_in[0]>1 and out_in[1]>1), 'The space shape of output is wrong!'
    
    #空间关联次数p_num
    p_add = False
    p_num = np.log(np.asarray(x_shape[1:3])) // np.log(np.asarray(attent_shape[0:2]))
    p_num = p_num.astype(dtype=np.int32, copy=False)
    assert p_num[0] == p_num[1], 'The space shape of attent is wrong!'
    p_num = p_num[0]
    p_shp = np.power(np.asarray(attent_shape[0:2]), p_num)
    p_shp = p_shp.astype(dtype=np.int32, copy=False)
    p_res = np.asarray(x_shape[1:3]) // p_shp
    p_shp = p_shp * p_res
    assert np.all(p_shp == np.asarray(x_shape[1:3])), 'The space shape of attent is wrong!'
    if np.any(p_res != np.array([1, 1])) or np.all(np.asarray(x_shape[1:3]) == np.array([1, 1])):
        p_num = p_num + 1
        p_add = True
    
    #通道关联次数c_num
    c_add = False
    c_num = np.log(x_shape[-1]) // np.log(attent_shape[-1])
    c_num = c_num.astype(dtype=np.int32, copy=False)
    c_shp = np.power(attent_shape[-1], c_num)
    c_shp = c_shp.astype(dtype=np.int32, copy=False)
    c_res = x_shape[-1] // c_shp
    c_shp = c_shp * c_res
    assert c_shp == x_shape[-1], 'The channel shape of attent is wrong!'
    if c_res != 1 or x_shape[-1] == 1:
        c_num = c_num + 1
        c_add = True
    
    #空间关联参数，本着pos_shp0和pos_shp1差异尽可能小的原则，这样参数利用率高
    p_shape = [] #[[8, 8, p1, p2]]*p_number
    for i in range(p_num):
        pos_shp = []
        #若面积增大且有p_res，则将其放到最后一层；若面积减小或相等且有p_res，则将其放到第一层
        #out_in[0]<=1 or out_in[0]>1肯定会发生，i==0和i==p_num-1也肯定会经过，p_res存在的话肯定会得到处理
        pos_shp0 = np.asarray(attent_shape[0:2])
        if (i == 0 and out_in[0] <= 1) or (i == p_num - 1 and out_in[0] > 1):
            if p_add:
                pos_shp0 = p_res
            pos_shp1 = pos_shp0 * out_in[0:2]
            pos_shp1 = pos_shp1.astype(dtype=np.int32, copy=False)
            assert np.all(pos_shp1 / pos_shp0 == out_in[0:2]), 'The space shape of output is wrong!'
        else:
            pos_shp1 = pos_shp0
        pos_shp.extend(list(pos_shp0))
        pos_shp.extend(list(pos_shp1))
        p_shape.append(pos_shp)
        
    #通道关联参数，本着chn_shp0和chn_shp1差异尽可能小的原则，这样参数利用率高
    c_shape = [] #[[16, c]]*c_number
    for i in range(c_num):
        chn_shp = []
        #若通道增多且有c_res，则将其放到最后一层；若通道减少或相等且有c_res，则将其放到第一层
        #out_in[-1]<=1 or out_in[-1]>1肯定会发生，i==0和i==c_num-1也肯定会经过，c_res存在的话肯定会得到处理
        chn_shp0 = attent_shape[-1]
        if (i == 0 and out_in[-1] <= 1) or (i == c_num -1 and out_in[-1] > 1):
            if c_add:
                chn_shp0 = c_res
            chn_shp1 = chn_shp0 * out_in[-1]
            chn_shp1 = chn_shp1.astype(dtype=np.int32, copy=False)
            assert chn_shp1 / chn_shp0 == out_in[-1], 'The channel shape of output is wrong!'
        else:
            chn_shp1 = chn_shp0
        chn_shp.append(chn_shp0)
        chn_shp.append(chn_shp1)
        c_shape.append(chn_shp)
    
    p_shp   = np.prod(p_shape, axis=0)
    p_srd   = p_shp[0:2] // p_shp[2:]
    assert np.all(p_shp[0:2] == np.asarray(x_shape[1:3])), 'The p_shape is wrong!'
    
    c_shp   = np.prod(c_shape, axis=0)
    c_srd   = c_shp[0] // c_shp[1]
    assert c_shp[0] == x_shape[-1], 'The c_shape is wrong!'
    
    def pos_atten1(tensor_in, layer):
        with tf.variable_scope('pos_atten1_'+str(layer), reuse=reuse) as scope:
            #关联空间特征
            x_shape = get_shape(tensor_in)                                            #[N, H, W, C]
            pra_num = x_shape[0]                                                      # N
            
            def dispatch(fet_pos):
                #fet_pos --> (H, W, C)
                fet_shp = np.asarray(x_shape[1:3])                                    #[H, W]
                
                for i in range(len(p_shape)):
                    pos_shp = p_shape[i]
                    fet_shp = fet_shp // np.asarray(pos_shp[0:2])                     #[H', W']
                    pra_num = fet_shp[0] * fet_shp[1]                                 # H'*W'
                    
                    def cond(i, fet_out):
                        c = tf.less(i, fet_shp[0] * fet_shp[1])
                        return c

                    def body(i, fet_out):
                        
                        ycd  = i // fet_shp[1]
                        xcd  = i  % fet_shp[1]
                        beg  = [ycd*pos_shp[0], xcd*pos_shp[1],  0]
                        siz  = [    pos_shp[0],     pos_shp[1], -1]
                        fet0 = tf.slice(fet_pos, beg, siz)                            #(h, w, C)
                        if np.any(np.asarray(pos_shp[0:2]) != np.asarray(pos_shp[2:])):
                            fet1 = tf.image.resize_images(fet0, pos_shp[2:], method=tf.image.ResizeMethod.BILINEAR, \
                                                          align_corners=False, preserve_aspect_ratio=False)
                        else:
                            fet1 = fet0
                        
                        
                        
                        return [i+1, fet_out]

                    i = tf.constant(0)
                    [i, fet_out] = tf.while_loop(cond, body, loop_vars=[i, fet_out], shape_invariants=None, \
                                                 parallel_iterations=pra_num, back_prop=True, swap_memory=True)
                    fet_pos = fet_out
                    
                return fet_pos
                    
                    
    
    def pos_group_bn1(tensor_in, layer):
        with tf.variable_scope('pos_group_bn1_'+str(layer), reuse=reuse) as scope:
            #提取空间特征
            x_shape = get_shape(tensor_in)                                            #(N, H, W, C)
            pra_num = x_shape[0] * x_shape[-1]                                        # N*C
            fet_pos = tf.transpose(tensor_in, [0, 3, 1, 2])                           #(N, C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[1], x_shape[2]])               #(N*C, H, W)
            
            def dispatch(fet_pos):
                fet_shp = np.asarray(x_shape[1:3])                                    #[H, W]
                fet_pos = tf.reshape(fet_pos, [1]+x_shape[1:3]+[1])                   #(1, H, W, 1)
                for i in range(len(p_shape)):
                    pos_shp        = p_shape[i]                                       #[H', W']
                    fet_shp        = fet_shp // np.asarray(pos_shp[0:2])
                    params['conv'] =  {'number': pos_shp[2]*pos_shp[3], 'shape': pos_shp[0:2], 'rate': 1, 'stride': pos_shp[0:2], \
                                       'padding': 'VALID', 'use_bias': False}
                    fet_pos        = conv1(fet_pos, i, params, mtrain)                #(C"', H', W', C")
                    fet_pos        = tf.transpose(fet_pos, [0, 3, 1, 2])              #(C"', C", H', W')把已关联特征放到下层继续关联剩下的
                    fet_pos        = tf.reshape(fet_pos, [-1]+list(fet_shp)+[1])      #(C"', H', W', 1)
                return fet_pos
            
            fet_pos = tf.map_fn(dispatch, fet_pos, dtype=tf.float32, parallel_iterations=pra_num, \
                                back_prop=True, swap_memory=True, infer_shape=True)   #(N*C, C"', 1, 1, 1)
            
            fet_pos = tf.reshape(fet_pos, [-1, p_shp[2]*p_shp[3]])                    #(N*C, C"')C"'是空间特征，做BN时应该对之外的维度做
            fet_pos = batchnorm1(fet_pos, 0, params, mtrain)                          #(N*C, C"')
            shape   = [-1] + list(np.asarray(p_shape)[:, 2:].reshape(-1))             #还原空间维度
            fet_pos = tf.reshape(fet_pos, shape)                                      #(N*C, H, W)
            perm    = [0] + [x for x in range(1, 1+2*len(p_shape), 2)][::-1] + [x for x in range(2, 2+2*len(p_shape), 2)][::-1]
            fet_pos = tf.transpose(fet_pos, perm)                                     #(N*C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[-1], p_shp[2], p_shp[3]])      #(N, C, H, W)
            fet_pos = tf.transpose(fet_pos, [0, 2, 3, 1])                             #(N, H, W, C)
            print_activations(fet_pos)
        return fet_pos
    
    
    
    
    
    
    
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    shape     = params['attent_unit']['shape']
    rate      = params['attent_unit']['rate']
    depth_key = params['attent_unit']['depth_key']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape     = tensor_in.get_shape().as_list()
    depth_input = x_shape[-1]
    height      = x_shape[ 1]
    width       = x_shape[ 2]
    
    with tf.variable_scope('attent_unit1_'+str(layer)) as scope:
    
        #使用3x3conv隔离特征
        params['conv'] = {'number':depth_input, 'shape':[3, 3], 'rate':1, 'stride':[1, 1], 'padding':'SAME'}
        tensor_in      = conv_bn_relu1(tensor_in, 0, params, mtrain)
        tensor_in      = conv_bn_relu1(tensor_in, 1, params, mtrain)
        #对keys的关联应该在放入位置向量之前，位置向量主要服务于关联中心点，对中心特征向量和其之外的特征向量之间的位置关系做描述
        params['conv'] = {'number':depth_key,   'shape':[1, 1], 'rate':1, 'stride':[1, 1], 'padding':'VALID', 'use_bias': True}
        tensor_key     = conv1(tensor_in, 0, params, mtrain)
        
        
        params['conv'] = {'number':depth_input, 'shape':[1, 1], 'rate':1, 'stride':[1, 1], 'padding':'VALID', 'use_bias': True}
        tensor_value   = conv1(tensor_in, 0, params, mtrain)
        
        #获取relative_position_embeddings #(64, 64, 64)
        PE = tf.get_variable(name='PE', shape=shape+depth_key, dtype=dtype, \
                             #initializer=tf.truncated_normal_initializer(stddev=wscale), \
                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                             regularizer=tf.contrib.layers.l2_regularizer(reg), trainable=trainable)
        
        tensor_out = tf.TensorArray(dtype=tf.float32, size=height*width, dynamic_size=False, clear_after_read=True, \
                                    infer_shape=True, element_shape=[depth_input+depth_key], colocate_with_first_write_call=True)
        
        def cond(i, tensor_out):
            c = tf.less(i, height*width)
            return c

        def body(i, tensor_out):
            
            ycd   = i // width
            xcd   = i  % width
            ymn   = ycd - ((shape[0] - 1) // 2) * rate
            xmn   = xcd - ((shape[1] - 1) // 2) * rate
            ycds  = tf.concat([[ymn], tf.tile([rate], [shape[0]-1])], axis=0)
            xcds  = tf.concat([[xmn], tf.tile([rate], [shape[1]-1])], axis=0)
            ycds  = tf.cumsum(ycds, axis=0, exclusive=False, reverse=False)
            xcds  = tf.cumsum(xcds, axis=0, exclusive=False, reverse=False)
            idxs  = tf.where(tf.logical_and(ycds>=0, ycds<height))
            ycds  = tf.gather_nd(ycds, idxs)
            idxs  = tf.where(tf.logical_and(xcds>=0, xcds<width ))
            xcds  = tf.gather_nd(xcds, idxs)
            ycds  = tf.tile(ycds[:, tf.newaxis], [1, tf.shape(xcds)[0]])
            xcds  = tf.tile(xcds[tf.newaxis, :], [tf.shape(ycds)[0], 1])
            crd0  = tf.stack([ycd, xcd], axis=0)                                 #(2)         实际中心
            crds0 = tf.concat([ycds, xcds], axis=-1)                             #(h, w, 2)   实际坐标
            fets0 = tf.gather_nd(tensor_value, crds0)                            #(h, w, c)   实际特征
            fets3 = tf.gather_nd(tensor_key,   crds0)                            #(h, w, c')  实际特征
            crd1  = (shape - 1) // 2                                             #(2)         相对中心
            crds1 = (crds0 - crd0) // rate                                       #(h, w, 2)   相对坐标
            crds1 = crds1 + crd1                                                 #(h, w, 2)   相对坐标
            fets1 = tf.gather_nd(PE, crds1)                                      #(h, w, c)   相对特征
            #fets2= tf.concat([fets0, fets1], axis=-1)                           #(h, w, c'') 融合特征
            crd3  = crd0 - crds0[0, 0]         #crd、crds下标换成1也一样           #(2)         相对坐标
            fet3  = tf.gather_nd(fets3, crd3)                                    #(c')        相对中心
            #计算注意力
            att3  = tf.einsum('ijk,k->ij', fets3, fet3)                          #(h, w)
            att3  = tf.exp(att3 / tf.sqrt(depth_key))                            #(h, w)
            att3  = att3 / tf.reduce_sum(att3)                                   #(h, w)
            fet0  = tf.einsum('ij,ijk->k', att3, fets0)                          #(c)
            fet1  = tf.einsum('ij,ijk->k', att3, fets1)                          #(c')
            fet2  = tf.concat([fet0, fet1], axis=-1)                             #(c'')
            #fet2 = tf.einsum('ij,ijk->k', att3, fets2)                          #(c'')
            tensor_out = tensor_out.write(i, fet2)                               #(h, w, c')
            return [i+1, tensor_out]
        
        i = tf.constant(0)
        [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                        parallel_iterations=128, back_prop=True, swap_memory=True)
        #使用1x1conv进行特征和位置向量的融合
        params['conv']  = {'number':depth_input, 'shape':[1, 1], 'rate':1, 'stride':[1, 1], 'padding':'VALID'}
        tensor_out      = conv_bn1(tensor_in, 0, params, mtrain)
        tensor_out      = relu1(tensor_out + tensor_in)
        return tensor_out
        return fet_pos
    
    
    def chn_atten1(tensor_in, layer):
        with tf.variable_scope('chn_atten1_'+str(layer), reuse=reuse) as scope:



            print_activations(fet_pos)
        return fet_pos
    
    with tf.variable_scope('atten1_'+str(layer), reuse=reuse) as scope:
        fet_pos0   = pos_atten1(tensor_in, 0)
        fet_chn0   = chn_atten1(tensor_in, 0)
        fet_pos1   = pos_atten1(fet_chn0,  1)
        fet_chn1   = chn_atten1(fet_pos0,  1)
        tensor_out = fet_pos1 + fet_chn1
        print_activations(tensor_out)
    return tensor_out



























def group_bn1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    #保证∏(i=1->n)xi = d(d为原本的参数维度，比如空间的面积或通道的数量，从而进行分组，以进行局部全连接)，参数复杂度∑(i=1->n)(xi)^2
    #不需要rate，因为本卷积关联整个图像
    #不需要pad， 因为本卷积只利用图像中的有效像素
    #stride不由卷积控制，而由avg_pool控制，先做avg_pool，再做relu
    #batchnorm只针对batch做，不针对通道或空间，因为通道和空间本身都是特征
    #当面积或通道数较小时，没有必要分块进行全连接
    
    output_shape = params['group']['output_shape'] #[H', W', C']
    filter_shape = params['group']['filter_shape'] #[h, w, c]    [8, 8, 64]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()   #(N, H, W, C)
    x_shape = get_shape(tensor_in)
    
    #降维时刚进入堆叠层时就降，升维时最终输出堆叠层时才升，以节省内存，在信息量不过损失的情况下，保持bottleneck连接
    out_in  = np.asarray(output_shape) / np.asarray(x_shape[1:])
    assert (out_in[0]<=1 and out_in[1]<=1) or (out_in[0]>1 and out_in[1]>1), 'The space shape of output is wrong!'
    
    #保证∏(i=1->n)xi = d(d为原本的参数维度，比如空间的面积或通道的数量，从而进行分组，以进行局部全连接)，参数复杂度∑(i=1->n)(xi)^2
    #空间关联次数p_num
    p_add = False
    p_num = np.log(np.asarray(x_shape[1:3])) // np.log(np.asarray(filter_shape[0:2]))
    p_num = p_num.astype(dtype=np.int32, copy=False)
    assert p_num[0] == p_num[1], 'The space shape of filter is wrong!'
    p_num = p_num[0]
    p_shp = np.power(np.asarray(filter_shape[0:2]), p_num)
    p_shp = p_shp.astype(dtype=np.int32, copy=False)
    p_res = np.asarray(x_shape[1:3]) // p_shp
    p_shp = p_shp * p_res
    assert np.all(p_shp == np.asarray(x_shape[1:3])), 'The space shape of filter is wrong!'
    if np.any(p_res != np.array([1, 1])) or np.all(np.asarray(x_shape[1:3]) == np.array([1, 1])):
        p_num = p_num + 1
        p_add = True
    
    #通道关联次数c_num
    c_add = False
    c_num = np.log(x_shape[-1]) // np.log(filter_shape[-1])
    c_num = c_num.astype(dtype=np.int32, copy=False)
    c_shp = np.power(filter_shape[-1], c_num)
    c_shp = c_shp.astype(dtype=np.int32, copy=False)
    c_res = x_shape[-1] // c_shp
    c_shp = c_shp * c_res
    assert c_shp == x_shape[-1], 'The channel shape of filter is wrong!'
    if c_res != 1 or x_shape[-1] == 1:
        c_num = c_num + 1
        c_add = True
    
    #空间关联参数，本着pos_shp0和pos_shp1差异尽可能小的原则，这样参数利用率高
    p_shape = [] #[[8, 8, p1, p2]]*p_number，只针对空间，8*8 × p1*p2 的全连接共p_number次，使用conv2d
    for i in range(p_num):
        pos_shp = []
        #若面积增大且有p_res，则将其放到最后一层；若面积减小或相等且有p_res，则将其放到第一层
        #out_in[0]<=1 or out_in[0]>1肯定会发生，i==0和i==p_num-1也肯定会经过，p_res存在的话肯定会得到处理
        pos_shp0 = np.asarray(filter_shape[0:2])
        if (i == 0 and out_in[0] <= 1) or (i == p_num - 1 and out_in[0] > 1):
            if p_add:
                pos_shp0 = p_res
            pos_shp1 = pos_shp0 * out_in[0:2]
            pos_shp1 = pos_shp1.astype(dtype=np.int32, copy=False)
            assert np.all(pos_shp1 / pos_shp0 == out_in[0:2]), 'The space shape of output is wrong!'
        else:
            pos_shp1 = pos_shp0
        pos_shp.extend(list(pos_shp0))
        pos_shp.extend(list(pos_shp1))
        p_shape.append(pos_shp)
    
    #通道关联参数，本着chn_shp0和chn_shp1差异尽可能小的原则，这样参数利用率高
    c_shape = [] #[[16, c]]*c_number，只针对通道，16  × c 的全连接共c_number次，使用conv1d
    for i in range(c_num):
        chn_shp = []
        #若通道增多且有c_res，则将其放到最后一层；若通道减少或相等且有c_res，则将其放到第一层
        #out_in[-1]<=1 or out_in[-1]>1肯定会发生，i==0和i==c_num-1也肯定会经过，c_res存在的话肯定会得到处理
        chn_shp0 = filter_shape[-1]
        if (i == 0 and out_in[-1] <= 1) or (i == c_num -1 and out_in[-1] > 1):
            if c_add:
                chn_shp0 = c_res
            chn_shp1 = chn_shp0 * out_in[-1]
            chn_shp1 = chn_shp1.astype(dtype=np.int32, copy=False)
            assert chn_shp1 / chn_shp0 == out_in[-1], 'The channel shape of output is wrong!'
        else:
            chn_shp1 = chn_shp0
        chn_shp.append(chn_shp0)
        chn_shp.append(chn_shp1)
        c_shape.append(chn_shp)
    
    p_shp   = np.prod(p_shape, axis=0)
    p_srd   = p_shp[0:2] // p_shp[2:]
    assert np.all(p_shp[0:2] == np.asarray(x_shape[1:3])), 'The p_shape is wrong!'
    
    c_shp   = np.prod(c_shape, axis=0)
    c_srd   = c_shp[0] // c_shp[1]
    assert c_shp[0] == x_shape[-1], 'The c_shape is wrong!'
    
    def pos_group_bn1(tensor_in, layer):
        with tf.variable_scope('pos_group_bn1_'+str(layer), reuse=reuse) as scope:
            #提取空间特征
            x_shape = get_shape(tensor_in)                                            #[N, H, W, C]
            pra_num = x_shape[0] * x_shape[-1]                                        # N*C
            fet_pos = tf.transpose(tensor_in, [0, 3, 1, 2])                           #(N, C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[1], x_shape[2]])               #(N*C, H, W)
            
            def dispatch(fet_pos):
                #fet_pos --> (H, W)
                fet_shp = np.asarray(x_shape[1:3])                                    #[H, W]
                fet_pos = tf.reshape(fet_pos, [1]+x_shape[1:3]+[1])                   #(1, H, W, 1)
                for i in range(len(p_shape)):
                    pos_shp        = p_shape[i]
                    fet_shp        = fet_shp // np.asarray(pos_shp[0:2])              #[H', W']
                    params['conv'] =  {'number': pos_shp[2]*pos_shp[3], 'shape': pos_shp[0:2], 'rate': 1, 'stride': pos_shp[0:2], \
                                       'padding': 'VALID', 'use_bias': False}
                    fet_pos        = conv1(fet_pos, i, params, mtrain)                #(C"', H', W', C")
                    fet_pos        = tf.transpose(fet_pos, [0, 3, 1, 2])              #(C"', C", H', W')把已关联特征放到下层继续关联剩下的
                    fet_pos        = tf.reshape(fet_pos, [-1]+list(fet_shp)+[1])      #(C"', H', W', 1)
                return fet_pos
            
            fet_pos = tf.map_fn(dispatch, fet_pos, dtype=tf.float32, parallel_iterations=pra_num, \
                                back_prop=True, swap_memory=True, infer_shape=True)   #(N*C, C"', 1, 1, 1)
            
            fet_pos = tf.reshape(fet_pos, [-1, p_shp[2]*p_shp[3]])                    #(N*C, C"')C"'是空间特征，做BN时应该对之外的维度做
            fet_pos = batchnorm1(fet_pos, 0, params, mtrain)                          #(N*C, C"')
            shape   = [-1] + list(np.asarray(p_shape)[:, 2:].reshape(-1))             #还原空间维度
            fet_pos = tf.reshape(fet_pos, shape)                                      #(N*C, H, W)
            perm    = [0] + [x for x in range(1, 1+2*len(p_shape), 2)][::-1] + [x for x in range(2, 2+2*len(p_shape), 2)][::-1]
            fet_pos = tf.transpose(fet_pos, perm)                                     #(N*C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[-1], p_shp[2], p_shp[3]])      #(N, C, H, W)
            fet_pos = tf.transpose(fet_pos, [0, 2, 3, 1])                             #(N, H, W, C)
            print_activations(fet_pos)
        return fet_pos
    
    def chn_group_bn1(tensor_in, layer):
        with tf.variable_scope('chn_group_bn1_'+str(layer), reuse=reuse) as scope:
            #提取通道特征
            x_shape = get_shape(tensor_in)                                            #[N, H, W, C]
            pra_num = x_shape[0] * x_shape[1] * x_shape[2]                            # N*H*W
            fet_chn = tf.reshape(tensor_in, [-1, x_shape[-1]])                        #(N*H*W, C)
            
            def dispatch(fet_chn):
                #fet_chn --> (C)
                fet_shp = x_shape[-1]                                                 #[C]
                fet_chn = tf.reshape(fet_chn, [1]+x_shape[-1]+[1])                    #(1, C, 1)
                for i in range(len(c_shape)):
                    chn_shp        = c_shape[i]
                    fet_shp        = fet_shp // chn_shp[0]                            #[C']
                    params['conv'] = {'number': chn_shp[1], 'shape': chn_shp[0], 'rate': 1, 'stride': chn_shp[0], \
                                      'padding': 'VALID', 'use_bias': False}
                    fet_chn        = conv2(fet_chn, i, params, mtrain)                #(C"', C', C")
                    fet_chn        = tf.transpose(fet_chn, [0, 2, 1])                 #(C"', C", C')把已关联特征放到下层继续关联剩下的
                    fet_chn        = tf.reshape(fet_chn, [-1, fet_shp, 1])            #(C"', C',  1)
                return fet_chn
            
            fet_chn = tf.map_fn(dispatch, fet_chn, dtype=tf.float32, parallel_iterations=pra_num, \
                                back_prop=True, swap_memory=True, infer_shape=True)   #(N*H*W, C"', 1, 1)
            
            fet_chn = tf.reshape(fet_chn, [-1, c_shp[1]])                             #(N*H*W, C"')C"'是通道特征，做BN时应该对之外的维度做
            fet_chn = batchnorm1(fet_chn, 1, params, mtrain)                          #(N*H*W, C"')
            shape   = [-1] + list(np.asarray(c_shape)[:, 1:].reshape(-1))             #还原通道维度
            fet_chn = tf.reshape(fet_chn, shape)                                      #(N*H*W, C)
            perm    = [0] + [x for x in range(1, 1+len(c_shape), 1)][::-1]
            fet_chn = tf.transpose(fet_chn, perm)                                     #(N*H*W, C)
            fet_chn = tf.reshape(fet_chn, [-1, x_shape[1], x_shape[2], c_shp[1]])     #(N, H, W, C)
            print_activations(fet_chn)
        return fet_chn
    
    with tf.variable_scope('group_bn1_'+str(layer), reuse=reuse) as scope:
        fet_pos0   = pos_group_bn1(tensor_in, 0)
        fet_chn0   = chn_group_bn1(tensor_in, 0)
        fet_pos1   = pos_group_bn1(fet_chn0,  1)
        fet_chn1   = chn_group_bn1(fet_pos0,  1)
        tensor_out = fet_pos1 + fet_chn1
        print_activations(tensor_out)
    return tensor_out


def group_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('group_bn_relu1_'+str(layer)) as scope:
        bn         = group_bn1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain) 
    return tensor_out


def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    output_shape = params['group_unit']['output_shape'] #(H', W', C')
    bottle_shape = params['group_unit']['bottle_shape'] #(H", W", C")
    filter_shape = params['group_unit']['filter_shape'] #(h, w, c)
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape = tensor_in.get_shape().as_list() #(N, H, W, C)
    x_shape = get_shape(tensor_in)            #(N, H, W, C)
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:

        if np.any(np.asarray(x_shape[1:]) != np.asarray(output_shape)): #深度可分离卷积!!!
            number         = output_shape[-1] // x_shape[-1]
            shape          = [3, 3]
            stride         = np.asarray(x_shape[1:3]) // np.asarray(output_shape[0:2])
            params['conv'] = {'number': number, 'shape': shape, 'rate': [1, 1], 'stride': stride, 'padding': 'SAME'}
            shortcut       = conv_bn3(tensor_in, 0, params, mtrain)
        else:
            shortcut       = tensor_in
        
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(tensor_in, 0, params, mtrain)
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(residual,  1, params, mtrain)
        params['group'] = {'output_shape': output_shape, 'filter_shape': filter_shape}
        residual        = group_bn1(residual,       0, params, mtrain)
        tensor_out      = relu1(shortcut+residual,  0, params, mtrain)
    return tensor_out
    
    
def group_block1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    block_setting = params['group_block']['block_setting']

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    tensor_out    = tensor_in
    out_list      = []
    for i, block in enumerate(block_setting):
        
        output_shape, bottle_shape, filter_shape, unit_number, unit_trainable = block
        params['com']['trainable'] = unit_trainable
        with tf.variable_scope('group_block1_'+str(layer)+'_'+str(i)) as scope:
            for j in range(unit_number):
                params['group_unit'] = {'output_shape':output_shape, 'bottle_shape':bottle_shape, 'filter_shape':filter_shape}
                tensor_out           = group_unit1(tensor_out, j, params, mtrain)
        out_list.append(tensor_out)
    return out_list














def group_bn1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    #保证∏(i=1->n)xi = d(d为原本的参数维度，比如空间的面积或通道的数量，从而进行分组，以进行局部全连接)，参数复杂度∑(i=1->n)(xi)^2
    #不需要rate，因为本卷积关联整个图像
    #不需要pad， 因为本卷积只利用图像中的有效像素
    #stride不由卷积控制，而由avg_pool控制，先做avg_pool，再做relu
    #batchnorm只针对batch做，不针对通道或空间，因为通道和空间本身都是特征
    #当面积或通道数较小时，没有必要分块进行全连接
    
    output_shape = params['group']['output_shape'] #[H', W', C']
    filter_shape = params['group']['filter_shape'] #[h, w, c]    [8, 8, 64]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape= tensor_in.get_shape().as_list()   #(N, H, W, C)
    x_shape = get_shape(tensor_in)
    
    #降维时刚进入堆叠层时就降，升维时最终输出堆叠层时才升，以节省内存，在信息量不过损失的情况下，保持bottleneck连接
    out_in  = np.asarray(output_shape) / np.asarray(x_shape[1:])
    assert (out_in[0]<=1 and out_in[1]<=1) or (out_in[0]>1 and out_in[1]>1), 'The space shape of output is wrong!'
    
    #保证∏(i=1->n)xi = d(d为原本的参数维度，比如空间的面积或通道的数量，从而进行分组，以进行局部全连接)，参数复杂度∑(i=1->n)(xi)^2
    #空间关联次数p_num
    p_add = False
    p_num = np.log(np.asarray(x_shape[1:3])) // np.log(np.asarray(filter_shape[0:2]))
    p_num = p_num.astype(dtype=np.int32, copy=False)
    assert p_num[0] == p_num[1], 'The space shape of filter is wrong!'
    p_num = p_num[0]
    p_shp = np.power(np.asarray(filter_shape[0:2]), p_num)
    p_shp = p_shp.astype(dtype=np.int32, copy=False)
    p_res = np.asarray(x_shape[1:3]) // p_shp
    p_shp = p_shp * p_res
    assert np.all(p_shp == np.asarray(x_shape[1:3])), 'The space shape of filter is wrong!'
    if np.any(p_res != np.array([1, 1])) or np.all(np.asarray(x_shape[1:3]) == np.array([1, 1])):
        p_num = p_num + 1
        p_add = True
    
    #通道关联次数c_num
    c_add = False
    c_num = np.log(x_shape[-1]) // np.log(filter_shape[-1])
    c_num = c_num.astype(dtype=np.int32, copy=False)
    c_shp = np.power(filter_shape[-1], c_num)
    c_shp = c_shp.astype(dtype=np.int32, copy=False)
    c_res = x_shape[-1] // c_shp
    c_shp = c_shp * c_res
    assert c_shp == x_shape[-1], 'The channel shape of filter is wrong!'
    if c_res != 1 or x_shape[-1] == 1:
        c_num = c_num + 1
        c_add = True
    
    #空间关联参数，本着pos_shp0和pos_shp1差异尽可能小的原则，这样参数利用率高
    p_shape = [] #[[8, 8, p1, p2]]*p_number，只针对空间，8*8 × p1*p2 的全连接共p_number次，使用conv2d
    for i in range(p_num):
        pos_shp = []
        #若面积增大且有p_res，则将其放到最后一层；若面积减小或相等且有p_res，则将其放到第一层
        #out_in[0]<=1 or out_in[0]>1肯定会发生，i==0和i==p_num-1也肯定会经过，p_res存在的话肯定会得到处理
        pos_shp0 = np.asarray(filter_shape[0:2])
        if (i == 0 and out_in[0] <= 1) or (i == p_num - 1 and out_in[0] > 1):
            if p_add:
                pos_shp0 = p_res
            pos_shp1 = pos_shp0 * out_in[0:2]
            pos_shp1 = pos_shp1.astype(dtype=np.int32, copy=False)
            assert np.all(pos_shp1 / pos_shp0 == out_in[0:2]), 'The space shape of output is wrong!'
        else:
            pos_shp1 = pos_shp0
        pos_shp.extend(list(pos_shp0))
        pos_shp.extend(list(pos_shp1))
        p_shape.append(pos_shp)
    
    #通道关联参数，本着chn_shp0和chn_shp1差异尽可能小的原则，这样参数利用率高
    c_shape = [] #[[16, c]]*c_number，只针对通道，16  × c 的全连接共c_number次，使用conv1d
    for i in range(c_num):
        chn_shp = []
        #若通道增多且有c_res，则将其放到最后一层；若通道减少或相等且有c_res，则将其放到第一层
        #out_in[-1]<=1 or out_in[-1]>1肯定会发生，i==0和i==c_num-1也肯定会经过，c_res存在的话肯定会得到处理
        chn_shp0 = filter_shape[-1]
        if (i == 0 and out_in[-1] <= 1) or (i == c_num -1 and out_in[-1] > 1):
            if c_add:
                chn_shp0 = c_res
            chn_shp1 = chn_shp0 * out_in[-1]
            chn_shp1 = chn_shp1.astype(dtype=np.int32, copy=False)
            assert chn_shp1 / chn_shp0 == out_in[-1], 'The channel shape of output is wrong!'
        else:
            chn_shp1 = chn_shp0
        chn_shp.append(chn_shp0)
        chn_shp.append(chn_shp1)
        c_shape.append(chn_shp)
    
    p_shp   = np.prod(p_shape, axis=0)
    p_srd   = p_shp[0:2] // p_shp[2:]
    assert np.all(p_shp[0:2] == np.asarray(x_shape[1:3])), 'The p_shape is wrong!'
    
    c_shp   = np.prod(c_shape, axis=0)
    c_srd   = c_shp[0] // c_shp[1]
    assert c_shp[0] == x_shape[-1], 'The c_shape is wrong!'
    
    def pos_group_bn1(tensor_in, layer):
        with tf.variable_scope('pos_group_bn1_'+str(layer), reuse=reuse) as scope:
            #提取空间特征(不去管通道)
            x_shape = get_shape(tensor_in)
            fet_pos = tf.transpose(tensor_in, [0, 3, 1, 2])                           #(N, C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[1], x_shape[2], 1])            #(N*C, H, W, 1)
            fet_shp = np.asarray(x_shape[1:3])
            for i in range(len(p_shape)):
                pos_shp        = p_shape[i]
                fet_shp        = fet_shp // np.asarray(pos_shp[0:2])
                params['conv'] =  {'number': pos_shp[2]*pos_shp[3], 'shape': pos_shp[0:2], 'rate': 1, 'stride': pos_shp[0:2], \
                                   'padding': 'VALID', 'use_bias': False}
                fet_pos        = conv1(fet_pos, i, params, mtrain)                    #(N*C, H', W', C")
                fet_pos        = tf.transpose(fet_pos, [0, 3, 1, 2])                  #(N*C, C", H', W')
                fet_pos        = tf.reshape(fet_pos, [-1, fet_shp[0], fet_shp[1], 1]) #(N*C*C", H', W', 1)
            fet_pos = tf.reshape(fet_pos, [-1, p_shp[2]*p_shp[3]])                    #(N*C, C"')C"'是空间特征，做BN时应该对之外的维度做
            fet_pos = batchnorm1(fet_pos, 0, params, mtrain)                          #(N*C, C"')
            shape   = [-1] + list(np.asarray(p_shape)[:, 2:].reshape(-1))             #还原空间维度
            fet_pos = tf.reshape(fet_pos, shape)                                      #(N*C, H, W)
            perm    = [0] + [x for x in range(1, 1+2*len(p_shape), 2)][::-1] + [x for x in range(2, 2+2*len(p_shape), 2)][::-1]
            fet_pos = tf.transpose(fet_pos, perm)                                     #(N*C, H, W)
            fet_pos = tf.reshape(fet_pos, [-1, x_shape[-1], p_shp[2], p_shp[3]])      #(N, C, H, W)
            fet_pos = tf.transpose(fet_pos, [0, 2, 3, 1])                             #(N, H, W, C)
            print_activations(fet_pos)
        return fet_pos
    
    def chn_group_bn1(tensor_in, layer):
        with tf.variable_scope('chn_group_bn1_'+str(layer), reuse=reuse) as scope:
            #提取通道特征
            x_shape = get_shape(tensor_in)
            fet_chn = tf.reshape(tensor_in, [-1, x_shape[-1], 1])                     #(N*H*W, C, 1)
            fet_shp = x_shape[-1]
            for i in range(len(c_shape)):
                chn_shp        = c_shape[i]
                fet_shp        = fet_shp // chn_shp[0]
                params['conv'] = {'number': chn_shp[1], 'shape': chn_shp[0], 'rate': 1, 'stride': chn_shp[0], \
                                  'padding': 'VALID', 'use_bias': False}
                fet_chn        = conv2(fet_chn, i, params, mtrain)                    #(N*H*W, C', C")
                fet_chn        = tf.transpose(fet_chn, [0, 2, 1])                     #(N*H*W, C", C')把已关联特征放到下层，继续关联剩下的
                fet_chn        = tf.reshape(fet_chn, [-1, fet_shp, 1])                #(N*H*W*C", C', 1)
            fet_chn = tf.reshape(fet_chn, [-1, c_shp[1]])                             #(N*H*W, C"')C"'是通道特征，做BN时应该对之外的维度做
            fet_chn = batchnorm1(fet_chn, 1, params, mtrain)                          #(N*H*W, C"')
            shape   = [-1] + list(np.asarray(c_shape)[:, 1:].reshape(-1))             #还原通道维度
            fet_chn = tf.reshape(fet_chn, shape)                                      #(N*H*W, C)
            perm    = [0] + [x for x in range(1, 1+len(c_shape), 1)][::-1]
            fet_chn = tf.transpose(fet_chn, perm)                                     #(N*H*W, C)
            fet_chn = tf.reshape(fet_chn, [-1, x_shape[1], x_shape[2], c_shp[1]])     #(N, H, W, C")
            print_activations(fet_chn)
        return fet_chn
    
    with tf.variable_scope('group_bn1_'+str(layer), reuse=reuse) as scope:
        fet_pos0   = pos_group_bn1(tensor_in, 0)
        fet_chn0   = chn_group_bn1(tensor_in, 0)
        fet_pos1   = pos_group_bn1(fet_chn0,  1)
        fet_chn1   = chn_group_bn1(fet_pos0,  1)
        tensor_out = fet_pos1 + fet_chn1
        print_activations(tensor_out)
    return tensor_out


def group_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in  = tensor_in[0]
    
    with tf.variable_scope('group_bn_relu1_'+str(layer)) as scope:
        bn         = group_bn1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain) 
    return tensor_out


def group_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    output_shape = params['group_unit']['output_shape'] #(H', W', C')
    bottle_shape = params['group_unit']['bottle_shape'] #(H", W", C")
    filter_shape = params['group_unit']['filter_shape'] #(h, w, c)
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape = tensor_in.get_shape().as_list() #(N, H, W, C)
    x_shape = get_shape(tensor_in)            #(N, H, W, C)
    
    with tf.variable_scope('group_unit1_'+str(layer)) as scope:

        if np.any(np.asarray(x_shape[1:]) != np.asarray(output_shape)): #深度可分离卷积!!!
            number         = output_shape[-1] // x_shape[-1]
            shape          = [3, 3]
            stride         = np.asarray(x_shape[1:3]) // np.asarray(output_shape[0:2])
            params['conv'] = {'number': number, 'shape': shape, 'rate': [1, 1], 'stride': stride, 'padding': 'SAME'}
            shortcut       = conv_bn3(tensor_in, 0, params, mtrain)
        else:
            shortcut       = tensor_in
        
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(tensor_in, 0, params, mtrain)
        params['group'] = {'output_shape': bottle_shape, 'filter_shape': filter_shape}
        residual        = group_bn_relu1(residual,  1, params, mtrain)
        params['group'] = {'output_shape': output_shape, 'filter_shape': filter_shape}
        residual        = group_bn1(residual,       0, params, mtrain)
        tensor_out      = relu1(shortcut+residual,  0, params, mtrain)
    return tensor_out
    
    
def group_block1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    block_setting = params['group_block']['block_setting']

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    tensor_out    = tensor_in
    out_list      = []
    for i, block in enumerate(block_setting):
        
        output_shape, bottle_shape, filter_shape, unit_number, unit_trainable = block
        params['com']['trainable'] = unit_trainable
        with tf.variable_scope('group_block1_'+str(layer)+'_'+str(i)) as scope:
            for j in range(unit_number):
                params['group_unit'] = {'output_shape':output_shape, 'bottle_shape':bottle_shape, 'filter_shape':filter_shape}
                tensor_out           = group_unit1(tensor_out, j, params, mtrain)
        out_list.append(tensor_out)
    return out_list
































def attent_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    reg       = params['com']['reg']
    wscale    = params['com']['wscale']
    dtype     = params['com']['dtype']
    reuse     = params['com']['reuse']
    is_train  = params['com']['is_train']
    trainable = params['com']['trainable']
    
    shape     = params['attent_unit']['shape']
    rate      = params['attent_unit']['rate']
    depth_key = params['attent_unit']['depth_key']
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape     = tensor_in.get_shape().as_list()
    depth_input = x_shape[-1]
    height      = x_shape[ 1]
    width       = x_shape[ 2]
    
    with tf.variable_scope('attent_unit1_'+str(layer)) as scope:
    
        #使用3x3conv隔离特征
        params['conv'] = {'number':depth_input, 'shape':[3, 3], 'rate':1, 'stride':[1, 1], 'padding':'SAME'}
        tensor_in      = conv_bn_relu1(tensor_in, 0, params, mtrain)
        tensor_in      = conv_bn_relu1(tensor_in, 1, params, mtrain)
        #对keys的关联应该在放入位置向量之前，位置向量主要服务于关联中心点，对中心特征向量和其之外的特征向量之间的位置关系做描述
        params['conv'] = {'number':depth_key,   'shape':[1, 1], 'rate':1, 'stride':[1, 1], 'padding':'VALID', 'use_bias': True}
        tensor_key     = conv1(tensor_in, 0, params, mtrain)
        
        
        params['conv'] = {'number':depth_input, 'shape':[1, 1], 'rate':1, 'stride':[1, 1], 'padding':'VALID', 'use_bias': True}
        tensor_value   = conv1(tensor_in, 0, params, mtrain)
        
        #获取relative_position_embeddings #(64, 64, 64)
        PE = tf.get_variable(name='PE', shape=shape+depth_key, dtype=dtype, \
                             #initializer=tf.truncated_normal_initializer(stddev=wscale), \
                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                             regularizer=tf.contrib.layers.l2_regularizer(reg), trainable=trainable)
        
        tensor_out = tf.TensorArray(dtype=tf.float32, size=height*width, dynamic_size=False, clear_after_read=True, \
                                    infer_shape=True, element_shape=[depth_input+depth_key], colocate_with_first_write_call=True)
        
        def cond(i, tensor_out):
            c = tf.less(i, height*width)
            return c

        def body(i, tensor_out):
            
            ycd   = i // width
            xcd   = i  % width
            ymn   = ycd - ((shape[0] - 1) // 2) * rate
            xmn   = xcd - ((shape[1] - 1) // 2) * rate
            ycds  = tf.concat([[ymn], tf.tile([rate], [shape[0]-1])], axis=0)
            xcds  = tf.concat([[xmn], tf.tile([rate], [shape[1]-1])], axis=0)
            ycds  = tf.cumsum(ycds, axis=0, exclusive=False, reverse=False)
            xcds  = tf.cumsum(xcds, axis=0, exclusive=False, reverse=False)
            idxs  = tf.where(tf.logical_and(ycds>=0, ycds<height))
            ycds  = tf.gather_nd(ycds, idxs)
            idxs  = tf.where(tf.logical_and(xcds>=0, xcds<width ))
            xcds  = tf.gather_nd(xcds, idxs)
            ycds  = tf.tile(ycds[:, tf.newaxis], [1, tf.shape(xcds)[0]])
            xcds  = tf.tile(xcds[tf.newaxis, :], [tf.shape(ycds)[0], 1])
            crd0  = tf.stack([ycd, xcd], axis=0)                                 #(2)         实际中心
            crds0 = tf.concat([ycds, xcds], axis=-1)                             #(h, w, 2)   实际坐标
            fets0 = tf.gather_nd(tensor_value, crds0)                            #(h, w, c)   实际特征
            fets3 = tf.gather_nd(tensor_key,   crds0)                            #(h, w, c')  实际特征
            crd1  = (shape - 1) // 2                                             #(2)         相对中心
            crds1 = (crds0 - crd0) // rate                                       #(h, w, 2)   相对坐标
            crds1 = crds1 + crd1                                                 #(h, w, 2)   相对坐标
            fets1 = tf.gather_nd(PE, crds1)                                      #(h, w, c)   相对特征
            #fets2= tf.concat([fets0, fets1], axis=-1)                           #(h, w, c'') 融合特征
            crd3  = crd0 - crds0[0, 0]         #crd、crds下标换成1也一样           #(2)         相对坐标
            fet3  = tf.gather_nd(fets3, crd3)                                    #(c')        相对中心
            #计算注意力
            att3  = tf.einsum('ijk,k->ij', fets3, fet3)                          #(h, w)
            att3  = tf.exp(att3 / tf.sqrt(depth_key))                            #(h, w)
            att3  = att3 / tf.reduce_sum(att3)                                   #(h, w)
            fet0  = tf.einsum('ij,ijk->k', att3, fets0)                          #(c)
            fet1  = tf.einsum('ij,ijk->k', att3, fets1)                          #(c')
            fet2  = tf.concat([fet0, fet1], axis=-1)                             #(c'')
            #fet2 = tf.einsum('ij,ijk->k', att3, fets2)                          #(c'')
            tensor_out = tensor_out.write(i, fet2)                               #(h, w, c')
            return [i+1, tensor_out]
        
        i = tf.constant(0)
        [i, tensor_out] = tf.while_loop(cond, body, loop_vars=[i, tensor_out], shape_invariants=None, \
                                        parallel_iterations=128, back_prop=True, swap_memory=True)
        #使用1x1conv进行特征和位置向量的融合
        params['conv']  = {'number':depth_input, 'shape':[1, 1], 'rate':1, 'stride':[1, 1], 'padding':'VALID'}
        tensor_out      = conv_bn1(tensor_in, 0, params, mtrain)
        tensor_out      = relu1(tensor_out + tensor_in)
        return tensor_out
        
        
def atten1(tensor_in=None, layer=0, params=None, mtrain=None):
    

    
    shape        = params['atten']['shape']        #attention关联的范围，比如[64, 64]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape      = get_shape(tensor_in)
    depth_input  = x_shape[-1]
    depth_bottle = depth_input // 4
    
    with tf.variable_scope('atten1_'+str(layer), reuse=reuse) as scope:
        #使用1x1conv降维
        params['conv'] = {'number':depth_bottle, 'shape':[1, 1], 'rate':1, 'stride':[1, 1], 'padding':'VALID'}
        fet_com        = conv_bn_relu1(tensor_in, 0, params, mtrain)


            
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    #x_shape = tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    kernel_shape  = [shape[0], shape[1], x_shape[3], number]
    kernel_stride = [1, stride[0], stride[1], 1]
    
    with tf.variable_scope('conv1_'+str(layer), reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=kernel_shape, dtype=dtype, \
                                 #initializer=tf.truncated_normal_initializer(stddev=wscale), \
                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                 #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
                                 regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                 trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name='biases', shape=[number], dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        if rate == 1:
            conv = tf.nn.conv2d(tensor_in, kernel, kernel_stride, padding=padding)
        else:
            conv = tf.nn.atrous_conv2d(tensor_in, kernel, rate, padding=padding)
        if use_bias:
            tensor_out = tf.nn.bias_add(conv, biases)
        else:
            tensor_out = conv
        #tf.summary.histogram('conv', tensor_out)
        print_activations(tensor_out)
    return tensor_out 