"""
Derived from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/
"""
import tensorflow as tf
import numpy as np
import math
def augment(x,domain='source'):
	if domain=='source':
		return tf.concat([x,x,tf.zeros_like(x)],axis=1)
	if domain=='target':
		return tf.concat([tf.zeros_like(x),x,-x],axis=1)

class LeNetModel(object):

    def __init__(self, num_classes=1000, is_training=True,image_size=28,dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
	self.default_image_size=image_size
        self.is_training=is_training
        self.num_channels=3
        self.mean=None
        self.bgr=False
        self.range=None

    def inference(self, x, domain='source',training=False):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 5, 5, 64, 1, 1, padding='VALID',bn=True, name='conv1')
        pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID', name='pool1')

	print 'conv1 ',conv1.get_shape()
	#print 'pool1 ',pool1.get_shape()
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(pool1, 5, 5, 64, 1, 1, padding='VALID',bn=True,name='conv2')
        pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name='pool2')
	print 'conv2 ',conv2.get_shape()
        '''
	conv3 = conv(conv2, 3, 3, 128, 2, 2, padding='VALID',bn=True,name='conv3')
	print 'conv3 ',conv3.get_shape()
	
	conv4 = conv(conv3, 3, 3, 128, 2, 2, padding='VALID',bn=True,name='conv4')
	print 'conv4 ',conv4.get_shape()
	'''


        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.contrib.layers.flatten(pool2)
        self.flattened=flattened
	fc1 = fc(flattened, flattened.get_shape()[-1], 500, bn=True,name='fc1')
	#fc1=augment(fc1,domain=domain)
	#fcmid=fc(fc1,100,100,bn=True,name='fcmid')
	fc2 = fc(fc1, 500, self.num_classes, relu=False,name='fc2')
	self.score=fc2
        return fc2
    def adoptimize(self,learning_rate,train_layers=[]):
        var_list=[v for v in tf.trainable_variables() if 'D' in v.name]
	D_weights=[v for v in var_list if 'weights' in v.name]
	D_biases=[v for v in var_list if 'biases' in v.name]
	print '=================Discriminator_weights====================='
	print D_weights
	print '=================Discriminator_biases====================='
	print D_biases
	
	self.Dregloss=0.0005*tf.reduce_mean([tf.nn.l2_loss(v) for v in var_list if 'weights' in v.name])
        D_op1 = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(self.D_loss+self.Dregloss, var_list=D_weights)
        D_op2 = tf.train.MomentumOptimizer(learning_rate*2.0,0.9).minimize(self.D_loss+self.Dregloss, var_list=D_biases)
        D_op=tf.group(D_op1,D_op2)
	#R_op=tf.train.MomentumOptimizer(learning_rate,0.9).minimize(self.Entropyloss)
	#----------------------------This is  adversarial Training Again ---------------------------------------
	return D_op
    def corrupt(self,x,noisetype='Gaussian'):
	if noisetype=='Gaussian':
		noise=tf.random_normal(shape=tf.shape(x),mean=0.0,stddev=0.1)
		return x+noise
	if noisetype=='impulse':
		noise=x*tf.keras.backend.random_binomial(tf.shape(x),p=0.5)	
		return noise
    def wganloss(self,x,xt,batch_size,lam=10.0):
        with tf.variable_scope('reuse_inference') as scope:
	    scope.reuse_variables()
            self.inference(x,training=True)
	    source_fc6=self.fc6
	    source_fc7=self.fc7
	    source_fc8=self.fc8
            source_softmax=self.output
	    source_output=outer(source_fc7,source_softmax)
            print 'SOURCE_OUTPUT: ',source_output.get_shape()
	    scope.reuse_variables()
            self.inference(xt,training=True)
	    target_fc6=self.fc6
	    target_fc7=self.fc7
	    target_fc8=self.fc8
            target_softmax=self.output
	    target_output=outer(target_fc7,target_softmax)
            print 'TARGET_OUTPUT: ',target_output.get_shape()
        with tf.variable_scope('reuse') as scope:
	    target_logits,_=D(target_fc8)
	    scope.reuse_variables()
	    source_logits,_=D(source_fc8)
	    eps=tf.random_uniform([batch_size,1],minval=0.0,maxval=1.0)
	    X_inter=eps*source_fc8+(1-eps)*target_fc8
	    grad = tf.gradients(D(X_inter), [X_inter])[0]
	    grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
	    grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)
	    D_loss=tf.reduce_mean(target_logits)-tf.reduce_mean(source_logits)+grad_pen
	    G_loss=tf.reduce_mean(source_logits)-tf.reduce_mean(target_logits)	
	    self.G_loss=G_loss
	    self.D_loss=D_loss
	    self.D_loss=0.3*self.D_loss
	    self.G_loss=0.3*self.G_loss
	    return G_loss,D_loss
    def adloss(self,x,xt,y):
        with tf.variable_scope('reuse_inference') as scope:
	    scope.reuse_variables()
	    #self.inference(x,training=True)
	    #hxs=source_feature=self.sourcefeature
            #scope.reuse_variables()
	    hxt=self.inference(xt,training=True)
	    #target_feature=self.targetfeature
	    #zt=self.latentfeature
	    #target_entfeature=self.entfeature
        
	'''
	with tf.variable_scope('reuse') as scope:
            source_logits,source_pred=D(source_feature)
            scope.reuse_variables()
            target_logits,target_pred=D(target_feature)
	'''
	
	#-------------------------construct L_adv(xt) using "open set domain adaptation by backpropagation"---------------------------------------------
	p_kone=tf.gather(tf.nn.softmax(hxt),indices=[self.num_classes-1],axis=1)
	print 'p(y=K+1) ', p_kone.get_shape()

	L_adv=0.5*tf.reduce_mean(tf.log(p_kone))+0.5*tf.reduce_mean((tf.log(1.0-(p_kone))))
	self.L_adv=L_adv
	
	'''
	self.Semanticloss=tf.constant(0.0)
	self.Drloss=tf.constant(0.0)
	#----------------------------Farthest Neighbor Resconstruction-----------------------
	#self.Entropyloss=tf.reduce_mean(tf.pow(xt_hat-corrupt_xt,2.))
	#self.Entropyloss=0.4*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=target_entfeature,labels=tf.nn.softmax(target_entfeature)))
	#-0.1*tf.reduce_mean(tf.pow(xs_hat-xs,2.))	
	self.Entropyloss=tf.constant(0.0)
	
	D_real_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits,labels=tf.ones_like(target_logits)))
        D_fake_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=source_logits,labels=tf.zeros_like(source_logits)))
        self.D_loss=D_real_loss+D_fake_loss
        #self.G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits,labels=tf.ones_like(target_logits)))
	#self.Q_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=source_logits,labels=tf.ones_like(target_logits)))
        self.G_loss=-self.D_loss
	tf.summary.scalar('G_loss',self.G_loss)
	tf.summary.scalar('JSD',self.G_loss/2+math.log(2))
	'''
        #self.G_loss=0.1*self.G_loss
	#self.D_loss=0.1*self.D_loss
    def loss(self, batch_x, batch_y=None):
        with tf.variable_scope('reuse_inference') as scope:
	    y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
	tf.summary.scalar('Closs',self.loss)
        return self.loss

    def optimize(self, learning_rate, train_layers,global_step):
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	print train_layers
	g_list=[v for v in tf.trainable_variables() if v.name.split('/')[1] in ['conv1','conv2','conv3','conv4','fc1','fcmid']]
	c_list=[v for v in tf.trainable_variables() if v.name.split('/')[1] in ['fc2']]
	self.Gregloss=0.0005*tf.reduce_mean([tf.nn.l2_loss(x) for x in g_list+c_list if 'weights' in x.name])
	
	g_weights=[v for v in g_list if 'weights' in v.name or 'gamma' in v.name]
	g_biases=[v for v in g_list if 'biases' in v.name or 'beta' in v.name]
	
	c_weights=[v for v in c_list if 'weights' in v.name or 'gamma' in v.name]
	c_biases=[v for v in c_list if 'biases' in v.name or 'beta' in v.name]

	
	print '==============g_weights= ',len(g_weights),'======================='
	print g_weights
	print '==============g_biases= ',len(g_biases),' ======================='
	print g_biases
	print '==============c_weights= ',len(c_weights),'======================='
	print c_weights
	print '==============c_biases= ',len(c_biases),' ======================='
	print c_biases
	
	#self.F_loss=self.Entropyloss
        Ls_loss=self.loss 
	L_adv=self.L_adv
	self.c_loss=Ls_loss+L_adv
	#+self.Gregloss
	self.g_loss=Ls_loss-L_adv
	#+self.Gregloss
	
	tf.summary.scalar('C_loss',self.c_loss)
	tf.summary.scalar('G_adv',self.g_loss)
	tf.summary.scalar('L_adv',self.L_adv)	
	
	#+global_step*self.G_loss+self.Entropyloss
	#+global_step*self.G_loss+self.Gregloss+global_step*self.Entropyloss
	#+global_step*self.Entropyloss+global_step*self.G_loss
        train_op1=tf.train.AdamOptimizer(learning_rate*1.0,0.9).minimize(self.c_loss, var_list=c_weights)
        train_op2=tf.train.AdamOptimizer(learning_rate*2.0,0.9).minimize(self.c_loss, var_list=c_biases)
        
	train_op3=tf.train.AdamOptimizer(learning_rate*1.0,0.9).minimize(self.g_loss, var_list=g_weights)
        train_op4=tf.train.AdamOptimizer(learning_rate*2.0,0.9).minimize(self.g_loss, var_list=g_biases)
		
	
        #train_op1=tf.train.AdamOptimizer(learning_rate*0.1,beta1=0.5,beta2=0.99).minimize(self.F_loss, var_list=finetune_list)
        #train_op2=tf.train.AdamOptimizer(learning_rate*1.0,beta1=0.5,beta2=0.99).minimize(self.F_loss, var_list=new_list)
	train_op=tf.group(train_op1,train_op2,train_op3,train_op4)
	return train_op
    def load_original_weights(self, session, skip_layers=[]):
        weights_dict = np.load('bvlc_alexnet.npy', encoding='bytes').item()

        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue

            if op_name == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope('reuse_inference/'+op_name, reuse=True):
	        print '=============================OP_NAME  ========================================'
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases')
	        	print op_name,var
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights')
	        	print op_name,var
                        session.run(var.assign(data))


"""
Helper methods
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, bn=False,padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters],initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
	if bn==True:
	    bias=tf.contrib.layers.batch_norm(bias,scale=True)
        #relu = tf.nn.relu(bias, name=scope.name)
	relu=leaky_relu(bias)
        return relu
def D(x):
    with tf.variable_scope('D'):
        num_units_in=int(x.get_shape()[-1])
        num_units_out=1
        weights = tf.get_variable('weights',initializer=tf.truncated_normal([num_units_in,1024],stddev=0.01))
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.zeros_initializer())
        hx=(tf.matmul(x,weights)+biases)
	ax=tf.nn.dropout(tf.nn.relu(hx),0.5)
	        

	weights2 = tf.get_variable('weights2',initializer=tf.truncated_normal([1024,1024],stddev=0.01))
        biases2 = tf.get_variable('biases2', shape=[1024], initializer=tf.zeros_initializer())
        hx2=(tf.matmul(ax,weights2)+biases2)
	ax2=tf.nn.dropout(tf.nn.relu(hx2),0.5)
	
	weights3 = tf.get_variable('weights3', initializer=tf.truncated_normal([1024,num_units_out],stddev=0.3))
        biases3 = tf.get_variable('biases3', shape=[num_units_out], initializer=tf.zeros_initializer())
        hx3=tf.matmul(ax2,weights3)+biases3
        return hx3,tf.nn.sigmoid(hx3)

def fc(x, num_in, num_out, name, relu=True,bn=False,bias=True,stddev=0.01):
    with tf.variable_scope(name) as scope:
        #weights = tf.get_variable('weights', initializer=tf.truncated_normal([num_in,num_out],stddev=stddev))
        weights = tf.get_variable('weights', shape=[num_in,num_out],initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[num_out]))
        act = tf.matmul(x, weights,name=scope.name)
	if bias==True:
	    act=act+biases
	if bn==True:
	    act=tf.contrib.layers.batch_norm(act,scale=True)
        if relu == True:
            relu = leaky_relu(act)
            return relu
        else:
            return act
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def outer(a,b):
        a=tf.reshape(a,[-1,a.get_shape()[-1],1])
        b=tf.reshape(b,[-1,1,b.get_shape()[-1]])
        c=a*b
        return tf.contrib.layers.flatten(c)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1],
                          padding = padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
