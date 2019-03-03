import tensorflow as tf
import tensorflow.contrib.layers as tcl

class lenet:
	def __init__(self):
		pass
	def forward(self,x,enc=True,dec=False,phase=True,keep_prob=0.5,reuse=False,nmc=6):
		net=tf.identity(x)
		reg=tcl.l2_regularizer(5e-4)
		conv_init=tcl.variance_scaling_initializer()
		if enc:
			with tf.variable_scope('gen',reuse=reuse):
				net=tcl.conv2d(net,20,5,1,'VALID',activation_fn=None,weights_initializer=conv_init,weights_regularizer=reg)
				net=tcl.batch_norm(net,is_training=phase,scale=True)
				net=tf.nn.leaky_relu(net,0.01)
				net=tcl.max_pool2d(net,2,2,'VALID')
				
				net=tcl.conv2d(net,50,5,1,'VALID',activation_fn=None,weights_initializer=conv_init,weights_regularizer=reg)
				net=tcl.batch_norm(net,is_training=phase,scale=True)
				net=tf.nn.leaky_relu(net,0.01)
				net=tcl.max_pool2d(net,2,2,'VALID')
				
				net=tf.nn.dropout(net,keep_prob=keep_prob)
				net=tcl.flatten(net)

				net=tcl.fully_connected(net,500,activation_fn=None,weights_regularizer=reg)
				net=tcl.batch_norm(net,is_training=phase,scale=True)
				net=tf.nn.leaky_relu(net,0.01)
		if dec:
			with tf.variable_scope('class',reuse=reuse):
				net=tcl.fully_connected(net,nmc,activation_fn=None,weights_regularizer=reg)
			
		return net
			
			
		
		
		
