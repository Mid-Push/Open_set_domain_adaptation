import tensorflow as tf
import tensorflow.contrib.layers as tcl

class lenet:
	def __init__(self):
		pass
	def forward(self,x,enc=True,dec=False,phase=True,keep_prob=0.5,reuse=False,nmc=6):
		net=tf.identity(x)
		if enc:
			with tf.variable_scope('gen',reuse=reuse):
				for i in xrange(2):
					net=tcl.conv2d(net,64,5,1,'VALID',activation_fn=None)
					net=tcl.batch_norm(net,is_training=phase)
					net=tf.nn.leaky_relu(net)
				for i in xrange(2):
					net=tcl.conv2d(net,128,3,2,'VALID',activation_fn=None)
					net=tcl.batch_norm(net,is_training=phase)
					net=tf.nn.leaky_relu(net)
				for i in xrange(2):
					net=tcl.fully_connected(net,100,activation_fn=None)
					net=tcl.batch_norm(net,is_training=phase)
					net=tf.nn.leaky_relu(net)
				net=tcl.flatten(net)

		if dec:
			with tf.variable_scope('class',reuse=reuse):
				net=tcl.fully_connected(net,nmc,activation_fn=None)
			
		return net
			
			
		
		
		
