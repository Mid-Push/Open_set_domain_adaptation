import tensorflow as tf
import tensorflow.contrib.layers as tcl
from nns import small,large
class DALearner:
	
	def __init__(self,name='small',source='mnist',target='usps',num_classes=6):
		if name=='small':
			self.default_image_size=28
			self.num_channels=1
			self.model=small.lenet()
		else:
			self.default_image_size=32
			self.num_channels=3
			self.model=large.lenet()
		self.num_classes=num_classes
		self.source=source
		self.target=target
		self.mean=None
		self.bgr=None
		self.range=None
	
	def loss(self,xs,ys,xt,phase=True,keep_prob=0.5,lamb=1.0):
		model=self.model
		
		src_e=model.forward(xs,enc=True,dec=False,reuse=False,phase=phase,keep_prob=keep_prob,nmc=self.num_classes)
		print src_e.get_shape()
		src_p=model.forward(src_e,enc=False,dec=True,reuse=False,phase=phase,keep_prob=keep_prob,nmc=self.num_classes)
		
		trg_e=model.forward(xt,enc=True,dec=False,reuse=True,phase=phase,keep_prob=keep_prob,nmc=self.num_classes)
		trg_p=model.forward(trg_e,enc=False,dec=True,reuse=True,phase=phase,keep_prob=keep_prob,nmc=self.num_classes)

		#----source classification loss------
		loss_src=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=src_p,labels=ys))	
		tf.summary.scalar('class/loss_src',loss_src)		

		#-------------------------construct L_adv(xt) using "open set domain adaptation by backpropagation"---------------------------------------------
		p_kone=tf.gather(tf.nn.softmax(trg_p),indices=[self.num_classes-1],axis=1)
		loss_adv=-0.5*tf.reduce_mean(tf.log(p_kone+1e-8))-0.5*tf.reduce_mean((tf.log(1.0-(p_kone)+1e-8)))
		tf.summary.scalar('adv/loss_trg_adv',loss_adv)
		loss_class=(
			    loss_src
			    +loss_adv
			    )
		loss_gen=(
			    loss_src
			    -lamb*loss_adv
			    )
		return loss_class,loss_gen,src_p,trg_p
		

			
				
		
