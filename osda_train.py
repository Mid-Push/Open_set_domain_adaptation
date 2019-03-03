import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import DALearner
from utils import utils
from utils import metrics
from preprocessing.preprocessing import preprocessing

import math

tf.app.flags.DEFINE_float('lr', '1e-3', 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('net','small', '[small,large]')
tf.app.flags.DEFINE_string('opt','mom', '[adam,mom]')
tf.app.flags.DEFINE_string('train','mnist', '[mnist,usps,svshn]')
tf.app.flags.DEFINE_string('test','usps', '[mnist,usps,svshn]')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')

#-------------------------open set domain adaptation----------------------------------------
NUM_CLASSES = 6
FLAGS = tf.app.flags.FLAGS

TRAIN_FILE=FLAGS.train
TEST_FILE=FLAGS.test

print TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE
print TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE
print TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE

TRAIN=utils.get_data(FLAGS.train,split='train',unk=False,shuffle=True,frange=[0.,1.])
VALID=utils.get_data(FLAGS.test,split='train',unk=True,shuffle=True,frange=[0.,1.])
TEST=utils.get_data(FLAGS.test,split='train',unk=True,shuffle=False,frange=[0.,1.])

def adaptation_factor(x):
	den=1.0+math.exp(-10*x)
	lamb=2.0/den-1.0
	return min(lamb,1.0)

def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    dropout_keep_prob = tf.placeholder(tf.float32)
    revgrad_lamb = tf.placeholder(tf.float32)
    is_training=tf.placeholder(tf.bool)    

    # Model
    model =DALearner(name=FLAGS.net,num_classes=NUM_CLASSES,source=FLAGS.train,target=FLAGS.test)
    # Placeholders
    x_s = tf.placeholder(tf.float32, [None]+TRAIN.image_shape,name='x')
    x_t = tf.placeholder(tf.float32, [None]+TEST.image_shape,name='xt')
    x=preprocessing(x_s,model)
    xt=preprocessing(x_t,model)
    tf.summary.image('Source Images',x)
    tf.summary.image('Target Images',xt)
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES],name='y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES],name='yt')
    loss_class,loss_gen,src_p,trg_p= model.loss(x, y, xt, keep_prob=dropout_keep_prob,phase=is_training,lamb=revgrad_lamb)
     
    #---- Optimizers--------- 
    main_vars=tf.trainable_variables()
    gen_vars=[var for var in main_vars if 'gen' in var.name]
    class_vars=[var for var in main_vars if 'class' in var.name]
    print gen_vars
    print class_vars
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gen_op=utils.get_optimizer(FLAGS.opt,FLAGS.lr,loss_gen,gen_vars) 
        class_op=utils.get_optimizer(FLAGS.opt,FLAGS.lr,loss_class,class_vars) 
    optimizer=tf.group(gen_op,class_op)

    #------------ A series of metrics for evaluation: OS,OS*,ALL,UNK----------------------------------
    target_predict=trg_p
    with tf.variable_scope('metrics') as scope:
    	os_acc,os_update_op=metrics.OS(hx=target_predict,y=yt,num_classes=NUM_CLASSES)
    	osstar_acc,osstar_update_op=metrics.OS_star(hx=target_predict,y=yt,num_classes=NUM_CLASSES)
    	all_acc,all_update_op=metrics.ALL(hx=target_predict,y=yt)
    	unk_acc,unk_update_op=metrics.UNK(hx=target_predict,y=yt,num_classes=NUM_CLASSES)
    	metrics_update_op=tf.group(os_update_op,osstar_update_op,all_update_op,unk_update_op)
    	metrics_variables=[v for v in tf.local_variables() if v.name.startswith('metrics')]
	reset_ops=[v.initializer for v in metrics_variables]
	print metrics_variables
	

    train_writer=tf.summary.FileWriter('./log/tensorboard')
    train_writer.add_graph(tf.get_default_graph())
    merged=tf.summary.merge_all()




    print '============================GLOBAL TRAINABLE VARIABLES ============================'
    print tf.trainable_variables(),'   ',len(tf.trainable_variables())
    #print '============================GLOBAL VARIABLES ======================================'
    #print tf.global_variables()
	
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
	saver=tf.train.Saver()
	train_writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.datetime.now()))
	for step in range(200*600):
            # Start training
	    batch_xs, batch_ys = TRAIN.next_batch(FLAGS.batch_size)
            Tbatch_xs, Tbatch_ys = VALID.next_batch(FLAGS.batch_size)
	    MAX_STEP=10000
	    constant=adaptation_factor(step*1.0/MAX_STEP)		
            summary,_=sess.run([merged,optimizer], feed_dict={x_s: batch_xs,x_t: Tbatch_xs,is_training:True,y: batch_ys,revgrad_lamb:constant,dropout_keep_prob:0.5,yt:Tbatch_ys})
	    train_writer.add_summary(summary,step)
	
            if step%600==0:
		epoch=step/600
                print("{} Start validation".format(datetime.datetime.now()))
		#print 'Epoch {0:<10} Step {1:<10} C_loss {2:<10} Advloss {3:<10}'.format(epoch,step,closs,advloss)
                test_acc = 0.
                test_count = 0
		bs=500
		print constant
		print 'test_counts ',len(TEST.labels)
                for _ in xrange((len(TEST.labels))/bs):
                    batch_tx, batch_ty = TEST.next_batch(bs)
                    sess.run(metrics_update_op, feed_dict={x_t: batch_tx, yt: batch_ty, is_training:False,dropout_keep_prob: 1.})
                    osacc,osstaracc,allacc,unkacc = sess.run([os_acc,osstar_acc,all_acc,unk_acc], feed_dict={x_t: batch_tx, yt: batch_ty, is_training:False,dropout_keep_prob: 1.})
                    test_count += bs
		res=len(TEST.labels)%bs
                if res>0:
		    batch_tx, batch_ty = TEST.next_batch(res)
                    sess.run(metrics_update_op, feed_dict={x_t: batch_tx, yt: batch_ty, is_training:False,dropout_keep_prob: 1.})
                    osacc,osstaracc,allacc,unkacc = sess.run([os_acc,osstar_acc,all_acc,unk_acc], feed_dict={x_t: batch_tx, yt: batch_ty, is_training:False,dropout_keep_prob: 1.})
		
		print "Epoch {4:<5} OS {0:<10} OS* {1:<10} ALL {2:<10} UNK {3:<10}".format(osacc,osstaracc,allacc,unkacc,epoch)
		sess.run(reset_ops)	
		if epoch==300:
		    return


if __name__ == '__main__':
    tf.app.run()
