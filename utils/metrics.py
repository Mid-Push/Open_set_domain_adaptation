import numpy as np
import tensorflow as tf
import math

def OS_star(hx,y,num_classes=3):
	disc_y=tf.argmax(y,1)
	disc_hx=tf.argmax(hx,1)
	known_indices=tf.where(tf.not_equal(disc_y,num_classes-1))
	known_hx=tf.gather(disc_hx,known_indices)
	known_y=tf.gather(disc_y,known_indices)
	known_hx=tf.reshape(known_hx,[-1])
	known_y=tf.reshape(known_y,[-1])
	osstar,osstar_update=tf.metrics.mean_per_class_accuracy(labels=known_y,predictions=known_hx,num_classes=num_classes)
	return osstar*(num_classes)*1.0/(num_classes-1.0),osstar_update

def OS(hx,y,num_classes=6):
	os,os_update=tf.metrics.mean_per_class_accuracy(labels=tf.argmax(y,1),predictions=tf.argmax(hx,1),num_classes=num_classes)
	return os,os_update

def ALL(hx,y):
	allacc,all_update=tf.metrics.accuracy(labels=tf.argmax(y,1),predictions=tf.argmax(hx,1))
	return allacc,all_update

def UNK(hx,y,num_classes=6):
	
	disc_y=tf.argmax(y,1)
	disc_hx=tf.argmax(hx,1)
	known_indices=tf.where(tf.equal(disc_y,num_classes-1))
	known_hx=tf.gather(disc_hx,known_indices)
	known_y=tf.gather(disc_y,known_indices)
	known_hx=tf.reshape(known_hx,[-1])
	known_y=tf.reshape(known_y,[-1])
	unk,unk_update=tf.metrics.accuracy(labels=known_y,predictions=known_hx)
	return unk,unk_update
