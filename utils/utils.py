import logging
import os.path

import requests
import tensorflow as tf
logger = logging.getLogger(__name__)
import sys
sys.path.append('../')

from dataset.mnist import MNIST
from dataset.svhn import SVHN
from dataset.usps import USPS


def get_data(name,split,unk,shuffle,frange):
	prefix='dataset/data/'
	if name=='mnist':
		return MNIST(path=prefix+'mnist',split=split,unk=unk,shuffle=shuffle,frange=frange)
	if name=='usps':
		return USPS(path=prefix+'usps',split=split,unk=unk,shuffle=shuffle,frange=frange)
	if name=='svhn':
		return SVHN(path=prefix+'svhn',split=split,unk=unk,shuffle=shuffle,frange=frange)

def get_optimizer(opt,lr,loss,var):
	if opt=='adam':
		return tf.train.AdamOptimizer(lr,0.9).minimize(loss,var_list=var)
	if opt=='mom':
		return tf.train.MomentumOptimizer(lr,0.9).minimize(loss,var_list=var)
def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)


def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
