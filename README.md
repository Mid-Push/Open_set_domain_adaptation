# Open_set_domain_adaptation
Last Update: 3, Mar, 2019. Code is complete Now.

Unofficial Tensorflow Implementation of 《Open Set Domain Adaptation by Backpropagation》

On SVHN->MNIST and MNIST->USPS, USPS->MNIST

## Usage:

python osda_train.py


## Results:
# my implementation

OS 85.7 OS* 85.6 ALL 85.8 UNK 85.9
# paper

OS 92.3 OS* 91.2 ALL 94.4 UNK 97.6

I'm trying to fix this gap at present. Any advice is welcome.  

![alt text](results/um.png "ALL Accuracy")
![alt text](results/ladv.png "Test Accuracy")
