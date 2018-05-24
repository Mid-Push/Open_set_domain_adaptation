# Open_set_domain_adaptation

Tensorflow Implementation of 《Open Set Domain Adaptation by Backpropagation》

On SVHN->MNIST and MNIST->USPS, USPS->MNIST

## Usage:

python osda_train.py

## Major Difference: Flip the sign of L_adv(xt) 
# (After confirmation with the authors, my thoughts are correct!)

### paper: 
Ladv(xt) = tlog(p(y = K + 1|xt)) + (1 − t) log(1 − p(y = K + 1|xt)

C: Ls(xs, ys) + Ladv(xt)

G: Ls(xs, ys) - Ladv(xt)

### my implementation:
Ladv(xt) = -(tlog(p(y = K + 1|xt)) + (1 − t) log(1 − p(y = K + 1|xt))

C: Ls(xs, ys) + Ladv(xt)

G: Ls(xs, ys) - Ladv(xt)

## Results:
# my implementation

OS 85.7 OS* 85.6 ALL 85.8 UNK 85.9
# paper

OS 92.3 OS* 91.2 ALL 94.4 UNK 97.6

 I'm trying to fix this gap at present. Any advice is welcome.  

![alt text](results/um.png "ALL Accuracy")
![alt text](results/l_adv.png "Test Accuracy")
