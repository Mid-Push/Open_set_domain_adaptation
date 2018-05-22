# Open_set_domain_adaptation

Tensorflow Implementation of 《Open Set Domain Adaptation by Backpropagation》

On SVHN->MNIST and MNIST->USPS, USPS->MNIST

## Usage:

python osda_train.py

## Major Difference: Flip the sign of L_adv(xt)

### paper: 
Ladv(xt) = tlog(p(y = K + 1|xt)) + (1 − t) log(1 − p(y = K + 1|xt)

C: Ls(xs, ys) + Ladv(xt)

G: Ls(xs, ys) - Ladv(xt)

### my implementation:
Ladv(xt) = -(tlog(p(y = K + 1|xt)) + (1 − t) log(1 − p(y = K + 1|xt))

C: Ls(xs, ys) + Ladv(xt)

G: Ls(xs, ys) - Ladv(xt)

## Results:
(OS performance is reported,i.e., 6 classes classification--[0,1,2,3,4,unknown])
![alt text](results/test_accuracy.png "Test Accuracy")
![alt text](results/l_adv.png "Test Accuracy")
