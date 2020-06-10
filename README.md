
# A Frank-Wolfe Framework for Efficient and Effective Adversarial Attacks

This is the code for the [https://arxiv.org/abs/1811.10828](https://arxiv.org/abs/1811.10828) "A Frank-Wolfe Framework for Efficient and Effective Adversarial Attacks" by [Jinghui Chen](http://web.cs.ucla.edu/~jhchen/), [Dongruo Zhou](https://sites.google.com/view/drzhou), [Jinfeng Yi](http://jinfengyi.net/), and [Quanquan Gu](http://web.cs.ucla.edu/~qgu/).
  

## Prerequisites
* Python (3.6.9)
* Tensorflow (1.15.0)
* Inception/ResNet pre-trained model 
* Download ImageNet validation set and put them in /imagenetdata/imgs/ folder


## Usage Examples:

#### Pretrained Models:
* Setup Inception V3 model:
```bash
  -  python3 setup_inception_v3.py
```

* Setup ResNet model:
```bash
  -  python3 setup_resnet.py
```
 
 
#### Arguments:
* ```arch```: network architecture, e.g. ["inception"](https://arxiv.org/abs/1512.00567), ["resnet"](https://arxiv.org/abs/1603.05027) 
* ```sample```: number of samples to attack
* ```eps```: epsilon, value 0.0 to enable grid search
* ```att_iter```: maximum number of iterations per attack
* ```att_lr```: attack learning rate (step size)
* ```grad_est```: zeroth-order gradient estimation batch size
* ```sensing```: type of sensing vectors, e.g. "gaussian", "sphere"
* ```beta1```: mementum parameter for FW
* ```order```: attack threat model type ("2" or "inf")
  

#### Basic Running Examples:

* Run white-box attack on Inception V3 model:
```bash
  -  CUDA_VISIBLE_DEVICES=0 python3 test_attack.py --arch "inception" --method "FW" --order "inf" --sample 250 --eps 0.05 --att_lr 0.1 --beta1 0.9
```
* Run black-box attack on ResNet V2 model:
```bash
  -  CUDA_VISIBLE_DEVICES=0 python3 test_attack_black.py --arch "resnet" --method "FW" --order "inf" --sample 1000 --eps 0.3 --att_lr 0.8 --grad_est 25 --sensing "sphere"
```
 
