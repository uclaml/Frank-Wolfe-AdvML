# Frank-Wolfe-AdvML
A Frank-Wolfe Framework for Efficient and Effective Adversarial Attacks



## Prerequisites: 
* Tensorflow
* Setup Inception/ResNet model (see details follows)
* Download ImageNet validation set and put them in /imagenetdata/imgs/ folder

 
## Command Line Arguments:
* --lr: (start) learning rate 
* --method: attack method, e.g., "FW_L2", "FW_Linf"
* --arch: network architecture, e.g. ["inception"](https://arxiv.org/abs/1512.00567), ["resnet"](https://arxiv.org/abs/1603.05027) 
* --sample: number of samples to attack
* --eps: epsilon, value 0.0 to enable grid search
* --maxiter: maximum number of iterations per attack
* --lambd: lambda
* --grad_est_batch: zeroth-order gradient estimation batch size
* --sensing: type of sensing vectors, e.g. "gaussian", "sphere"


## Usage Examples:
* Setup Inception V3 model:
```bash
  -  python3 setup_inception_v3.py
```

* Setup Inception V3 model:
```bash
  -  python3 setup_resnet.py
```

* Run white-box attack on Inception V3 model:
```bash
  -  python3 test_attack.py --method "FW_Linf" --arch "inception" --sample 1 --eps 0.05 --lr 0.005 --lambd 5
```
* Run black-box attack on ResNet V2 model:
```bash
  -  python3 test_attack_black.py --method "FW_L2" --arch "resnet" --sample 500 --maxiter 1000 --eps 5 --lr 0.03 --lambd 30 --delta 0.001
```
 
