import tensorflow as tf
import numpy as np
import random
import time
 

from EAD import EADEN

from l2_attack import CarliniL2
from li_attack import CarliniLi

from IFGM import IFGM
from FW import FW
from PGD import PGD

from classifier import classifier

from PIL import Image
 

def generate_data(data, samples, targeted=True, start=0, num_attacks = 1, inception=False):
     
    inputs = []
    targets = []
    # random.seed(1234)
    
    x = data.test_data
    y = data.test_labels
        
    cl = classifier(sess, model, loss_type = args.loss)
    eval_origin = cl.classify(x, y)

    legal_ind = [ind for ind in range(len(eval_origin)) if eval_origin[ind]]
    # print (len(legal_ind))
    
    # print (data.test_data.shape, data.test_labels.shape)

    data.test_data = data.test_data[legal_ind]
    data.test_labels = data.test_labels[legal_ind]
    
    # print (data.test_data.shape, data.test_labels.shape)

    
    for i in range(samples):
        
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), num_attacks)
                # print ('Target Label: ', seq)
                # print ('Correct Label: ', np.argmax(data.test_labels[start+i]))
            else:
                seq = [(np.argmax(data.test_labels[start+i]) + 1) % 10]

            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])
            
    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', default=0.01, type=float, help='FW learning rate')
    parser.add_argument('--method', '-m', help='attack method')
    parser.add_argument('--arch', '-a', default='inception', help='attack architecture')    
    parser.add_argument('--loss', default='cross_entropy', help='loss type')
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    parser.add_argument('--sample', default=1, type=int, help='number of samples to attack')
    parser.add_argument('--eps', default=0.05, type=float, help='epsilon')
    parser.add_argument('--maxiter', default=1000, type=int, help='max number of iterations')
    parser.add_argument('--numatt', default=1, type=int, help='number of targeted attacks per image')
    parser.add_argument('--start', default=0, type=int, help='start image index')
    parser.add_argument('--lambd', default=5, type=float, help='lambda')
    parser.add_argument('--out_steps', default=10, type=int, help='output steps')


    args = parser.parse_args()
    print(args)
    inception = 1
    with tf.Session() as sess:
        print('Loading model...')
        if args.arch == 'resnet':
            from setup_resnet import ImageNet, resnet_model
            data, model = ImageNet(), resnet_model(sess)
            print ('ImageNet Resnet Model Loaded')
        elif args.arch == 'inception':
            from setup_inception_v3 import ImageNet, inception_model
            data, model = ImageNet(), inception_model(sess)
            print ('ImageNet Inception Model Loaded')    
        elif args.arch == 'mnist':
            from setup_mnist import MNIST, MNISTModel
            data, model = MNIST(), MNISTModel("models/mnist",sess)
            inception = 0
            print ('MNIST Model Loaded')
        else:
            print ('Unknown Arch')
            import sys
            sys.exit()   
            

        
        print ('lr: ', args.lr)
        print ('inception? ', inception)
        
 
        if args.method == 'FW_L2':
            attack = FW(sess, model, batch_size=args.batch, ord=2, eps=args.eps, inception=inception, 
                        nb_iter=args.maxiter, lr = args.lr, loss_type = args.loss, lambd = args.lambd,
                        output_steps = args.out_steps)
        elif args.method == 'FW_Linf':
            attack = FW(sess, model, batch_size=args.batch, ord=np.inf, eps=args.eps, inception=inception, 
                        nb_iter=args.maxiter, lr = args.lr, loss_type = args.loss, lambd = args.lambd,
                        output_steps = args.out_steps)
        else:
            print ('Unknown Attack Methods')
            import sys
            sys.exit()
               
        
        print('Generate data')
        inputs, targets = generate_data(data, samples=args.sample, targeted=True, start=args.start, num_attacks = args.numatt,
                                        inception=inception)
        print('Inputs Shape: ', inputs.shape)
        
        timestart = time.time()
        adv, time_images, succ_rate, stop_iter = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

 