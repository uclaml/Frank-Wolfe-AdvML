import tensorflow as tf
import numpy as np
import random
import time
import sys
import pickle
import json

from classifier import classifier
from utils import *

from FW import FW

import argparse


def generate_data(data, samples, targeted=True, start=0, num_attacks = 1, is_imagenet=False):

    inputs = []
    targets = []
    
    x = data.test_data
    y = data.test_labels
        
    cl = classifier(sess, model)
    eval_origin = cl.classify(x, y)
    legal_ind = [ind for ind in range(len(eval_origin)) if eval_origin[ind]]

    data.test_data = data.test_data[legal_ind]
    data.test_labels = data.test_labels[legal_ind]
        
    for i in range(samples):
        
        if targeted:
            if is_imagenet:
                seq = random.sample(range(1,1001), num_attacks)
            else:
                seq = [(np.argmax(data.test_labels[start+i]) + 1) % 10]
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])) and (is_imagenet == False):
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


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--att_lr', default=0.01, type=float, help='attack learning rate')
    parser.add_argument('--order', default="inf", help='attack order')
    parser.add_argument('--method', '-m', default="PGD", help='attack method')
    parser.add_argument('--arch', '-a', default='inception', help='target architecture')
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    parser.add_argument('--sample', default=10000, type=int, help='number of samples to attack')
    parser.add_argument('--eps', default=0.3, type=float, help='attack epsilon')
    parser.add_argument('--att_iter', default=1000, type=int, help='max number of attack iterations')
    parser.add_argument('--targeted', default=1, type=int, help='targeted attack: 1 nontargeted attack: -1')
    parser.add_argument('--beta1', default=0.99, type=float, help='beta1 for FW')

    args = parser.parse_args()
    print(args)

    with tf.Session() as sess:
        print('Loading model...')
        if args.arch == 'resnet':
            from setup_resnet import ImageNet, resnet_model
            data, model = ImageNet(), resnet_model(sess)
            is_imagenet = 1
            print ('ImageNet Resnet Model Loaded')
        elif args.arch == 'inception':
            from setup_inception_v3 import ImageNet, inception_model
            data, model = ImageNet(), inception_model(sess)
            is_imagenet = 1
            print ('ImageNet Inception Model Loaded')            
        elif args.arch == 'cifar':
            from setup_cifar import CIFAR, CIFARModel
            data, model = CIFAR(), CIFARModel("models/cifar",sess)
            is_imagenet = 0
            print ('CIFAR10 Model Loaded')
        elif args.arch == 'mnist':
            from setup_mnist import MNIST, MNISTModel
            data, model = MNIST(), MNISTModel("models/mnist",sess)
            is_imagenet = 0
            print ('MNIST Model Loaded')
        else:
            print ('Unknown Arch')
            import sys
            sys.exit() 

        TARGETED = True if args.targeted == 1 else False
        ORDER = 2 if args.order == "2" else np.inf
         
        attack = FW(sess, model, batch_size=args.batch, eps=args.eps, lr=args.att_lr, att_iter=args.att_iter,
                    targeted=TARGETED, order=ORDER, beta1=args.beta1)


        # generate data
        print('Generate data')
        inputs, targets = generate_data(data, samples=args.sample, targeted=True, is_imagenet=is_imagenet)
        print('Inputs Shape: ', inputs.shape)
         
        # start attacking
        total_iter = 0
        total_succ = 0
        adv = []
        stop_time = []
        stop_iter = []
        finished = []

        total_batch = int(np.ceil(len(inputs) / args.batch))

        timestart = time.time()
        for i in range(total_batch):
            start = i * args.batch
            end = min((i + 1) * args.batch, len(inputs))
            ind = range(start, end) 

            adv_b, stop_time_b, stop_iter_b, finished_b = attack.attack(inputs[ind], targets[ind], data_ori=inputs[ind])

            adv.extend(adv_b)
            stop_time.extend(stop_time_b)
            stop_iter.extend(stop_iter_b)
            finished.extend(finished_b)

            total_iter += sum(stop_iter_b)
            total_succ += sum(finished_b)

            print ('batch: ', i + 1, ' avg iter: ', total_iter / (i + 1), 'total succ: ', total_succ)
            print ('===========================')

        timeend = time.time()
        print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")

        adv = np.array(adv)
        stop_time = np.array(stop_time)
        stop_iter = np.array(stop_iter)
        finished = np.array(finished)

        l2 = []
        linf = []
        for i in range(len(adv)):
            l2_sample = np.sum((adv[i] - inputs[i]) ** 2) ** .5
            linf_sample = np.max(np.abs(adv[i] - inputs[i]))
            if finished[i]:
                l2.append(l2_sample)
                linf.append(linf_sample)

        l2 = np.array(l2)
        linf = np.array(linf)
        print ("======================================")
        print("Total L2 distortion: ", np.mean(l2))
        print("Total Linf distortion: ", np.mean(linf))
        print("Mean Time: ", np.mean(stop_time))
        print("Mean Iter: ", np.mean(stop_iter))
        print("Succ Rate: ", np.mean(finished))

        summary_txt = 'L2 distortion: ' + str(np.mean(l2)) + ' Linf distortion: ' + str(
            np.mean(linf)) + ' Mean Time: ' + str(np.mean(stop_time)) + ' Total Time: ' + str(
            timeend - timestart) + ' Mean Iter: ' + str(np.mean(stop_iter)) + ' Succ Rate: ' + str(np.mean(finished))
        with open(args.method + '_' + args.order + '_' + args.arch + '_whitebox' + '_summary' + '.txt', 'w') as f:
            json.dump(summary_txt, f)

 