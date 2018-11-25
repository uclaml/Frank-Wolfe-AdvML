import tensorflow as tf
import numpy as np
import random
import time
 
from FW_black import FW_black
from classifier import classifier


from PIL import Image
 

def generate_data(data, samples, targeted=True, start=0, num_attacks = 1, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample some targets  
    """
    inputs = []
    targets = []
    random.seed(1234)
    
    x = data.test_data
    y = data.test_labels
    
    # Filter samples that are correctly classified by the target classifier    
    cl = classifier(sess, model, loss_type = args.loss)
    eval_origin = cl.classify(x, y)

    legal_ind = [ind for ind in range(len(eval_origin)) if eval_origin[ind]]

    data.test_data = data.test_data[legal_ind]
    data.test_labels = data.test_labels[legal_ind]

    
    for i in range(samples):
        
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), num_attacks)
                print ('Target Label: ', seq)
                print ('Correct Label: ', np.argmax(data.test_labels[start+i]))
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
    parser.add_argument('--method', help='attack method')
    parser.add_argument('--arch', '-a', default='inception', help='attack architecture')    
    parser.add_argument('--loss', default='cross_entropy', help='loss function type: cross_entropy / cw')
    parser.add_argument('--grad_est_batch', default=25, type=int, help='gradient estimation batch size')
    parser.add_argument('--sample', default=1, type=int, help='number of samples to attack')
    parser.add_argument('--eps', default=0.05, type=float, help='epsilon limit')
    parser.add_argument('--delta', default=0.01, type=float, help='gradient estimation parameter')
    parser.add_argument('--maxiter', default=10000, type=int, help='max number of iterations')
    parser.add_argument('--numatt', default=1, type=int, help='number of target labels to attack per image')
    parser.add_argument('--start', default=0, type=int, help='start image index')
    parser.add_argument('--sensing', default='gaussian', help='sensing vector type: gaussian / sphere')
    parser.add_argument('--init_const', default=10, type=float, help='initial_const for ZOO')
    parser.add_argument('--bin_step', default=1, type=int, help='binary search steps for ZOO')
    parser.add_argument('--lambd', default=30, type=float, help='lambda')


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
        else:
            print ('Unknown Arch')
            import sys
            sys.exit() 
 
 
        if args.method == 'FW_L2':
            attack = FW_black(sess, model, grad_est_batch_size=args.grad_est_batch, ord=2, eps=args.eps, 
                              inception=inception, nb_iter=args.maxiter, lr = args.lr, loss_type = args.loss, 
                              sensing_type = args.sensing, delta = args.delta, lambd = args.lambd)
        elif args.method == 'FW_Linf':
            attack = FW_black(sess, model, grad_est_batch_size=args.grad_est_batch, ord=np.inf, eps=args.eps, 
                              inception=inception, nb_iter=args.maxiter, lr = args.lr, loss_type = args.loss, 
                              sensing_type = args.sensing, delta = args.delta, lambd = args.lambd)
        else:
            print ('Unknown Attack Methods')
            import sys
            sys.exit()
               

        print('Generating data')
        inputs, targets = generate_data(data, samples=args.sample, targeted=True, start=args.start, num_attacks = args.numatt,
                                        inception=inception)

        
        
        timestart = time.time()
        adv, query_images, time_images, succ_rate = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

  