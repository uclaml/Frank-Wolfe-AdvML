import sys
import tensorflow as tf
import numpy as np
from six.moves import xrange
import time


from utils import get_dist, norm_ball_proj_inner, grad_normalization, eps_search


class FW_black:
    def __init__(self, sess, model, nb_iter=10000, grad_est_batch_size=25, ord=np.inf, eps=0.05, clip_min=0, clip_max=1, targeted=True, inception=False, lr = 0.03, delta = 0.01, loss_type = 'cross_entropy', sensing_type = 'gaussian', lambd = 30, beta1 = 0.999, beta2 = 0.99, adaptive = False, output_steps = 10, test = False):

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.nb_iter = nb_iter
        self.model = model
        self.targeted = targeted
        self.grad_est_batch_size = grad_est_batch_size
        self.batch_size = 1  # Only support batch_size = 1 in black-box setting
        self.ord = ord
        self.epsilon = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.inception = inception
        self.lr = lr
        self.delta = delta
        self.loss_type = loss_type
        self.sensing_type = sensing_type
        self.lambd = lambd
        self.beta1 = beta1
        self.beta2 = beta2
        self.adaptive = adaptive
        self.output_steps = output_steps
        self.test = test

        self.shape = (self.batch_size,image_size,image_size,num_channels)
        self.single_shape = (image_size,image_size,num_channels)

        self.img = tf.placeholder(tf.float32, self.shape)
        self.lab = tf.placeholder(tf.float32, (self.batch_size,num_labels))
 
        def get_loss(eval_points, labels):
            logits, pred = self.model.predict(eval_points)
            
            print (' shape: ',logits.shape, labels.shape)
            if self.loss_type == 'cross_entropy':
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            elif self.loss_type == 'cw':
                correct_logit = tf.reduce_sum(labels * logits, axis=1)
                wrong_logit = tf.reduce_max((1-labels) * logits, axis=1)
                loss = tf.maximum(wrong_logit - correct_logit, -50)
            else:
                print ('Unknown Loss Type')
                import sys
                sys.exit()
            eval_adv = tf.equal(pred, tf.argmax(labels, 1))
            return loss, pred, eval_adv
        
        
        self.loss, self.pred, self.eval_adv = get_loss(self.img, self.lab)
        if not self.targeted:
            self.loss = -self.loss
        self.tloss = tf.reduce_sum(self.loss)

        self.gradients, = tf.gradients(self.loss, self.img)
   
        # GRADIENT ESTIMATION GRAPH
        grad_estimates = []
        final_losses = []
        
        noise_pos = tf.random_normal((self.grad_est_batch_size,) + self.single_shape)
        
        if self.sensing_type == 'sphere':
            reduc_ind = list(xrange(1, len(self.shape)))
            noise_norm = tf.sqrt(tf.reduce_sum(tf.square(noise_pos), reduction_indices=reduc_ind, keep_dims=True))
            noise_pos = noise_pos / noise_norm
            d = np.prod(self.single_shape)
            noise_pos = noise_pos * (d**0.5)
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
        elif self.sensing_type == 'gaussian':
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
        else:
            print ('Unknown Sensing Type')
            import sys
            sys.exit()
        
        self.grad_est_imgs = self.img + self.delta * noise
        self.grad_est_labs = tf.ones([self.grad_est_batch_size * 2, 1]) * self.lab
        
        print (self.grad_est_imgs.shape, self.grad_est_labs.shape)
        

        grad_est_losses, _, _ = get_loss(self.grad_est_imgs, self.grad_est_labs)
        grad_est_losses_tiled = tf.tile(tf.reshape(grad_est_losses, (-1, 1, 1, 1)), (1,) + self.single_shape)
        grad_estimates.append(tf.reduce_mean(grad_est_losses_tiled * noise, axis=0)/self.delta)
        final_losses.append(grad_est_losses)
        self.grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
        self.final_losses = tf.concat(final_losses, axis=0)
        
    

 

    def attack(self, inputs, targets):
         
        # GRADIENT ESTIMATION EVAL
        def get_grad_est(x, batch_lab, num_batches):
            losses = []
            grads = []
            for _ in range(num_batches):
                final_losses, grad_estimate = self.sess.run([self.final_losses, self.grad_estimate], {self.img: x, self.lab: batch_lab})
                losses.append(final_losses)
                grads.append(grad_estimate)
            grads = np.array(grads)
            losses = np.array(losses)
            return losses.mean(), np.mean(grads, axis=0, keepdims = True)
        
        adv = []
        adv = np.array(inputs)
        query_images = []
        query_images = np.zeros((len(inputs)))
        time_images = []
        time_images = np.zeros((len(inputs)))
        
        succ = 0


        for i in range(len(inputs)):
            start = time.time()
            batch_data = inputs[i:i+1]
            batch_lab = targets[i:i+1]

            x = batch_data
            num_batches = 1
            max_lr = self.lr
            current_ep = self.epsilon*self.lambd
            num_queries = 0
            last_ls = []
 
            
            print ('--------------------')
            for iteration in range(self.nb_iter):
                loss, pred, eval_adv, true_grad = self.sess.run([self.tloss, self.pred, self.eval_adv, self.gradients], {self.img: x, self.lab: batch_lab})
            
                # Get zeroth-order gradient estimates
                l, grad = get_grad_est(x, batch_lab, num_batches)
                num_queries += num_batches*self.grad_est_batch_size *2
                                

                # LR Decaying
                current_lr = self.lr / (iteration + 1)**0.5
    
 
                grad_normalized = grad_normalization(grad, self.ord)
#                 grad_normalized = grad_normalization(true_grad, self.ord)
 

                v = - current_ep * grad_normalized + batch_data
                d = v - x
        
                g = self.epsilon*np.sum(np.abs(true_grad)) - np.sum((batch_data - x)*true_grad)
                 
                x = x + current_lr * d

                eta = x - batch_data
                x = batch_data + norm_ball_proj_inner(eta, self.ord, self.epsilon)
                x = np.clip(x, self.clip_min, self.clip_max)

                succ += eval_adv

                if self.test:
                    if succ ==1:
                        print ('succ, queries: ', num_queries)
                    print (g)
                    if g<1:
                        break
                else:
                    if iteration % self.output_steps == 0:
                        dist = get_dist(x, batch_data, self.ord)
                        print("Iter: {}, Loss: {:0.5f}, Queries: {},  Dist: {:0.5f}, Eps:  {}, lr: {:.5f}, Pred: {},  Eval: {}, g: {}".format(iteration, l, num_queries, dist, current_ep, current_lr, pred, eval_adv, g))
                    if eval_adv:
                        break


            time_images[i] = time.time() - start
            query_images[i] = num_queries
            if eval_adv:
                adv[i] = x
            print ('Succ ', succ, ' / ', (i+1), ' rate: ', succ/(i+1))    
        print ('Total Succ ', succ, ' / ', len(inputs), ' rate: ', succ/len(inputs))  
#         print (adv[0], query_images)
        return adv, query_images, time_images, succ/len(inputs)
    
    
    
    
    