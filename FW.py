import sys
import tensorflow as tf
import numpy as np
import time
from utils import *


class FW:
    def __init__(self, sess, cnn_model, att_iter=20, batch_size=50, order=np.inf, eps=0.3, clip_min=0, clip_max=1,
                 targeted=True, lr=0.01, beta1=0.99):

        image_size, num_channels, num_labels = cnn_model.image_size, cnn_model.num_channels, cnn_model.num_labels
        self.sess = sess
        self.att_iter = att_iter
        self.cnn_model = cnn_model
        self.targeted = targeted
        self.batch_size = batch_size
        self.ord = order
        self.epsilon = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.lr = lr
        self.beta1 = beta1

        self.img = tf.placeholder(tf.float32, (None, image_size, image_size, num_channels))
        self.lab = tf.placeholder(tf.float32, (None, num_labels))

        self.logits, self.pred = self.cnn_model.predict(self.img)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.lab)
        self.eval_adv = tf.equal(self.pred, tf.argmax(self.lab, 1))

        self.tloss = tf.reduce_sum(self.loss)
        self.gradients, = tf.gradients(self.loss, self.img)

    def eval_image(self, inputs, targets):
        loss, losses, eval_adv = self.sess.run([self.tloss, self.loss, self.eval_adv],
                                               {self.img: inputs, self.lab: targets})
        return loss, losses, eval_adv

    def get_grad(self, inputs, targets):
        grad = self.sess.run(self.gradients, {self.img: inputs, self.lab: targets})
        return grad

    def attack(self, inputs, targets, data_ori):

        x = np.copy(inputs)
        stop_time = np.zeros((len(inputs)))
        stop_iter = np.zeros((len(inputs)))
        m_t = np.zeros_like(inputs)

        loss_init, _, eval_adv = self.eval_image(inputs, targets)
        finished_mask = np.logical_not(eval_adv) if not self.targeted else eval_adv
        succ_sum = sum(finished_mask)

        dist = get_dist(inputs, data_ori, self.ord)
        print ("Init Loss : % 5.3f, Dist: % 5.3f, Finished: % 3d " % (
            loss_init, dist, succ_sum))

        if succ_sum == len(inputs):
            return x, stop_time, stop_iter, finished_mask

        last_ls = []
        hist_len = 2
        min_lr = 1e-3
        current_lr = self.lr

        for iteration in range(self.att_iter):
            start_time = time.time()
            grad = self.get_grad(x, targets)

            m_t = m_t * self.beta1 + grad * (1 - self.beta1)
            grad_normalized = grad_normalization(m_t, self.ord)

            v_t = - self.epsilon * grad_normalized + data_ori
            d_t = v_t - x

            new_x = x + (-1 if not self.targeted else 1) * current_lr * d_t
            new_x = data_ori + norm_ball_proj_inner(new_x - data_ori, self.ord, self.epsilon)
            new_x = np.clip(new_x, self.clip_min, self.clip_max)

            mask = finished_mask.reshape(-1, *[1] * 3)
            x = new_x * (1. - mask) + x * mask
            stop_time += (time.time() - start_time) * (1. - finished_mask)
            stop_iter += 1 * (1. - finished_mask)

            loss, _, eval_adv = self.eval_image(x, targets)
            tmp = np.logical_not(eval_adv) if not self.targeted else eval_adv
            finished_mask = np.logical_or(finished_mask, tmp)
            succ_sum = sum(finished_mask)

            if iteration % 1 == 0:
                dist = get_dist(x, data_ori, self.ord)
                print ("Iter : % 3d, Loss : % 5.3f, Dist: % 5.3f, lr: % 5.3f, Finished: % 3d " % (
                    iteration, loss, dist, current_lr, succ_sum))

            if succ_sum == len(inputs):
                break
        return x, stop_time, stop_iter, finished_mask
