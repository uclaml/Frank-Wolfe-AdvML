import time
from utils import *
import tensorflow as tf


class FW_black:
    def __init__(self, sess, cnn_model, att_iter=10000, grad_est_batch_size=25, order=np.inf, eps=0.05, clip_min=0,
                 clip_max=1, targeted=True, lr=0.01, delta=0.01, sensing_type='gaussian', q_limit=50000, beta1=0.99):

        image_size, num_channels, num_labels = cnn_model.image_size, cnn_model.num_channels, cnn_model.num_labels
        self.sess = sess
        self.att_iter = att_iter
        self.cnn_model = cnn_model
        self.targeted = targeted
        self.grad_est_batch_size = grad_est_batch_size
        self.batch_size = 1  # Only support batch_size = 1 in black-box setting
        self.ord = order
        self.epsilon = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.lr = lr
        self.delta = delta
        self.sensing_type = sensing_type
        self.q_limit = q_limit
        self.beta1 = beta1

        self.shape = (None, image_size, image_size, num_channels)
        self.single_shape = (image_size, image_size, num_channels)

        self.img = tf.placeholder(tf.float32, self.shape)
        self.lab = tf.placeholder(tf.float32, (None, num_labels))

        self.logits, self.pred = self.cnn_model.predict(self.img)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.lab)
        self.eval_adv = tf.equal(self.pred, tf.argmax(self.lab, 1))

        self.tloss = tf.reduce_sum(self.loss)
        self.gradients, = tf.gradients(self.loss, self.img)

        # GRADIENT ESTIMATION GRAPH
        grad_estimates = []
        final_losses = []

        noise_pos = tf.random_normal((self.grad_est_batch_size,) + self.single_shape)
        if self.sensing_type == 'sphere':
            reduc_ind = list(range(1, len(self.shape)))
            noise_norm = tf.sqrt(tf.reduce_sum(tf.square(noise_pos), reduction_indices=reduc_ind, keep_dims=True))
            noise_pos = noise_pos / noise_norm
            d = np.prod(self.single_shape)
            noise_pos = noise_pos * (d ** 0.5)
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
        elif self.sensing_type == 'gaussian':
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
        else:
            print ('Unknown Sensing Type')
            import sys
            sys.exit()

        self.grad_est_imgs = self.img + self.delta * noise
        self.grad_est_labs = tf.ones([self.grad_est_batch_size * 2, 1]) * self.lab

        grad_est_logits, _ = self.cnn_model.predict(self.grad_est_imgs)
        grad_est_losses = tf.nn.softmax_cross_entropy_with_logits(logits=grad_est_logits, labels=self.grad_est_labs)
        grad_est_losses_tiled = tf.tile(tf.reshape(grad_est_losses, (-1, 1, 1, 1)), (1,) + self.single_shape)
        grad_estimates.append(tf.reduce_mean(grad_est_losses_tiled * noise, axis=0) / self.delta)
        final_losses.append(grad_est_losses)
        self.grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
        self.final_losses = tf.concat(final_losses, axis=0)

    def eval_image(self, inputs, targets):
        loss, pred, eval_adv = self.sess.run([self.tloss, self.pred, self.eval_adv],
                                             {self.img: inputs, self.lab: targets})
        return loss, pred, eval_adv

    # GRADIENT ESTIMATION EVAL
    def get_grad_est(self, x, batch_lab, num_batches):
        losses = []
        grads = []
        for _ in range(num_batches):
            final_losses, grad_estimate = self.sess.run([self.final_losses, self.grad_estimate], {self.img: x,
                                                                                                  self.lab: batch_lab})
            losses.append(final_losses)
            grads.append(grad_estimate)
        grads = np.array(grads)
        losses = np.array(losses)
        return losses.mean(), np.mean(grads, axis=0, keepdims=True)

    def attack(self, inputs, targets, data_ori):

        adv = np.copy(inputs)
        stop_query = np.zeros((len(inputs)))
        stop_time = np.zeros((len(inputs)))

        loss_init, pred_init, eval_adv = self.eval_image(inputs, targets)
        finished_mask = np.logical_not(eval_adv) if not self.targeted else eval_adv
        succ_sum = sum(finished_mask)

        dist = get_dist(inputs, data_ori, self.ord)
        print ("Init Loss : % 5.3f, Dist: % 5.3f, Finished: % 3d " % (
            loss_init, dist, succ_sum))

        if succ_sum == len(inputs):
            return inputs, stop_query, stop_time, finished_mask

        for i in range(len(inputs)):

            data = inputs[i:i + 1]
            lab = targets[i:i + 1]
            ori = data_ori[i:i + 1]
            x = data
            num_batches = 1
            m_t = np.zeros_like(data)

            last_ls = []
            hist_len = 5
            min_lr = 1e-3
            current_lr = self.lr
            start_decay = 0

            for iteration in range(self.att_iter):
                start_time = time.time()

                stop_query[i] += num_batches * self.grad_est_batch_size * 2

                if stop_query[i] > self.q_limit:
                    stop_query[i] = self.q_limit
                    break

                # Get zeroth-order gradient estimates
                _, grad = self.get_grad_est(x, lab, num_batches)
                # momentum
                m_t = m_t * self.beta1 + grad * (1 - self.beta1)
                grad_normalized = grad_normalization(m_t, self.ord)

                s_t = - (-1 if not self.targeted else 1) * self.epsilon * grad_normalized + ori
                d_t = s_t - x
                current_lr = self.lr if start_decay == 0 else self.lr / (iteration - start_decay + 1) ** 0.5
                new_x = x + current_lr * d_t
                new_x = np.clip(new_x, self.clip_min, self.clip_max)

                x = new_x
                stop_time[i] += (time.time() - start_time)

                loss, pred, eval_adv = self.eval_image(x, lab)

                last_ls.append(loss)
                last_ls = last_ls[-hist_len:]
                if last_ls[-1] > 0.999 * last_ls[0] and len(last_ls) == hist_len:
                    if start_decay == 0:
                        start_decay = iteration - 1
                        print ("[log] start decaying lr")
                    last_ls = []

                finished_mask[i] = np.logical_not(eval_adv[0]) if not self.targeted else eval_adv[0]

                if iteration % 10 == 0:
                    dist = get_dist(x, ori, self.ord)
                    print ("Iter: %3d, Loss: %5.3f, Dist: %5.3f, Lr: %5.4f, Finished: %3d, Query: %3d"
                        % (iteration, loss, dist, current_lr, succ_sum, stop_query[i]))

                if finished_mask[i]:
                    break

            adv[i] = new_x

            dist = get_dist(x, ori, self.ord)
            print ("End Loss : % 5.3f, Dist: % 5.3f, Finished: % 3d,  Query: % 3d " % (
                loss, dist, finished_mask[i], stop_query[i]))

        return adv, stop_time, stop_query, finished_mask
