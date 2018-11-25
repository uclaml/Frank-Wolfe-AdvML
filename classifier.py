

import sys
import tensorflow as tf
import numpy as np
from six.moves import xrange

class classifier:
    def __init__(self, sess, model, samples = 1, loss_type = 'cross_entropy'):

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.loss_type = loss_type
        self.samples = samples

        self.shape = (samples,image_size,image_size,num_channels)

        self.img = tf.placeholder(tf.float32, self.shape)
        self.lab = tf.placeholder(tf.float32, (samples,num_labels))
        
        def get_loss(eval_points, labels):
            logits, pred = self.model.predict(eval_points)
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
            return loss, pred, eval_adv, logits
  
        self.loss, self.pred, self.eval_adv, self.logits = get_loss(self.img, self.lab)

        
 
 

    def classify(self, inputs, targets):
        preds = []
        for i in range(len(inputs)):
            batch_data = inputs[i:i+1]
            batch_lab = targets[i:i+1]
            x = batch_data
            pred, eval_adv, logits= self.sess.run([self.pred, self.eval_adv, self.logits], {self.img: x, self.lab: batch_lab})
            preds.append(eval_adv[0])
#         preds, eval_adv, logits= self.sess.run([self.pred, self.eval_adv, self.logits], {self.img: inputs, self.lab: targets})
           
        return preds