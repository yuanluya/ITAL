import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pdb

class multiclass_cn:
    def __init__(self, input_shape, label_shape, feat_dim, config, lr):
        self.inputs_ = tf.placeholder(tf.float32, [None, input_shape, input_shape, 1])
        self.labels_ = tf.placeholder(tf.float32, [None, label_shape])

        self.layers_ = [self.inputs_]
        for idx, (out_dim, kernel_size, stride, pool) in enumerate(config):
            out = tf.layers.conv2d(self.layers_[-1], out_dim, kernel_size = kernel_size, strides = stride, padding = 'SAME', 
                                  activation = tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 5e-2))
            if pool:   
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')
            self.layers_.append(out)
        
        self.features_ = tf.layers.dense(tf.layers.flatten(self.layers_[-1]), units = feat_dim, activation = tf.nn.tanh)
        self.logits_ = tf.layers.dense(self.features_, units = 10, activation = None, name = 'logit')

        self.params_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
                                     if v.name.startswith('logit')]
        self.true_ = tf.equal(tf.argmax(self.logits_,1), tf.argmax(self.labels_,1))
        self.loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_, logits = self.logits_))
        self.reg_loss_ = tf.add_n([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'bias' not in v.name])
        self.loss_ += 0 * self.reg_loss_
        self.accuracy_ = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits_,1), tf.argmax(self.labels_,1)), tf.float32))

        self.optimizer_ = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.7, beta2 = 0.95).minimize(self.loss_)

def train_mnist():
    configuration = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    configuration.gpu_options.allow_growth = True

    mnist = input_data.read_data_sets("MNIST", one_hot = True)

    input_shape = 28
    label_shape = 10
    feat_dim = 24
    epochs = 1000
    batch_size = 128
    config = [(64, 3, 1, True), (32, 3, 1, True), (32, 3, 1, False)]
    initial_lr = 1e-3
    accuracy = []
    loss = []
    train_features = np.ndarray((0, feat_dim))
    cf = multiclass_cn(input_shape, label_shape, feat_dim, config, initial_lr)
    sess = tf.Session(config = configuration)
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Start Training.")
    for i in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = np.reshape(batch_xs, (batch_size, input_shape, input_shape, 1))
        batch_ys = np.reshape(batch_ys, (batch_size, label_shape))
        _, l, acc, feat, logit, t = sess.run([cf.optimizer_, cf.loss_, cf.accuracy_, cf.features_, cf.logits_, cf.true_],
                                             feed_dict={cf.inputs_: batch_xs, cf.labels_: batch_ys})
        accuracy.append(acc)
        loss.append(l)
        if i % 100 == 0:
            # print(feat[0])
            # print(logit[0])
            # print(batch_ys[0])
            # print(t[0])
            print("Iteration %d loss: %f accuracy: %f\n" % (i, l, acc))

    print("Start Testing.")
    test_xs = np.reshape(np.load("testx.npy"), (-1, input_shape, input_shape, 1))
    test_ys = np.reshape(np.load("testy.npy"), (-1, label_shape))
    test_features, test_loss, test_acc = sess.run([cf.features_, cf.loss_,  cf.accuracy_], feed_dict={cf.inputs_: test_xs, cf.labels_: test_ys})
    print("Test loss: %f accuracy: %f" % (test_loss, test_acc))

    print('getting training image features')
    all_xs = np.load("trainx.npy")
    all_ys = np.load("trainy.npy")
    all_xs = np.reshape(all_xs, (all_xs.shape[0], input_shape, input_shape, 1))
    all_ys = np.reshape(all_ys, (all_xs.shape[0], label_shape))
    current_img_idx = 0
    while current_img_idx < all_xs.shape[0]:
        batch_xs = all_xs[current_img_idx: current_img_idx + batch_size, ...]
        feat = sess.run(cf.features_, feed_dict={cf.inputs_: batch_xs})
        train_features = np.concatenate((train_features, feat))
        current_img_idx += batch_size
    
    gt_weights, gt_bias = sess.run(cf.params_)
    gt_weights = np.concatenate([gt_weights, np.expand_dims(gt_bias, 0)], 0).T

    np.save("mnist_train_features.npy", train_features)
    np.save("mnist_test_features.npy", test_features)
    np.save("mnist_train_labels.npy", all_ys)
    np.save("mnist_test_labels.npy", test_ys)
    np.save("mnist_tf_gt_weights.npy", gt_weights)

    # plt.plot(loss, 'b', label = "Train Loss")
    # # plt.plot(test_loss, 'r', label = "Test Loss")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    train_mnist()