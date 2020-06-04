import numpy as np
import tensorflow as tf
import copy
from collections import defaultdict
from easydict import EasyDict as edict

import pdb

class Map:
    def __init__(self, sess, config):
        self.config_ = config
        self.sess_ = sess
        self.shape_ = self.config_.shape
        self.num_states_ = self.shape_ * self.shape_
        self.actions_ = 'udlr'
        self.state_feats_ = np.identity(self.num_states_)
        if self.config_.shuffle_state_feat:
            self.feat_idx_ = np.arange(self.num_states_)
            np.random.shuffle(self.feat_idx_)
            self.state_feats_ = self.state_feats_[self.feat_idx_, ...]
        self.transition_map_ = []

        self.move_eps_ = 0.18
        self.death_prob_ = 0.02
        self.gamma_ = 0.5
        self.vi_eps_ = 1e-6
        self.reward_param_ = None
        if self.config_.approx_type == 'p-norm':
            self.approx_k_ = self.config_.approx_k
            self.approx_max_ = lambda vals, k, a: np.power(np.sum(np.power(np.array(vals), k), axis = a), 1.0 / k)
            self.approx_max_tf_ = lambda vals, k, a: tf.math.pow(tf.reduce_sum(tf.math.pow(vals, k), axis = a), 1.0 / k)
        elif self.config_.approx_type == 'gsm':
            self.approx_k_ = self.config_.approx_k
            self.approx_max_ = lambda vals, k, a: np.max(np.array(vals), axis = a) +\
                                                  np.log(np.sum(np.exp(k * (np.array(vals) -\
                                                                            np.max(np.array(vals),
                                                                                   axis = a,
                                                                                   keepdims = True))), axis = a)) / k
            self.approx_max_tf_ = lambda vals, k, a: tf.reduce_max(vals, axis = a) +\
                                                     tf.log(tf.reduce_sum(tf.exp(k * (vals -\
                                                                                      tf.reduce_max(vals, axis = a,
                                                                                                    keepdims = True))), axis = a)) / k

        def idx2vec(idx):
            a = np.zeros([1, self.num_states_])
            np.put(a, idx, 1)
            return a
        self.idx2vec_ = idx2vec

        def printval(val_map):
            for i in range(self.shape_):
                print(val_map[i * self.shape_: (i + 1) * self.shape_])
            return
        self.printval_ = printval

        for aidx, a in enumerate(self.actions_):
            self.transition_map_.append(np.zeros(shape = [1, self.num_states_, self.num_states_]))
            for s in range(self.num_states_):
                dest = self.get_dest(s, a)
                for d in dest:
                    self.transition_map_[aidx][0, s, d] = dest[d]
        self.transition_map_ = np.concatenate(self.transition_map_, axis = 0)

        self.value_map_0_ = tf.placeholder(dtype = tf.float64, shape = [self.num_states_, 1])
        self.value_map_1_ = tf.placeholder(dtype = tf.float64, shape = [self.num_states_, 1])
        self.gradient_map_0_ = tf.placeholder(dtype = tf.float64, shape = [self.num_states_, self.num_states_])
        self.gradient_map_1_ = tf.placeholder(dtype = tf.float64, shape = [self.num_states_, self.num_states_])
        self.rewards_tf_ = tf.placeholder(dtype = tf.float64, shape = [self.num_states_, 1])
        self.factor_map_ = tf.placeholder(dtype = tf.float64, shape = [len(self.actions_), self.num_states_, 1])
        self.transition_map_tf_ = tf.constant(self.transition_map_, dtype = tf.float64)
        self.tf_counter_ = tf.constant(0, tf.int32)
        self.condition_ = lambda v0, v1, counter: tf.greater(tf.reduce_max(abs(v0 - v1)), self.vi_eps_)

        def val_iter(v0, v1, counter):
            temp = self.approx_max_tf_(tf.matmul(self.transition_map_tf_,
                                        tf.tile(tf.expand_dims(self.gamma_ * v1 + self.rewards_tf_, 0),
                                        (len(self.actions_), 1, 1))), self.approx_k_, 0)
            return [v1, temp, counter + 1]

        def val_iter_hardmax(v0, v1, counter):
            temp = tf.reduce_max(tf.matmul(self.transition_map_tf_,
                                tf.tile(tf.expand_dims(self.gamma_ * v1 + self.rewards_tf_, 0),
                                (len(self.actions_), 1, 1))), axis = 0)
            return [v1, temp, counter + 1]

        self.value_iter_tf_ = tf.while_loop(self.condition_, val_iter, [self.value_map_0_, self.value_map_1_, self.tf_counter_])
        self.value_iter_hardmax_tf_ = tf.while_loop(self.condition_, val_iter_hardmax, [self.value_map_0_, self.value_map_1_, self.tf_counter_])
        def grad_iter(v0, v1, counter):
            temp = tf.reduce_sum(self.factor_map_ * tf.matmul(self.transition_map_tf_,
                                                              tf.tile(tf.expand_dims(self.state_feats_ + self.gamma_ * v1, 0),
                                                                      (len(self.actions_), 1, 1))), axis = 0)
            return [v1, temp, counter + 1]

        self.gradient_iter_tf_ = tf.while_loop(self.condition_, grad_iter, [self.gradient_map_0_, self.gradient_map_1_, self.tf_counter_])

    def get_dest(self, start, direct):
        assert(start < self.num_states_)
        targets = defaultdict(float)
        for a in self.actions_:
            if a == 'u':
                dest = start - self.shape_ if start >= self.shape_ else start
            elif a == 'd':
                dest = start + self.shape_ if start < self.num_states_ - self.shape_ else start
            elif a == 'l':
                dest = start - 1 if start % self.shape_ != 0 else start
            elif a == 'r':
                dest = start + 1 if start % self.shape_ != self.shape_ - 1 else start
            if a == direct:
                targets[dest] += 1 - self.move_eps_ - self.death_prob_
            else:
                targets[dest] += self.move_eps_ / 3

        return targets

    def reward_generate(self, num_peak):
        peaks = set()
        while len(list(peaks)) < num_peak:
            peak_x = np.random.randint(self.shape_)
            peak_y = np.random.randint(self.shape_)
            peaks.add((peak_x, peak_y))
        peaks = list(peaks)
        rewards = 0 * np.ones(shape = [1, self.num_states_])
        for peak in peaks:
            rewards[0, peak[0] + peak[1] * self.shape_] = 1

        return rewards

    def value_iter(self, reward_param, value_map_init = None, rewards = None, hard_max = False):
        if rewards is None:
            rewards = np.sum(self.state_feats_ * reward_param, axis = 1, keepdims = True)
        value_map_ = np.zeros([self.num_states_, 1])
        value_map = np.random.uniform(-1, 1, size = [self.num_states_, 1]) if value_map_init is None else value_map_init
        diff = 1e3
        vi_iter = 0
        while diff > self.vi_eps_:
            vi_iter += 1
            if not hard_max:
                value_map_ = self.approx_max_(np.matmul(self.transition_map_,
                                                         np.tile(np.expand_dims(self.gamma_ * value_map + rewards, 0),
                                                                 (len(self.actions_), 1, 1))),
                                              self.approx_k_, 0)
            else:
                value_map_ = np.max(np.matmul(self.transition_map_,
                                              np.tile(np.expand_dims(self.gamma_ * value_map + rewards, 0),
                                                      (len(self.actions_), 1, 1))), axis = 0)
            diff = np.max(abs(value_map_ - value_map))
            np.copyto(value_map, value_map_)


        q_map = np.matmul(self.transition_map_,
                          np.tile(np.expand_dims(self.gamma_ * value_map + rewards, 0), (len(self.actions_), 1, 1)))[:, :, 0].T

        return value_map, q_map, vi_iter

    def value_iter_tf(self, reward_param, value_map_init = None, rewards = None, hard_max = False):
        if rewards is None:
            rewards = np.sum(self.state_feats_ * reward_param, axis = 1, keepdims = True)
        value_map_ = np.zeros([self.num_states_, 1])
        value_map = np.random.uniform(-1, 1, size = [self.num_states_, 1]) if value_map_init is None else value_map_init

        value_iter_func = self.value_iter_hardmax_tf_ if hard_max else self.value_iter_tf_

        [_, value_map, vi_iter] = self.sess_.run(value_iter_func, feed_dict = {self.value_map_0_: value_map_,
                                                                                   self.value_map_1_: value_map,
                                                                                   self.rewards_tf_: rewards})

        q_map = np.matmul(self.transition_map_,
                          np.tile(np.expand_dims(self.gamma_ * value_map + rewards, 0), (len(self.actions_), 1, 1)))[:, :, 0].T

        return value_map, q_map, vi_iter

    def grads_iter(self, q_map, value_map_init = None):
        value_map_ = np.zeros(shape = [self.num_states_, self.num_states_])
        value_map = np.random.uniform(size = [self.num_states_, self.num_states_]) if value_map_init is None else value_map_init
        diff = 1e3
        vi_iter = 0
        if self.config_.approx_type == 'p-norm':
            factor = np.power(np.sum(np.power(q_map, self.approx_k_), axis = 1), (1 - self.approx_k_) / self.approx_k_)
            factor_a = np.power(q_map, self.approx_k_ - 1)
        elif self.config_.approx_type == 'gsm':
            q_map_balance = q_map - np.max(q_map, axis = 1, keepdims = True)
            factor = 1 / np.sum(np.exp(self.approx_k_ * q_map_balance), axis = 1)
            factor_a = np.exp(self.approx_k_ * q_map_balance)
        factor_map = np.expand_dims((np.expand_dims(factor, 1) * factor_a).T, 2)

        while diff > self.vi_eps_:
            vi_iter += 1
            value_map_ = np.sum(factor_map * np.matmul(self.transition_map_,
                                                       np.tile(np.expand_dims(self.state_feats_ + self.gamma_ * value_map, 0),
                                                               (len(self.actions_), 1, 1))), axis = 0)
            diff = np.max(abs(value_map_ - value_map))
            np.copyto(value_map, value_map_)

        q_map = np.transpose(np.matmul(self.transition_map_,
                                       np.tile(np.expand_dims(self.gamma_ * value_map + self.state_feats_, 0),
                                               (len(self.actions_), 1, 1))), (1, 0, 2))

        return value_map, q_map, vi_iter

    def grads_iter_tf(self, q_map, value_map_init = None):
        value_map_ = np.zeros(shape = [self.num_states_, self.num_states_])
        value_map = np.random.uniform(size = [self.num_states_, self.num_states_]) if value_map_init is None else value_map_init
        if self.config_.approx_type == 'p-norm':
            factor = np.power(np.sum(np.power(q_map, self.approx_k_), axis = 1), (1 - self.approx_k_) / self.approx_k_)
            factor_a = np.power(q_map, self.approx_k_ - 1)
        elif self.config_.approx_type == 'gsm':
            q_map_balance = q_map - np.max(q_map, axis = 1, keepdims = True)

            factor = 1 / np.sum(np.exp(self.approx_k_ * q_map_balance), axis = 1)
            factor_a = np.exp(self.approx_k_ * q_map_balance)
        factor_map = np.expand_dims((np.expand_dims(factor, 1) * factor_a).T, 2)

        [_, value_map, vi_iter] = self.sess_.run(self.gradient_iter_tf_, feed_dict = {self.gradient_map_0_: value_map_,
                                                                                      self.gradient_map_1_: value_map,
                                                                                      self.factor_map_: factor_map})

        q_map = np.transpose(np.matmul(self.transition_map_,
                                       np.tile(np.expand_dims(self.gamma_ * value_map + self.state_feats_, 0),
                                               (len(self.actions_), 1, 1))), (1, 0, 2))

        return value_map, q_map, vi_iter

    def test_walk(self, reward_param, action_prob, starts, greedy = True):
        rewards = []
        map_rewards = np.sum(self.state_feats_ * reward_param, axis = 1, keepdims = True)
        for trail in range(len(starts)):
            s = starts[trail]
            death = False
            reward = 0
            while not death:
                if greedy:
                    aidx = np.argmax(action_prob[s])
                else:
                    aidx = np.random.choice(len(self.actions_), p = action_prob[s])
                sidx = np.random.choice(self.num_states_ + 1, p = list(self.transition_map_[aidx, s, :]) + [self.death_prob_])
                if sidx == self.num_states_:
                    death = True
                else:
                    s = sidx
                    reward += map_rewards[s]
            rewards.append(reward)
        return np.mean(rewards)



def main():
    shape = 8
    config = edict({'shape': shape, 'approx_type': 'p-norm', 'shuffle_state_feat': False})
    game_map = Map(config)
    r_param = np.zeros(shape = [1, game_map.num_states_])
    r_param[0, shape * shape - 1] = 10

    val_map_gt, _ = game_map.value_iter(r_param, hard_max = True)
    game_map.printval_(val_map_gt)

    val_map_gsm, q_map_gsm = game_map.value_iter(r_param)
    _, qg_map_gsm = game_map.grads_iter(q_map_gsm)
    print('p-norm diff mean: %f, max: %f' % (np.mean(abs(val_map_gt - val_map_gsm)), np.max(abs(val_map_gt - val_map_gsm))))

    config = edict({'shape': shape, 'approx_type': 'gsm', 'shuffle_state_feat': False})
    game_map = Map(config)
    val_map_p, q_map_p = game_map.value_iter(r_param)
    _, qg_map_p = game_map.grads_iter(q_map_p)
    print('gsm diff mean: %f, max: %f' % (np.mean(abs(val_map_gt - val_map_p)), np.max(abs(val_map_gt - val_map_p))))

    print(np.mean(abs(qg_map_p - qg_map_gsm)), np.max(abs(qg_map_p - qg_map_gsm)))
    pdb.set_trace()

if __name__ == '__main__':
    main()
