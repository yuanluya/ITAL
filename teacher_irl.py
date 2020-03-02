import numpy as np

from OIRL.map import Map

import pdb

class TeacherIRL:
    def __init__(self, config, gt_r_param):
        self.config_ = config
        self.reward_param_ = gt_r_param
        self.map_ = Map(config)
        self.rewards_ = np.sum(self.map_.state_feats_ * self.reward_param_, axis = 1)
        self.value_map_, self.q_map_ = self.map_.value_iter(self.reward_param_)
        self.optimal_actions_ = np.argmax(self.q_map_, axis = 1)
        self.mini_batch_indices_ = None
        self.l_ = self.config_.beta * self.q_map_[(range(self.map_.num_states_), self.optimal_actions_)] - np.log(np.sum(np.exp(self.config_.beta * self.q_map_), axis = 1))

    def sample(self):
        self.mini_batch_indices_ = np.random.randint(0, self.map_.num_states_, size = self.config_.sample_size)
        return
    
    def choose(self, learner_param, lr):
        assert(self.mini_batch_indices_ is not None)
        _, q_map = self.map_.value_iter(learner_param)
        _, qg_map = self.map_.grads_iter(q_map)
        opt_actions = self.optimal_actions_[self.mini_batch_indices_]
        
        exp_q = np.exp(self.config_.beta * q_map[self.mini_batch_indices_, ...])
        action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
        gradients = self.config_.beta * (qg_map[self.mini_batch_indices_, opt_actions, ...] -\
                                         np.sum(np.expand_dims(action_prob, 2) * qg_map[self.mini_batch_indices_, ...], axis = 1))
        
        vals = np.sum(lr * lr * np.square(gradients), axis = 1) - 2 * lr * np.sum((learner_param - self.reward_param_) * gradients, axis = 1)
        return np.argmin(vals)

    def choose_imit(self, learner_rewards, lr):
        assert(self.mini_batch_indices_ is not None)
        _, q_map = self.map_.value_iter(None, learner_rewards)
        _, qg_map = self.map_.grads_iter(q_map)
        opt_actions = self.optimal_actions_[self.mini_batch_indices_]

        exp_q = np.exp(self.config_.beta * q_map[self.mini_batch_indices_, ...])
        action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
        gradients = self.config_.beta * (qg_map[self.mini_batch_indices_, opt_actions, ...] -\
                                         np.sum(np.expand_dims(action_prob, 2) * qg_map[self.mini_batch_indices_, ...], axis = 1))
        
        l_stu = self.config_.beta * q_map[(self.mini_batch_indices_, opt_actions)] -\
                np.log(np.sum(np.exp(self.config_.beta * q_map[self.mini_batch_indices_, ...]), axis = 1))
        l_tea = self.l_[self.mini_batch_indices_]
        
        vals = np.sum(lr * lr * np.square(gradients), axis = 1) - 2 * lr * (l_stu - l_tea)
        return np.argmin(vals)

        return