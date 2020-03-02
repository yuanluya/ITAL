import numpy as np
from collections import defaultdict
from easydict import EasyDict as edict

import pdb

class Map:
    def __init__(self, config):
        self.config_ = config
        self.shape_ = self.config_.shape
        self.num_states_ = self.shape_ * self.shape_
        self.actions_ = 'udlr'
        self.state_feats_ = np.identity(self.num_states_)
        if self.config_.shuffle_state_feat:
            self.feat_idx_ = np.arange(self.num_states_)
            np.random.shuffle(self.feat_idx_)
            self.state_feats_ = self.state_feats_[self.feat_idx_, ...]

        self.move_eps_ = 0.15
        self.gamma_ = 0.8
        self.vi_eps_ = 1e-6
        self.reward_param_ = None
        if self.config_.approx_type == 'p-norm':
            self.approx_k_ = 112
            self.approx_max_ = lambda vals, k: np.power(np.sum(np.power(np.array(vals), k)), 1.0 / k)
        elif self.config_.approx_type == 'gsm':
            self.approx_k_ = 15.7
            self.approx_max_ = lambda vals, k: np.log(np.sum(np.exp(k * np.array(vals)))) / k

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
                targets[dest] += 1 - self.move_eps_
            else:
                targets[dest] += self.move_eps_ / 3
        
        return targets

    def value_iter(self, reward_param, rewards = None, hard_max = False):
        if rewards is None:
            rewards = np.sum(self.state_feats_ * reward_param, axis = 1)
        value_map_ = np.zeros(self.num_states_)
        value_map = np.random.uniform(-1, 1, size = self.num_states_)
        diff = 1e3
        vi_iter = 0
        while diff > self.vi_eps_:
            vi_iter += 1
            for i in range(self.num_states_):
                T = []
                for a in self.actions_:
                    dest = self.get_dest(i, a)
                    t = 0
                    for d in dest:
                        t += dest[d] * (rewards[d] + self.gamma_ * value_map[d])
                    T.append(t)
                if not hard_max:
                    value_map_[i] = self.approx_max_(T, self.approx_k_)
                else:
                    value_map_[i] = np.max(T)
            diff = np.max(abs(value_map_ - value_map))
            np.copyto(value_map, value_map_)

        q_map = np.zeros([self.num_states_, len(self.actions_)])
        for i in range(self.num_states_):
            for aidx, a in enumerate(self.actions_):
                dest = self.get_dest(i, a)
                for d in dest:
                    q_map[i, aidx] += dest[d] * (rewards[d] + self.gamma_ * value_map[d])
        
        return value_map, q_map
    
    def grads_iter(self, q_map):
        value_map_ = np.zeros(shape = [self.num_states_, self.num_states_])
        value_map = np.random.uniform(size = [self.num_states_, self.num_states_])
        diff = 1e3
        vi_iter = 0
        while diff > self.vi_eps_:
            vi_iter += 1
            for i in range(self.num_states_):
                T = []
                if self.config_.approx_type == 'p-norm':
                    factor = np.power(np.sum(np.power(q_map[i, ...], self.approx_k_)), (1 - self.approx_k_) / self.approx_k_)
                elif self.config_.approx_type == 'gsm':
                    factor = 1 / np.sum(np.exp(self.approx_k_ * q_map[i, ...]))
                for aidx, a in enumerate(self.actions_):
                    dest = self.get_dest(i, a)
                    t = 0
                    if self.config_.approx_type == 'p-norm':
                        factor_a = np.power(q_map[i, aidx], self.approx_k_ - 1)
                    elif self.config_.approx_type == 'gsm':
                        factor_a = np.exp(self.approx_k_ * q_map[i, aidx])
                    for d in dest:
                        t += factor * factor_a * dest[d] * (self.state_feats_[d, ...] + self.gamma_ * value_map[d, ...])
                    T.append(t) 
                value_map_[i, :] = np.sum(np.array(T), axis = 0)
            diff = np.max(abs(value_map_ - value_map))
            np.copyto(value_map, value_map_)
        
        q_map = np.zeros([self.num_states_, len(self.actions_), self.num_states_])
        for i in range(self.num_states_):
            for aidx, a in enumerate(self.actions_):
                dest = self.get_dest(i, a)
                for d in dest:
                    q_map[i, aidx, :] += dest[d] * (self.state_feats_[d, ...] + self.gamma_ * value_map[d, ...])
        
        return value_map, q_map




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