import numpy as np
import tensorflow as tf
import copy
from tqdm import tqdm

import pdb

class LearnerIRL:
    def __init__(self, sess, map_input, config):
        self.config_ = config
        self.lr_ = self.config_.lr
        self.sess_ = sess
        self.map_ = map_input
        self.particles_ = np.random.uniform(-2, 2, size = [self.config_.particle_num, self.config_.shape ** 2])
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        self.initial_val_maps_ = {}
        self.initial_valg_maps_ = {}
        self.value_iter_op_ = self.map_.value_iter_tf if self.config_.use_tf else self.map_.value_iter
        self.gradient_iter_op_ = self.map_.grads_iter_tf if self.config_.use_tf else self.map_.grads_iter
        for i in range(self.config_.particle_num + 1):
            self.initial_val_maps_[i] = np.random.uniform(0, 1, size = [self.map_.num_states_, 1])
            self.initial_valg_maps_[i] = np.random.uniform(-1, 1, size = [self.map_.num_states_, self.map_.num_states_])

    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        for i in range(self.config_.particle_num + 1):
            self.initial_val_maps_[i] = np.random.uniform(0, 1, size = [self.map_.num_states_, 1])
            self.initial_valg_maps_[i] = np.random.uniform(-1, 1, size = [self.map_.num_states_, self.map_.num_states_])
    
    def learn_cont(self, mini_batch_indices, opt_actions, data_idx, gradients, step, gt_w, K = None, batch = False):
        prev_mean = copy.deepcopy(self.current_mean_)
        exp_cache_prev_func = lambda w_est: -1 * self.config_.beta_select *\
                                            ((self.config_.lr ** 2) * np.sum(np.square(gradients), axis = 1) +\
                                            2 * self.config_.lr * np.sum((prev_mean - w_est) * gradients, axis = 1))
        exp_cache_func = lambda vals: np.exp(vals - np.max(vals))
        teacher_sample_lle_func = lambda exps: np.log((exps / np.sum(exps))[data_idx])
        lle_gradient_func = lambda exps: 2 * self.config_.beta_select * self.config_.lr * gradients[data_idx: data_idx + 1, ...] -\
                                                2 * self.config_.beta_select * self.config_.lr * np.sum(gradients *\
                                                np.expand_dims(exps, -1) / np.sum(exps), axis = 0, keepdims = True)
        
        def get_grad():
            val_map, q_map, _ = self.value_iter_op_(self.current_mean_, value_map_init = self.initial_val_maps_[0], hard_max = True)
            if np.sum(np.isnan(val_map)) > 0:
                pdb.set_trace()
            self.initial_val_maps_[0] = val_map
            valg_map, qg_map, _ = self.gradient_iter_op_(q_map, value_map_init = self.initial_valg_maps_[0])
            if np.sum(np.isnan(valg_map)) > 0:
                print("STEP", step)
                print("PARTICLE NUM", i)
                pdb.set_trace()
            self.initial_valg_maps_[0] = valg_map
            
            if batch:
                exp_q = np.exp(self.config_.beta * q_map[mini_batch_indices, ...])
                action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
                gradients = self.config_.beta * (qg_map[mini_batch_indices, opt_actions, ...] -\
                                                 np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices, ...], axis = 1))
                return np.mean(gradients, axis = 0, keepdims = True)

            action_q = q_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...]
            exp_q = np.exp(self.config_.beta * (action_q - np.max(action_q)))
            action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
            particle_gradient = self.config_.beta * (qg_map[mini_batch_indices[data_idx], opt_actions[data_idx]: opt_actions[data_idx] + 1, ...] -\
                                                        np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...], axis = 1))
            if np.sum(np.isnan(particle_gradient)) > 0:
                pdb.set_trace()
            if np.sum(particle_gradient != 0) == 0:
                pdb.set_trace()
            return particle_gradient
        
        def get_new_lle():
            val_map, q_map, _ = self.value_iter_op_(self.current_mean_, value_map_init = self.initial_val_maps_[0], hard_max = True)
            self.initial_val_maps_[0] = val_map
            plle = self.config_.beta * q_map[(mini_batch_indices, opt_actions)] -\
                   np.log(np.sum(np.exp(self.config_.beta * q_map[mini_batch_indices, ...]), axis = 1))
            return plle

        if K is None:
            total_lle = -1 * np.inf
            steps = 0
            while True:
                gradient_tf = get_grad()
                self.current_mean_ += self.config_.lr * gradient_tf
                lle_gradient = lle_gradient_func(exp_cache_func(exp_cache_prev_func(self.current_mean_)))
                self.current_mean_ += self.config_.lr * lle_gradient
                new_lle = get_new_lle()
                exp_cache = exp_cache_func(exp_cache_prev_func(self.current_mean_))
                current_lle = teacher_sample_lle_func(exp_cache) + new_lle[data_idx]
                steps += 1
                if total_lle >= current_lle:
                    break
                else:
                    total_lle = current_lle
        elif K > 0:
            for i in range(K):
                gradient_tf = get_grad()
                self.current_mean_ += self.config_.lr * gradient_tf
                lle_gradient = lle_gradient_func(exp_cache_func(exp_cache_prev_func(self.current_mean_)))
                self.current_mean_ += self.config_.lr * lle_gradient
        else:
            for i in range(abs(K)):
                gradient_tf = get_grad()
                self.current_mean_ += self.config_.lr * gradient_tf

        return self.current_mean_, -1

    def learn(self, mini_batch_indices, opt_actions, data_idx, gradients, step, gt_w, random_prob = None):
        particle_gradients = []
        for i in range(self.config_.particle_num):
            val_map, q_map, _ = self.value_iter_op_(self.particles_[i: i + 1, ...], value_map_init = self.initial_val_maps_[i], hard_max = True)
            if np.sum(np.isnan(val_map)) > 0:
                pdb.set_trace()
            self.initial_val_maps_[i] = val_map
            valg_map, qg_map, _ = self.gradient_iter_op_(q_map, value_map_init = self.initial_valg_maps_[i])
            if np.sum(np.isnan(valg_map)) > 0:
                print("STEP", step)
                print("PARTICLE NUM", i)
                pdb.set_trace()
            self.initial_valg_maps_[i] = valg_map

            action_q = q_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...]
            exp_q = np.exp(self.config_.beta * (action_q - np.max(action_q)))
            action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
            particle_gradient = self.config_.beta * (qg_map[mini_batch_indices[data_idx], opt_actions[data_idx]: opt_actions[data_idx] + 1, ...] -\
                                                     np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...], axis = 1))
            particle_gradients.append(particle_gradient)
        particle_gradients = np.concatenate(particle_gradients, axis = 0)
        if np.sum(np.isnan(particle_gradients)) > 0:
            pdb.set_trace()
        self.particles_ += self.lr_ * particle_gradients
        eliminate = 0

        gradient = gradients[data_idx: data_idx + 1, ...]
        target_center = self.current_mean_ + self.lr_ * gradient
        val_target = self.lr_ * self.lr_ * np.sum(np.square(gradient)) +\
                            2 * self.lr_ * np.sum((self.current_mean_ - self.particles_) * gradient, axis = 1)

        gradients_cache = self.lr_ * self.lr_ * np.sum(np.square(gradients), axis = 1)
        scale = self.config_.noise_scale_min + (self.config_.noise_scale_max - self.config_.noise_scale_min) *\
                np.exp (-1 * step / self.config_.noise_scale_decay)
        #scale = np.power(0.5, int(1.0 * step / self.config_.noise_scale_decay)) * self.config_.noise_scale_max

        to_be_replaced = []
        for i in range(self.config_.particle_num):
            if random_prob is not None:
                rd = np.random.choice(2, p = [1 - random_prob, random_prob])
                if rd == 1:
                    to_be_replaced.append(i)
                continue
            particle_cache = self.current_mean_ - self.particles_[i: i + 1, ...]
            count = 0
            for j in range(mini_batch_indices.shape[0]):
                if j != data_idx:
                    val_cmp = gradients_cache[j] + 2 * self.lr_ * np.sum(particle_cache * gradients[j: j + 1, ...])
                    if val_target[i] - val_cmp > 1e-8:
                        count += 1
                    if count == self.config_.replace_count:
                        to_be_replaced.append(i)
                        break

        to_be_kept = list(set(range(self.config_.particle_num)) - set(to_be_replaced))
        cosine = 0
        #min_idx = to_be_kept[np.argmin(np.array(move_dists)[np.array(to_be_kept)])] if len(to_be_kept) > 0 else None
        if len(to_be_replaced) > 0:
            if len(to_be_kept) > 0 and step > 2:
                new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                             np.mean(self.particles_[np.array(to_be_kept), ...], axis = 0, keepdims = True)
                # new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                #              self.particles_[min_idx: min_idx + 1, ...]
            else:
                new_center = target_center

            new_val_map, new_q_map, _ = self.value_iter_op_(new_center, value_map_init = self.initial_val_maps_[self.config_.particle_num], hard_max = True)
            if np.sum(np.isnan(new_val_map)) > 0:
                pdb.set_trace()
            self.initial_val_maps_[self.config_.particle_num] = new_val_map
            new_valg_map, _, _ = self.gradient_iter_op_(new_q_map, value_map_init = self.initial_valg_maps_[self.config_.particle_num])
            self.initial_valg_maps_[self.config_.particle_num] = new_valg_map

            replace_center = np.mean(self.particles_[np.array(to_be_replaced), ...], axis = 0)
            # kept_dist = np.sum(np.square(new_center - gt_w))
            # replace_dist = np.sum(np.square(replace_center - gt_w))
            prod = np.sum((gt_w - new_center) * (gt_w - replace_center))
            norm = np.sum(np.square((gt_w - new_center)))
            cosine = prod - norm

        for i in to_be_replaced:
            # if len(to_be_kept) > 0:
            #     scale = 0.5 * abs(self.lr_ * np.mean(particle_gradients, axis = 0))
            noise = np.random.normal(scale = scale, size = [1, self.config_.shape ** 2])
                        # noise = t.rvs(df = 5, scale = scale,
                        #               size = [1, self.config_.num_classes, self.config_.data_dim + 1])
            #rd = np.random.choice(2, p = [1 - replace_ratio, replace_ratio])
            rd = np.random.rand()
            if rd < 1  - self.config_.prob:
                self.particles_[i: i + 1, ...] += 0 #target_center + (noise if random_prob != 1 else 0)
            else:
                self.particles_[i: i + 1, ...] = new_center + (noise if random_prob != 1 else 0)
                self.initial_val_maps_[i] = copy.deepcopy(new_val_map)
                self.initial_valg_maps_[i] = copy.deepcopy(new_valg_map)
            eliminate += 1
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        return eliminate, cosine

    def learn_imit_cont(self, mini_batch_indices, opt_actions, data_idx, lle, step, gt_w, K = None):
        def get_grads_lle():
            val_map, q_map, _ = self.value_iter_op_(self.current_mean_, value_map_init = self.initial_val_maps_[0], hard_max = True)
            self.initial_val_maps_[0] = val_map
            valg_map, qg_map, _ = self.gradient_iter_op_(q_map, value_map_init = self.initial_valg_maps_[0])
            self.initial_valg_maps_[0] = valg_map

            action_q = q_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...]
            exp_q = np.exp(self.config_.beta * (action_q - np.max(action_q)))
            action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
            particle_gradients = self.config_.beta * (qg_map[mini_batch_indices, opt_actions, ...] -\
                                                     np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices, ...], axis = 1))
            plle = self.config_.beta * q_map[(mini_batch_indices, opt_actions)] -\
                   np.log(np.sum(np.exp(self.config_.beta * q_map[mini_batch_indices, ...]), axis = 1))
            if np.sum(np.isnan(particle_gradients)) > 0:
                pdb.set_trace()
            if np.sum(particle_gradients != 0) == 0:
                pdb.set_trace()
            return particle_gradients, plle

        gradients, _ = get_grads_lle()
        exp_cache_prev_func = lambda new_lle: -1 * self.config_.beta_select *\
                                            ((self.config_.lr ** 2) * np.sum(np.square(gradients), axis = 1) +\
                                            2 * self.config_.lr * (lle - new_lle))
        exp_cache_func = lambda vals: np.exp(vals - np.max(vals))
        teacher_sample_lle_func = lambda exps: np.log((exps / np.sum(exps))[data_idx])
        lle_gradient_func = lambda w_est_loss_grad, exps:\
                                2 * self.config_.beta_select * self.config_.lr * w_est_loss_grad[data_idx: data_idx + 1, ...] -\
                                2 * self.config_.beta_select * self.config_.lr * np.sum(w_est_loss_grad *\
                                np.expand_dims(exps, -1) / np.sum(exps), axis = 0, keepdims = True)
        
        current_w_losses_gradients = copy.deepcopy(gradients)
        current_lle_as_loss = copy.deepcopy(lle)          
        exp_cache = exp_cache_func(exp_cache_prev_func(current_lle_as_loss))
        if K is None:
            total_lle = -1 * np.inf
            exp_cache = exp_cache_func(exp_cache_prev_func(current_lle_as_loss))
            steps = 0
            while True:
                self.current_mean_ += self.config_.lr * current_w_losses_gradients[data_idx: data_idx + 1, ...]
                current_w_losses_gradients, current_lle_as_loss = get_grads_lle()
                exp_cache = exp_cache_func(exp_cache_prev_func(current_lle_as_loss))
                lle_gradient = lle_gradient_func(current_w_losses_gradients, exp_cache)
                self.current_mean_ += self.config_.lr * lle_gradient
                current_w_losses_gradients, current_lle_as_loss = get_grads_lle()
                exp_cache = exp_cache_func(exp_cache_prev_func(current_lle_as_loss))
                current_lle = teacher_sample_lle_func(exp_cache) + current_lle_as_loss[data_idx]
                steps += 1
                if total_lle >= current_lle:
                    break
                else:
                    total_lle = current_lle
        elif K > 0:
            for i in range(K):
                self.current_mean_ += self.config_.lr * current_w_losses_gradients[data_idx: data_idx + 1, ...]
                current_w_losses_gradients, current_lle_as_loss = get_grads_lle()
                exp_cache = exp_cache_func(exp_cache_prev_func(current_lle_as_loss))
                lle_gradient = lle_gradient_func(current_w_losses_gradients, exp_cache)
                self.current_mean_ += self.config_.lr * lle_gradient
                current_w_losses_gradients, _ = get_grads_lle()

        return self.current_mean_, -1

    #lle is ok to share, because as long as the reward is the same, lle is the same
    # gradients should not be shared, as it depends on different state feature, which is private knowledge
    def learn_imit(self, mini_batch_indices, opt_actions, data_idx, lle, step, gt_w):
        _, q_map, _ = self.value_iter_op_(self.current_mean_, value_map_init = self.initial_val_maps_[self.config_.particle_num])
        _, qg_map, _ = self.gradient_iter_op_(q_map, value_map_init = self.initial_valg_maps_[self.config_.particle_num])

        exp_q = np.exp(self.config_.beta * q_map[mini_batch_indices, ...])
        action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
        gradients = self.config_.beta * (qg_map[mini_batch_indices, opt_actions, ...] -\
                                         np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices, ...], axis = 1))

        particle_gradients = []
        for i in range(self.config_.particle_num):
            val_map, q_map, _ = self.value_iter_op_(self.particles_[i: i + 1, ...], value_map_init = self.initial_val_maps_[i], hard_max = True)
            self.initial_val_maps_[i] = val_map
            valg_map, qg_map, _ = self.gradient_iter_op_(q_map, value_map_init = self.initial_valg_maps_[i])
            self.initial_valg_maps_[i] = valg_map

            action_q = q_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...]
            exp_q = np.exp(self.config_.beta * (action_q - np.max(action_q)))
            action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
            particle_gradient = self.config_.beta * (qg_map[mini_batch_indices[data_idx], opt_actions[data_idx]: opt_actions[data_idx] + 1, ...] -\
                                                     np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...], axis = 1))
            particle_gradients.append(particle_gradient)
        self.particles_ += self.lr_ * np.concatenate(particle_gradients, axis = 0)
        
        new_particle_lles = []
        for i in range(self.config_.particle_num):
            val_map, q_map, _ = self.value_iter_op_(self.particles_[i: i + 1, ...], value_map_init = self.initial_val_maps_[i], hard_max = True)
            self.initial_val_maps_[i] = val_map
            plle = self.config_.beta * q_map[(mini_batch_indices, opt_actions)] -\
                   np.log(np.sum(np.exp(self.config_.beta * q_map[mini_batch_indices, ...]), axis = 1))
            new_particle_lles.append(plle)

        eliminate = 0 
        gradient = gradients[data_idx: data_idx + 1, ...]
        target_center = self.current_mean_ - self.lr_ * gradient
        val_target = self.lr_ * self.lr_ * np.sum(np.square(gradient))
        scale = self.config_.noise_scale_min + (self.config_.noise_scale_max - self.config_.noise_scale_min) *\
                np.exp (-1 * step / self.config_.noise_scale_decay)
        #scale = np.power(0.5, int(1.0 * step / self.config_.noise_scale_decay)) * self.config_.noise_scale_max
        to_be_replaced = []
        gradient_cache = self.lr_ * self.lr_ * np.sum(np.square(gradients), axis = 1)
        for i in range(self.config_.particle_num):
            val_target_temp = val_target + 2 * self.lr_ * (lle[data_idx] - new_particle_lles[i][data_idx])
            val_cmps = gradient_cache + 2 * self.lr_ * (lle - new_particle_lles[i])
            count = 0
            for j in range(mini_batch_indices.shape[0]):
                if j != data_idx and val_target_temp - val_cmps[j] > 1e-8:
                    count += 1
                if count == self.config_.replace_count:
                    to_be_replaced.append(i)
                    eliminate += 1
                    break

        to_be_kept = list(set(range(0, self.config_.particle_num)) - set(to_be_replaced))
        cosine = 0
        if len(to_be_replaced) > 0:
            if len(to_be_kept) > 0 and step > 2:
                new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                             np.mean(self.particles_[np.array(to_be_kept), ...], axis = 0, keepdims = True)
            else:
                new_center = target_center

            new_val_map, new_q_map, _ = self.value_iter_op_(new_center, value_map_init = self.initial_val_maps_[self.config_.particle_num], hard_max = True)
            self.initial_val_maps_[self.config_.particle_num] = new_val_map
            new_valg_map, _, _ = self.gradient_iter_op_(new_q_map, value_map_init = self.initial_valg_maps_[self.config_.particle_num])
            self.initial_valg_maps_[self.config_.particle_num] = new_valg_map

            replace_center = np.mean(self.particles_[np.array(to_be_replaced), ...], axis = 0)
            prod = np.sum((gt_w - new_center) * (gt_w - replace_center))
            norm = np.sum(np.square((gt_w - new_center)))
            cosine = prod - norm

        for i in to_be_replaced:
            noise = noise = np.random.normal(scale = scale, size = [1, self.config_.shape ** 2])
                        # noise = t.rvs(df = 5, scale = scale,
                        #               size = [1, self.config_.num_classes, self.config_.data_dim + 1])
            rd = np.random.rand()
            if rd < 1 - self.config_.prob:
                self.particles_[i: i + 1, ...] += 0 #target_center + (noise if random_prob != 1 else 0)
            else:
                self.particles_[i: i + 1, ...] = new_center + noise
                self.initial_val_maps_[i] = copy.deepcopy(new_val_map)
                self.initial_valg_maps_[i] = copy.deepcopy(new_valg_map)

        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        return eliminate, cosine

    def current_action_prob(self):
        _, self.q_map_, _ = self.value_iter_op_(self.current_mean_, hard_max = True)
                                          #value_map_init = self.initial_val_maps_[self.config_.particle_num])
        q_balance = self.q_map_ - np.max(self.q_map_, axis = 1, keepdims = True)
        exp_q = np.exp(self.config_.beta * q_balance)
        self.action_probs_ = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
        return self.action_probs_
