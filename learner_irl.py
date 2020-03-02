import numpy as np
import copy
from tqdm import tqdm
from OIRL.map import Map

import pdb

class LearnerIRL:
    def __init__(self, config):
        self.config_ = config
        self.map_ = Map(config)
        self.particles_ = np.random.uniform(-2, 2, size = [self.config_.particle_num, self.config_.shape ** 2])
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
    
    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
    
    def learn(self, mini_batch_indices, opt_actions, data_idx, gradients, step, gt_w, random_prob = None):
        particle_gradients = []
        for i in tqdm(range(self.config_.particle_num)):
            _, q_map = self.map_.value_iter(self.particles_[i: i + 1, ...])
            _, qg_map = self.map_.grads_iter(q_map)
            
            exp_q = np.exp(self.config_.beta * q_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...])
            action_prob = exp_q / np.sum(exp_q, axis = 1, keepdims = True)
            particle_gradient = self.config_.beta * (qg_map[mini_batch_indices[data_idx], opt_actions[data_idx]: opt_actions[data_idx] + 1, ...] -\
                                                     np.sum(np.expand_dims(action_prob, 2) * qg_map[mini_batch_indices[data_idx]: mini_batch_indices[data_idx] + 1, ...], axis = 1))
            particle_gradients.append(particle_gradient)
        self.particles_ += self.config_.lr * np.concatenate(particle_gradients, axis = 0)
        eliminate = 0
        
        gradient = gradients[data_idx: data_idx + 1, ...]
        target_center = self.current_mean_ + self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient)) +\
                            2 * self.config_.lr * np.sum((self.current_mean_ - self.particles_) * gradient, axis = 1)

        gradients_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), axis = 1)
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
                    val_cmp = gradients_cache[j] + 2 * self.config_.lr * np.sum(particle_cache * gradients[j: j + 1, ...])
                    if val_target[i] - val_cmp > 1e-8:
                        count += 1
                    if count == self.config_.replace_count:
                        to_be_replaced.append(i)
                        break
        
        to_be_kept = list(set(range(self.config_.particle_num)) - set(to_be_replaced))
        #min_idx = to_be_kept[np.argmin(np.array(move_dists)[np.array(to_be_kept)])] if len(to_be_kept) > 0 else None
        if len(to_be_replaced) > 0:
            if len(to_be_kept) > 0 and step > 10:
                new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                             np.mean(self.particles_[np.array(to_be_kept), ...], axis = 0, keepdims = True)
                # new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                #              self.particles_[min_idx: min_idx + 1, ...]
            else:
                new_center = target_center

        replace_center = np.mean(self.particles_[np.array(to_be_replaced), ...], axis = 0)
        # kept_dist = np.sum(np.square(new_center - gt_w))
        # replace_dist = np.sum(np.square(replace_center - gt_w))
        prod = np.sum((gt_w - new_center) * (gt_w - replace_center))
        norm = np.sqrt(np.sum(np.square((gt_w - new_center)))) * np.sqrt(np.sum(np.square((gt_w - replace_center))))
        cosine = np.arccos(prod / norm)
        for i in to_be_replaced:
            noise = np.random.normal(scale = scale,
                                     size = [1, self.config_.shape ** 2])
                        # noise = t.rvs(df = 5, scale = scale,
                        #               size = [1, self.config_.num_classes, self.config_.data_dim + 1])
            #rd = np.random.choice(2, p = [1 - replace_ratio, replace_ratio])
            rd = np.random.rand()
            if rd < 1  - self.config_.prob:
                self.particles_[i: i + 1, ...] += 0 #target_center + (noise if random_prob != 1 else 0)
            else:
                self.particles_[i: i + 1, ...] = new_center + (noise if random_prob != 1 else 0)
            eliminate += 1

        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        return self.current_mean_, eliminate, cosine

    def learn_sur(self, data_pool, data_y, data_idx, gradients, prev_loss, step, gt_w):
        new_particle_losses = []
        gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, ...],
                                                        self.W_: self.particles_,
                                                        self.y_: data_y[data_idx: data_idx + 1, :]})

        self.particles_ -= self.config_.lr * gradient_tf[0]
        move_dists = np.sum(np.square(gradient_tf[0]), axis = (1, 2))
        for i in range(self.config_.particle_num):
            losses = self.sess_.run(self.losses_, {self.X_: data_pool,
                                                    self.W_: self.particles_[i: i + 1, ...],
                                                    self.y_: data_y})
            new_particle_losses.append(losses)

        eliminate = 0
        
        gradient = gradients[data_idx: data_idx + 1, ...]
        target_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient))
        scale = self.config_.noise_scale_min + (self.config_.noise_scale_max - self.config_.noise_scale_min) *\
                np.exp (-1 * step / self.config_.noise_scale_decay)
        #scale = np.power(0.5, int(1.0 * step / self.config_.noise_scale_decay)) * self.config_.noise_scale_max
        to_be_replaced = []
        gradient_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), axis = (1, 2))
        for i in range(self.config_.particle_num):
            val_target_temp = val_target - 2 * self.config_.lr * (prev_loss[data_idx] - new_particle_losses[i][data_idx])
            val_cmps = gradient_cache - 2 * self.config_.lr * (prev_loss - new_particle_losses[i])
            count = 0
            for j in range(data_pool.shape[0]):
                if j != data_idx and val_target_temp - val_cmps[j] > 1e-8:
                    count += 1
                if count == self.config_.replace_count:
                    to_be_replaced.append(i)
                    eliminate += 1
                    break

        to_be_kept = list(set(range(0, self.config_.particle_num)) - set(to_be_replaced))
        if len(to_be_replaced) > 0:
            if len(to_be_kept) > 0 and step > 10:
                new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                             np.mean(self.particles_[np.array(to_be_kept), ...], axis = 0, keepdims = True)
            else:
                new_center = target_center

        prod = np.sum((gt_w - new_center) * (gt_w - replace_center))
        norm = np.sqrt(np.sum(np.square((gt_w - new_center)))) * np.sqrt(np.sum(np.square((gt_w - replace_center))))
        cosine = np.arccos(prod / norm)
        for i in to_be_replaced:
            noise = np.random.normal(scale = scale,
                                     size = [1, self.config_.num_classes, self.config_.data_dim + 1])
                        # noise = t.rvs(df = 5, scale = scale,
                        #               size = [1, self.config_.num_classes, self.config_.data_dim + 1])
            rd = np.random.rand()
            if rd < 1 - self.config_.prob:
                self.particles_[i: i + 1, ...] += 0 #target_center + (noise if random_prob != 1 else 0)
            else:
                self.particles_[i: i + 1, ...] = new_center + noise
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        return self.current_mean_, eliminate, kept_dist, cosine