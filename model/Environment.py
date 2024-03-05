#!/usr/bin/env python
# coding: utf-8


import torch.nn as nn
import torch
import numpy as np
import sys; sys.path.append('../analysis')


class Env(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.__dict__ .update(arg.__dict__)
        
    def reset(self, goal_radius=None, target_position=None, 
              pro_gains=None, pro_noise_std=None,
              perturbation_velocities=None, perturbation_start_t=None):
        
        # reward zone radius
        self.goal_radius = goal_radius
        if goal_radius is None:
            self.goal_radius = torch.zeros(1).uniform_(*self.goal_radius_range)
        # target position
        self.target_position = target_position
        if target_position is None:
            target_rel_r = torch.sqrt(torch.zeros(1).uniform_(*self.initial_radius_range**2))
            target_rel_ang = torch.zeros(1).uniform_(*self.relative_angle_range)
            rel_phi = np.pi/2 - target_rel_ang
            target_x = target_rel_r * torch.cos(rel_phi)
            target_y = target_rel_r * torch.sin(rel_phi)
            self.target_position = torch.tensor([target_x, target_y]).view([-1, 1])
            self.init_target_position_polar = torch.tensor([target_rel_r, rel_phi]).view([-1, 1])
        self.target_position_obs = self.target_position.clone()
        
        # joystick gain
        self.pro_gains = pro_gains
        if pro_gains is None:
            if 'gain' not in self.task:
                gain_value = 1                
            else:
                gain_value = torch.zeros(1).uniform_(*self.process_gain_range)
                if self.exclude_gain is not None:
                    while self.exclude_gain[0] < gain_value < self.exclude_gain[1]:
                        gain_value = torch.zeros(1).uniform_(*self.process_gain_range)
            
            self.pro_gains = gain_value * self.process_gain_default
                
        # process noise std
        self.pro_noise_std = pro_noise_std
        if pro_noise_std is None and self.pro_noise_range is not None:
            self.pro_noise_std = self.pro_gains * torch.zeros(1).uniform_(*self.pro_noise_range)
                
        # perturbation
        self.perturbation_velocities = perturbation_velocities
        self.perturbation_start_t = perturbation_start_t
        if perturbation_velocities is None and 'perturbation' in self.task:
            self.perturbation_velocities = torch.zeros(2)
            self.perturbation_velocities[0].uniform_(*self.perturbation_velocity_range[:2])
            self.perturbation_velocities[1].uniform_(*self.perturbation_velocity_range[2:])
            if self.exclude_perturb is not None:
                while self.exclude_perturb[0] < self.perturbation_velocities[0] < self.exclude_perturb[1]:
                    self.perturbation_velocities[0].uniform_(*self.perturbation_velocity_range[:2])
                while self.exclude_perturb[2] < self.perturbation_velocities[1] < self.exclude_perturb[3]:
                    self.perturbation_velocities[1].uniform_(*self.perturbation_velocity_range[2:])
            
            self.perturbation_start_t = torch.randint(*self.perturbation_start_t_range, ())

        self.Perturbation_generator() if 'perturbation' in self.task else None

        return torch.tensor([0, 0, np.pi / 2, 0, 0]).view([-1, 1])


    def forward(self, x, a, t):    
        if not self.target_fullon and t == self.target_offT:
            self.target_position_obs *= 0  # disable target
        
        relative_dist = torch.dist(x[:2], self.target_position)
        reached_target = relative_dist < self.goal_radius
        
        if 'perturbation' in self.task and t in range(self.perturbation_start_t, 
                                                      self.perturbation_start_t + self.gaussian_v_array.shape[0]):
            self.perturbation_vt = self.gaussian_v_array[t - self.perturbation_start_t]
            self.perturbation_wt = self.gaussian_w_array[t - self.perturbation_start_t]
        else:
            self.perturbation_vt = torch.tensor(0)
            self.perturbation_wt = torch.tensor(0)
            
        next_x = self.dynamics(x, a.view(-1))

        return next_x, reached_target, relative_dist
    
    def dynamics(self, x, a):
        px, py, heading_angle, lin_vel, ang_vel = torch.split(x.view(-1), 1)
        
        eta = self.pro_noise_std * torch.randn(2) if self.pro_noise_std is not None else torch.zeros(2)
        
        px = px + lin_vel * torch.cos(heading_angle) * self.DT 
        py = py + lin_vel * torch.sin(heading_angle) * self.DT 
        heading_angle = heading_angle + ang_vel * self.DT 
        lin_vel = torch.tensor(0) * lin_vel + self.pro_gains[0] * a[0] + eta[0] + self.perturbation_vt
        ang_vel = torch.tensor(0) * ang_vel + self.pro_gains[1] * a[1] + eta[1] + self.perturbation_wt
        
        next_x = torch.stack([px, py, heading_angle, lin_vel, ang_vel]).view([-1, 1])
        return next_x
    
    def Perturbation_generator(self):
        sigma = self.perturbation_std
        duration = self.perturbation_duration
        assert duration % 2 == 0, 'Duration should be even.'
        
        mu = torch.tensor(duration / 2)
        x = torch.arange(duration + 1)
        gaussian_identity = torch.exp(-1/2 * ((x-mu) / sigma)**2)

        self.gaussian_v_array = gaussian_identity * self.perturbation_velocities[0]
        self.gaussian_w_array = gaussian_identity * self.perturbation_velocities[1]
        
    def return_reward(self, x, reward_mode='mixed'):
        def gauss_reward_func(distance):
            rew_std = self.goal_radius / 2
            cov_tar = torch.eye(2) * rew_std ** 2

            reward = torch.exp(-1/2 * distance.T @ cov_tar.inverse() @ distance)
            return reward
        
        def mixed_reward_func(distance):
            rew_std = self.goal_radius / 1.5
            cov_tar = torch.eye(2) * rew_std ** 2

            if torch.norm(distance) < self.goal_radius:
                reward = torch.ones(1)
            else:
                reward = torch.exp(-1/2 * distance.T @ cov_tar.inverse() @ distance)
            return reward.clamp(min=0.001)
        
        def const_reward_func(distance):
            if torch.norm(distance) < self.goal_radius:
                reward = torch.ones(1)
            else:
                reward = torch.zeros(1)
            return reward
        
        
        distance = x[:2] - self.target_position
        if reward_mode == 'gauss':
            reward = gauss_reward_func(distance)
        elif reward_mode == 'mixed':
            reward = mixed_reward_func(distance)
        elif reward_mode == 'const':
            reward = const_reward_func(distance)
        else:
            raise ValueError('Reward mode does not exist!')

        return reward.view(1, 1, -1)  * self.REWARD_SCALE
    
    def is_stop(self, x, action):
        stop = (action.abs() < self.TERMINAL_ACTION).all()
        return stop

