#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random


transition = namedtuple('transition', ('state', 'EKFstate', 'action', 'reward'))

class ReplayMemory():
    def __init__(self, MEMORY_SIZE, BATCH_SIZE):
        self.MEMORY_SIZE = MEMORY_SIZE
        self.SMALL_MEMORY_SIZE = int(MEMORY_SIZE / 10)
        self.BATCH_SIZE = BATCH_SIZE
        self.memory = []
        self.position = 0

    def push(self, *args):
        # a ring buffer
        if len(self.memory) < self.MEMORY_SIZE:
            self.memory.append(None)
            
        self.memory[self.position] = transition(*args)
        self.position = (self.position+1) % self.MEMORY_SIZE
        
    def sample(self):
        # 1. Sample a small memory.
        if len(self.memory) < self.SMALL_MEMORY_SIZE:
            small_memory = self.memory
        else:
            small_memory = random.sample(self.memory, self.SMALL_MEMORY_SIZE)
            
        # 2. Sample a trial length.
        traj_len = [traj.reward.shape[0] for traj in small_memory]
        traj_len = random.sample(traj_len, 1)[0]
        
        # 3. Get trials with same length.
        small_memory = [traj for traj in small_memory if traj.reward.shape[0] == traj_len]
        
        # 4. Sample a mini batch.
        batch = random.sample(small_memory, min(self.BATCH_SIZE, len(small_memory)))
        batch = transition(*zip(*batch))
        
        return batch
        
    def load(self, memory):          
        self.memory, self.position = memory
        
    def reset(self):
        self.memory = []
        self.position = 0
    

class BeliefStep(nn.Module):        # EKF belief
    def __init__(self, arg):
        super().__init__()
        self.__dict__ .update(arg.__dict__)
        self.__dict__.pop('obs_noise_range')
        self.obs_noise_range = arg.obs_noise_range
        
        self.H = torch.zeros(self.OBS_DIM, self.STATE_DIM)
        self.H[0, -2] = 1
        self.H[1, -1] = 1
     
    @property
    def obs_noise_range(self):
        return self._obs_noise_range
    
    @obs_noise_range.setter
    def obs_noise_range(self, value):
        self._obs_noise_range = [0, 0] if value is None else value
    
    def reset(self, pro_gains, pro_noise_std, target_position, obs_noise_std=None, epsilon=1e-8):
        self.pro_gains = pro_gains
        pro_noise_std = epsilon if pro_noise_std is None else pro_noise_std
        self.pro_noise_var = pro_noise_std ** 2
        self.target_position = target_position
        self.obs_noise_std = obs_noise_std
        
        if self.obs_noise_std is None:
            self.obs_noise_std = torch.zeros(1).uniform_(
                                    self.obs_noise_range[0], 
                                    self.obs_noise_range[1]) * pro_gains
        self.obs_noise_var = self.obs_noise_std ** 2
        
        x = torch.cat([-self.target_position, torch.tensor([np.pi / 2, 0, 0]).view(-1, 1)], dim=0)
        P = torch.eye(self.STATE_DIM) * epsilon  # belief uncertainty
        b = x, P  # belief
        state = self.b_reshape(b)
        return b, state
    
    def b_reshape(self, b): # reshape belief for policy
        x, P = b

        P_x = P[:self.POS_DIM, :self.POS_DIM]
        P_x = P_x[list(torch.triu_indices(self.POS_DIM, self.POS_DIM))]
        P_v = torch.diag(P[-self.OBS_DIM:, -self.OBS_DIM:])
        
        state = torch.cat([x.view(-1), P_x, P_v])
        return state.view(1, -1)
    
    def observation(self, x, obs_noise_std=None):
        obs_noise_std = self.obs_noise_std if obs_noise_std is None else obs_noise_std
        zita = (obs_noise_std * torch.randn(self.OBS_DIM)).view([-1, 1])
        o_t = self.H @ x + zita
        return o_t

    def forward(self, b, o_t, a, perturbation_vt, perturbation_wt):
        Q_t = torch.zeros(self.STATE_DIM, self.STATE_DIM)
        Q_t[-self.ACTION_DIM:, -self.ACTION_DIM:] = self.pro_noise_var * torch.eye(self.ACTION_DIM)
        R_t = self.obs_noise_var * torch.eye(self.OBS_DIM)
        
        # predict b_ and P_
        b_last, P_last = b
        b_ = self.dynamics(b_last, a.view(-1), perturbation_vt, perturbation_wt)
        A = self.A_t(b_last)
        P_ = A @ P_last @ A.T + Q_t
        
        # calculate K_t
        y_t = o_t - self.H @ b_
        S_t = self.H @ P_ @ self.H.T + R_t
        K_t = P_ @ self.H.T @ S_t.inverse()
        
        # update b_t and P_t
        b_t = b_ + K_t @ y_t
        P_t = (torch.eye(self.STATE_DIM) - K_t @ self.H) @ P_
        b = b_t, P_t
        return b
    
    def dynamics(self, b_last, a, perturbation_vt, perturbation_wt):
        px, py, heading_angle, lin_vel, ang_vel = torch.split(b_last.view(-1), 1)

        px = px + lin_vel * torch.cos(heading_angle) * self.DT 
        py = py + lin_vel * torch.sin(heading_angle) * self.DT 
        heading_angle = heading_angle + ang_vel * self.DT 
        lin_vel = torch.tensor(0) * lin_vel + self.pro_gains[0] * a[0] + perturbation_vt
        ang_vel = torch.tensor(0) * ang_vel + self.pro_gains[1] * a[1] + perturbation_wt
        
        return torch.stack([px, py, heading_angle, lin_vel, ang_vel]).view([-1, 1])
    
    def A_t(self, b_last):
        px, py, heading_angle, lin_vel, ang_vel = torch.split(b_last.view(-1), 1)
        A_t = torch.zeros(self.STATE_DIM, self.STATE_DIM)
        A_t[:self.POS_DIM, :self.POS_DIM] = torch.eye(self.POS_DIM)
        A_t[0, 2] = - lin_vel * torch.sin(heading_angle) * self.DT
        A_t[0, 3] = torch.cos(heading_angle) * self.DT
        A_t[1, 2] = lin_vel * torch.cos(heading_angle) * self.DT
        A_t[1, 3] = torch.sin(heading_angle) * self.DT
        A_t[2, 4] = self.DT
        return A_t


class ActionNoise():
    def __init__(self, ACTION_DIM, mean, std):
        self.mu = torch.ones(ACTION_DIM) * mean
        self.std = std
        self.ACTION_DIM = ACTION_DIM

    def reset(self, mean, std):
        self.mu = torch.ones(self.ACTION_DIM) * mean
        self.std = std

    def noise(self):
        n = torch.randn(self.ACTION_DIM)
        return self.mu + self.std * n


class Agent():
    def __init__(self, arg, Actor, Critic):
        self.__dict__ .update(arg.__dict__)

        self.actor = Actor(self.EKF_STATE_DIM, self.ACTION_DIM, self.FC_SIZE).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()
        self.actor_optim = self.optimizer(self.actor.parameters(), lr=self.lr, eps=self.eps)
        
        self.critic = Critic(self.OBS_DIM, self.ACTION_DIM, self.TARGET_DIM, 
                             self.RNN_SIZE, self.FC_SIZE, self.RNN).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.eval()
        self.critic_optim = self.optimizer(self.critic.parameters(), lr=self.lr, eps=self.eps)
        
        self.memory = ReplayMemory(arg.MEMORY_SIZE, arg.BATCH_SIZE)
        self.bstep = BeliefStep(arg)
        
        self.initial_episode = 0
        self.it = 0


    def select_action(self, EKFstate, action_noise=None):            
        with torch.no_grad():
            action = self.actor(EKFstate)
            
        action = action.cpu()
        action_raw = action.clone()
        if (action_noise is not None) and (action_raw.abs() > self.TERMINAL_ACTION).any():
            action += action_noise.noise().view_as(action)

        return action.clamp(-1, 1), action_raw
    
    def target_smoothing(self, next_actions):
        mask_stop = (next_actions.view(-1, self.ACTION_DIM).abs().max(dim=1).values < self.TERMINAL_ACTION
                        ).view(-1, 1).repeat(1, self.ACTION_DIM).view_as(next_actions)
        mask_nonstop_pos = (next_actions > self.TERMINAL_ACTION) & (~mask_stop)
        mask_nonstop_neg = (next_actions < -self.TERMINAL_ACTION) & (~mask_stop)
        mask_nonstop_other = (next_actions.abs() < self.TERMINAL_ACTION) & (~mask_stop)

        next_actions[mask_stop] = (next_actions[mask_stop]                         + torch.zeros_like(next_actions[mask_stop]).normal_(
                                                mean=0, std=self.policy_noise)
                        ).clamp(-self.TERMINAL_ACTION, self.TERMINAL_ACTION)

        next_actions[mask_nonstop_pos] = (next_actions[mask_nonstop_pos]                         + torch.zeros_like(next_actions[mask_nonstop_pos]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(self.TERMINAL_ACTION, 1)

        next_actions[mask_nonstop_neg] = (next_actions[mask_nonstop_neg]                         + torch.zeros_like(next_actions[mask_nonstop_neg]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(-1, -self.TERMINAL_ACTION)

        next_actions[mask_nonstop_other] = (next_actions[mask_nonstop_other]                         + torch.zeros_like(next_actions[mask_nonstop_other]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(-1, 1)
        
        return next_actions

    def update_parameters(self, batch):
        states = torch.cat(batch.state, dim=1)
        EKFstates = torch.cat(batch.EKFstate, dim=1)
        actions =  torch.cat(batch.action, dim=1)
        rewards = torch.cat(batch.reward, dim=1)
        dones = torch.zeros_like(rewards)
        dones[-1] = 1
        
        with torch.no_grad():
            # get next action and apply target policy smoothing
            next_states = torch.zeros_like(states)
            next_states[:-1] = states[1:]
            next_EKFstates = torch.zeros_like(EKFstates)
            next_EKFstates[:-1] = EKFstates[1:]
            next_actions = self.target_actor(next_EKFstates)
            next_actions = self.target_smoothing(next_actions)

            # compute the target Q
            _, _, t1_hidden1, t1_hidden2 = self.target_critic(states[:1], actions[:1], return_hidden=True)
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions, 
                                                      hidden_in1=t1_hidden1, hidden_in2=t1_hidden2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1-dones) * self.GAMMA * target_Q

        # current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # delay policy updates
        if self.it % self.POLICY_FREQ == 0:
            # define actor loss
            actor_loss = - self.critic.Q1(states, self.actor(EKFstates)).mean()
            
            # optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)
        else:
            actor_loss = torch.tensor([0])

        return actor_loss.detach().item(), critic_loss.detach().item()

    def learn(self):
        batch = self.memory.sample()
        loss_logs = self.update_parameters(batch)
        self.it += 1
        return loss_logs

    def save(self, save_memory, episode, pre_phase=False, full_param=True):
        if pre_phase:
            file = self.data_path / f'{self.filename}-{episode}_pre.pth.tar'
        else:
            file = self.data_path / f'{self.filename}-{episode}.pth.tar'
            
        state = {'actor_dict': self.actor.state_dict(),
                 'critic_dict': self.critic.state_dict()}
        if full_param:
            state.update({'target_actor_dict': self.target_actor.state_dict(),
                          'target_critic_dict': self.target_critic.state_dict(),
                          'actor_optimizer_dict': self.actor_optim.state_dict(),
                          'critic_optimizer_dict': self.critic_optim.state_dict(),
                          'episode': episode})
        if save_memory:
            state['memory'] = (self.memory.memory, self.memory.position)

        torch.save(state, file)

    def load(self, filename, load_memory, load_optimzer, full_param=False):
        self.filename = filename
        file = self.data_path / f'{self.filename}.pth.tar'
        state = torch.load(file)

        self.actor.load_state_dict(state['actor_dict'])
        self.critic.load_state_dict(state['critic_dict'])
        if full_param:
            self.target_actor.load_state_dict(state['target_actor_dict'])
            self.target_critic.load_state_dict(state['target_critic_dict'])
            self.initial_episode = state['episode']
        
        if load_memory is True:
            self.memory.load(state['memory'])
        if load_optimzer is True:
            self.actor_optim.load_state_dict(state['actor_optimizer_dict'])
            self.critic_optim.load_state_dict(state['critic_optimizer_dict'])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.TAU) + param.data * self.TAU)
            
    def mirror_traj(self, states, EKFstates, actions, mirrored_index=(1, 3, 4), mirrored_index_EKF=(0, 4, 6, 9)):
        # state index 1: w; 3: action aw; 4: target x
        states_ = states.clone()
        states_[..., mirrored_index] = - states_[..., mirrored_index]
        
        # 0: x_mean; 4: w_mean; 6: cov(x, y); 9: cov(y, theta)
        EKFstates_ = EKFstates.clone()
        EKFstates_[..., mirrored_index_EKF] = - EKFstates_[..., mirrored_index_EKF]
        EKFstates_[..., 2] = np.pi - EKFstates_[..., 2]  # 2: head direction
        
        # 1 of action indexes angular action aw
        actions_ = actions.clone()
        actions_[..., 1] = - actions_[..., 1]
        
        return states_, EKFstates_, actions_
    

