#!/usr/bin/env python
# coding: utf-8


import config.config as config
import pandas as pd
from Environment import Env
import torch
import numpy as np


class Validation():
    def __init__(self, task, agent_name, validation_size=300, 
                 target_positions=None, perturbs_info=None, 
                 enable_noise=False, save_freq=500, extensive=False):
        self.task = task
        self.enable_noise = enable_noise
        if 'gain' in task:
            if extensive:
                self.task_sets = [f'gain{i}x' for i in [1, 2, 3, 4]]
            else:
                self.task_sets = [f'gain{i}x' for i in [1, 1.5, 2]]
        elif 'perturbation' in task:
            if extensive:
                self.task_sets = [f'perturbation{i}' for i in range(3)]
            else:
                self.task_sets = [f'perturbation{i}' for i in range(2)]
        else:
            raise ValueError('No such a task.')

        self.agent_name = agent_name
        self.validation_size = validation_size
        self.data = pd.DataFrame(columns=['episode', 'task', 'reward_fraction', 'error_distance',
                                          'error_distance_shoot', 'reward_rate', 'TD_error_abs', 'TD_error',
                                          'value_est', 'ordered_episode'])
        self.ordered_episode = -1
        self.save_freq = save_freq
        
        if target_positions is not None:
            self.target_positions = target_positions
            self.perturbation_velocitiess, self.perturbation_start_ts, self.perturbation_velocitiess_large = perturbs_info
        else:
            self.target_positions = self.perturbation_velocitiess = self.perturbation_start_ts = [None] * validation_size
            self.perturbation_velocitiess_large = [None] * validation_size
        
    def __call__(self, agent, episode):
        self.ordered_episode += self.save_freq
        if 'RNN' in self.agent_name:
            self.validation_rnn(agent, episode)
        elif self.agent_name == 'EKF':
            self.validation_ekf(agent, episode)
        else:
            raise ValueError('No such an agent.')
            
        return self.data
        
    def validation_rnn(self, agent, episode):
        for task in self.task_sets:
            if 'gain' in task:
                if 'control' in self.task:
                    arg = config.ConfigGainControl(pro_noise=agent.pro_noise_range[0],
                                                   obs_noise=agent.obs_noise_range[0])
                else:
                    arg = config.ConfigGain()
                gain = float(task.replace('gain', '').replace('x', ''))
                arg.process_gain_range = [gain, gain]
                if not self.enable_noise:
                    arg.pro_noise_range = None
                    arg.obs_noise_range = None
                perturbation_velocitiess = self.perturbation_velocitiess
            elif 'perturbation' in task:
                arg = config.ConfigPerturb()
                if '0' in task:
                    perturbation_velocitiess = [torch.zeros(2) for _ in range(len(self.perturbation_velocitiess))]
                elif '1' in task:
                    perturbation_velocitiess = self.perturbation_velocitiess
                else:
                    perturbation_velocitiess = self.perturbation_velocitiess_large
                
            arg.device = agent.device
            arg.target_fullon = agent.target_fullon
            env = Env(arg)
                
            rewarded_number_log = 0
            dist_log = 0
            dist_log_shoot = 0
            tot_t = 0
            td_errors = 0
            td_errors_abs = 0
            values_est = 0
        
            for target_position, perturbation_velocities, perturbation_start_t in zip(self.target_positions, 
                                                                                      perturbation_velocitiess,
                                                                                      self.perturbation_start_ts):
                cross_start_threshold = False
                reward = torch.zeros(1, 1, 1)
                x = env.reset(target_position=target_position, perturbation_velocities=perturbation_velocities, 
                              perturbation_start_t=perturbation_start_t)
                agent.bstep.reset(env.pro_gains)
                last_action = torch.zeros([1, 1, arg.ACTION_DIM])
                last_action_raw = last_action.clone()

                state = torch.cat([x[-arg.OBS_DIM:].view(1, 1, -1), last_action,
                           env.target_position_obs.view(1, 1, -1)], dim=2).to(arg.device)

                hiddenin_self = None
                
                states = []; actions = []; rewards = []
                    
                for t in range(arg.EPISODE_LEN):
                    if not cross_start_threshold and (last_action_raw.abs() > arg.TERMINAL_ACTION).any():
                        cross_start_threshold = True
                        
                    action, action_raw, hiddenout_self = agent.select_action(state, hiddenin_self,
                                                                             action_noise=None)
                    next_x, reached_target, relative_dist = env(x, action, t)
                    next_ox = agent.bstep(next_x)
                    next_state = torch.cat([next_ox.view(1, 1, -1), action,
                                            env.target_position_obs.view(1, 1, -1)], dim=2).to(arg.device)

                    is_stop = env.is_stop(x, action)
                    
                    if is_stop and cross_start_threshold:
                        reward = env.return_reward(x, reward_mode='mixed')
                    
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    
                    if is_stop and cross_start_threshold:
                        break

                    last_action_raw = action_raw
                    last_action = action
                    state = next_state
                    x = next_x
                    hiddenin_self = hiddenout_self

                #log stuff 
                rewarded_number_log += int(reached_target & is_stop)
                dist_log += relative_dist.item()
                tot_t += t
                
                # compute over-/under-shoot
                d1 = np.sqrt(env.target_position[0] ** 2 + env.target_position[1] ** 2)
                r1 = (env.target_position[0] ** 2 + env.target_position[1] ** 2) / (2 * env.target_position[0])
                radian1 = 2 * r1 * np.arcsin(d1 / (2 * r1))

                x_end = x[0]; y_end = x[1]
                d2 = np.sqrt(x_end ** 2 + y_end ** 2)
                r2 = (x_end ** 2 + y_end ** 2) / (2 * x_end)
                radian2 = 2 * r2 * np.arcsin(d2 / (2 * r2))

                sign = -1 if radian2 < radian1 else 1
                dist_log_shoot += sign * relative_dist.item()
                
                # compute TD errors and estimated value
                states = torch.cat(states)
                actions = torch.cat(actions).to(arg.device)
                rewards = torch.cat(rewards).to(arg.device)
                dones = torch.zeros_like(rewards); dones[-1] = 1
                with torch.no_grad():
                    Q = agent.critic.Q1(states, actions)
                    next_Q = torch.zeros_like(Q); next_Q[:-1] = Q[1:]
                    td_error = rewards + (1-dones) * arg.GAMMA * next_Q - Q
                    
                td_errors += td_error.mean().item(); td_errors_abs += abs(td_error).mean().item();
                values_est += Q.mean().item()
                    

            self.data = self.data.append(
                        pd.DataFrame({'episode': [episode], 'task': [task],
                                      'reward_fraction': [rewarded_number_log / self.validation_size], 
                                      'error_distance': [dist_log * arg.LINEAR_SCALE / self.validation_size],
                                      'error_distance_shoot': [dist_log_shoot * arg.LINEAR_SCALE /
                                                               self.validation_size],
                                      'reward_rate': [rewarded_number_log / (tot_t * arg.DT)],
                                      'TD_error_abs': [td_errors_abs / self.validation_size],
                                      'TD_error': [td_errors / self.validation_size],
                                       'value_est': [values_est / self.validation_size],
                                      'ordered_episode': [self.ordered_episode]}), 
                                      ignore_index=True)
            
            
        
    def validation_ekf(self, agent, episode):
        for task in self.task_sets:
            if 'gain' in task:
                if 'control' in self.task:
                    arg = config.ConfigGainControl(pro_noise=agent.pro_noise_range[0],
                                                   obs_noise=agent.obs_noise_range[0])
                else:
                    arg = config.ConfigGain()
                gain = float(task.replace('gain', '').replace('x', ''))
                arg.process_gain_range = [gain, gain]
                if not self.enable_noise:
                    arg.pro_noise_range = None
                    arg.obs_noise_range = None
                perturbation_velocitiess = self.perturbation_velocitiess
            elif 'perturbation' in task:
                arg = config.ConfigPerturb()
                if '0' in task:
                    perturbation_velocitiess = [torch.zeros(2) for _ in range(len(self.perturbation_velocitiess))]
                elif '1' in task:
                    perturbation_velocitiess = self.perturbation_velocitiess
                else:
                    perturbation_velocitiess = self.perturbation_velocitiess_large

            arg.device = agent.device
            env = Env(arg)

            rewarded_number_log = 0
            dist_log = 0
            dist_log_shoot = 0
            tot_t = 0
            td_errors = 0
            td_errors_abs = 0
            values_est = 0
            
            for target_position, perturbation_velocities, perturbation_start_t in zip(self.target_positions, 
                                                                                      perturbation_velocitiess,
                                                                                      self.perturbation_start_ts):
                cross_start_threshold = False
                reward = torch.zeros(1, 1, 1)
                x = env.reset(target_position=target_position, perturbation_velocities=perturbation_velocities, 
                              perturbation_start_t=perturbation_start_t)
                b, EKFstate = agent.bstep.reset(env.pro_gains, env.pro_noise_std, env.target_position)
                EKFstate = EKFstate.to(arg.device).unsqueeze(0)
                
                last_action = torch.zeros(1, 1, arg.ACTION_DIM)
                last_action_raw = last_action.clone()

                state = torch.cat([x[-arg.OBS_DIM:].view(1, 1, -1), last_action,
                                   env.target_position_obs.view(1, 1, -1)], dim=2).to(arg.device)
                
                states = []; EKFstates = []; actions = []; rewards = []

                for t in range(arg.EPISODE_LEN):
                    if not cross_start_threshold and (last_action_raw.abs() > arg.TERMINAL_ACTION).any():
                        cross_start_threshold = True
                        
                    action, action_raw = agent.select_action(EKFstate, action_noise=None)
                    next_x, reached_target, relative_dist = env(x, action, t)
                    next_ox = agent.bstep.observation(next_x)
                    next_state = torch.cat([next_ox.view(1, 1, -1), action,
                                            env.target_position_obs.view(1, 1, -1)], dim=2).to(arg.device)
                    next_b = agent.bstep(b, next_ox, action, env.perturbation_vt, env.perturbation_wt)
                    next_EKFstate = agent.bstep.b_reshape(next_b).to(arg.device).unsqueeze(0)

                    is_stop = env.is_stop(x, action)  
                    
                    if is_stop and cross_start_threshold:
                        reward = env.return_reward(x, reward_mode='mixed')

                    states.append(state)
                    EKFstates.append(EKFstate)
                    actions.append(action)
                    rewards.append(reward)
                    
                    if is_stop and cross_start_threshold:
                        break

                    last_action_raw = action_raw
                    last_action = action
                    state = next_state
                    EKFstate = next_EKFstate
                    x = next_x
                    b = next_b

                #log stuff 
                rewarded_number_log += int(reached_target & is_stop)
                dist_log += relative_dist.item()
                tot_t += t
                
                # compute over-/under-shoot
                d1 = np.sqrt(env.target_position[0] ** 2 + env.target_position[1] ** 2)
                r1 = (env.target_position[0] ** 2 + env.target_position[1] ** 2) / (2 * env.target_position[0])
                radian1 = 2 * r1 * np.arcsin(d1 / (2 * r1))

                x_end = x[0]; y_end = x[1]
                d2 = np.sqrt(x_end ** 2 + y_end ** 2)
                r2 = (x_end ** 2 + y_end ** 2) / (2 * x_end)
                radian2 = 2 * r2 * np.arcsin(d2 / (2 * r2))

                sign = -1 if radian2 < radian1 else 1
                dist_log_shoot += sign * relative_dist.item()
                
                # compute TD errors and estimated value
                states = torch.cat(states)
                EKFstates = torch.cat(EKFstates)
                actions = torch.cat(actions).to(arg.device)
                rewards = torch.cat(rewards).to(arg.device)
                dones = torch.zeros_like(rewards); dones[-1] = 1
                with torch.no_grad():
                    if agent.critic.l1.in_features == arg.EKF_STATE_DIM + arg.ACTION_DIM:
                        Q = agent.critic.Q1(EKFstates, actions)
                    else:
                        Q = agent.critic.Q1(states, actions)
                    next_Q = torch.zeros_like(Q); next_Q[:-1] = Q[1:]
                    td_error = rewards + (1-dones) * arg.GAMMA * next_Q - Q
                    
                td_errors += td_error.mean().item(); td_errors_abs += abs(td_error).mean().item();
                values_est += Q.mean().item()
                
            
            self.data = self.data.append(
                        pd.DataFrame({'episode': [episode], 'task': [task],
                                      'reward_fraction': [rewarded_number_log / self.validation_size], 
                                      'error_distance': [dist_log * arg.LINEAR_SCALE / self.validation_size],
                                      'error_distance_shoot': [dist_log_shoot * arg.LINEAR_SCALE /
                                                               self.validation_size],
                                      'reward_rate': [rewarded_number_log / (tot_t * arg.DT)],
                                      'TD_error_abs': [td_errors_abs / self.validation_size],
                                      'TD_error': [td_errors / self.validation_size],
                                       'value_est': [values_est / self.validation_size],
                                      'ordered_episode': [self.ordered_episode]}), 
                                      ignore_index=True)

