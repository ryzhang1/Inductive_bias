#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import config.config as config
import pandas as pd
from Environment import Env
import torch


# In[ ]:


class Validation():
    def __init__(self, task, agent_name, validation_size=200, 
                 target_positions=None, perturbs_info=None, 
                 enable_noise=False, save_freq=500):
        self.task = task
        self.enable_noise = enable_noise
        if 'noise' in task:
            if '_' in task:
                self.task_sets = [task]
            else:
                self.task_sets = [f'noise{i}' for i in [0, 0.4, 0.8]]
        elif 'normal' in task:
            self.task_sets = ['gain1x']
        elif 'gain' in task:
            self.task_sets = [f'gain{i}x' for i in range(1, 5)]
        elif 'perturbation' in task:
            self.task_sets = [f'perturbation{i}' for i in range(3)]
        else:
            raise ValueError('No such a task.')

        self.agent_name = agent_name
        self.validation_size = validation_size
        self.data = pd.DataFrame(columns=['episode', 'task', 'reward_fraction', 'error_distance',
                                          'reward_rate', 'ordered_episode'])
        self.ordered_episode = - 1
        self.save_freq = save_freq
        
        if target_positions is not None:
            self.target_positions = target_positions
            self.perturbation_velocitiess, self.perturbation_start_ts, self.perturbation_velocitiess_large = perturbs_info
        else:
            self.target_positions = self.perturbation_velocitiess = self.perturbation_start_ts = [None] * validation_size
            self.perturbation_velocitiess_large = [None] * validation_size
        
    def __call__(self, agent, episode):
        self.ordered_episode += self.save_freq
        if 'LSTM' in self.agent_name:
            self.validation_lstm(agent, episode)
        elif self.agent_name == 'EKF':
            self.validation_ekf(agent, episode)
        else:
            raise ValueError('No such an agent.')
            
        return self.data
        
    def validation_lstm(self, agent, episode):
        for task in self.task_sets:
            if 'gain' in task:
                if 'control' in self.task:
                    arg = config.ConfigGainControl(pro_noise=agent.pro_noise_range[0], obs_noise=agent.obs_noise_range[0])
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
            elif 'noise' in task:
                if '_' in task:
                    arg = config.ConfigNoiseControl(task=task.split('_')[2])
                else:
                    arg = config.ConfigNoise()
                    obs_noise = float(task.replace('noise', ''))
                    arg.obs_noise_range = [obs_noise, obs_noise]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                perturbation_velocitiess = self.perturbation_velocitiess
                
            arg.device = agent.device
            arg.target_fullon = agent.target_fullon
            env = Env(arg)
                
            rewarded_number_log = 0
            dist_log = 0
            tot_t = 0
        
            for target_position, perturbation_velocities, perturbation_start_t in zip(self.target_positions, 
                                                                                      perturbation_velocitiess,
                                                                                      self.perturbation_start_ts):
                cross_start_threshold = False
                x = env.reset(target_position=target_position, perturbation_velocities=perturbation_velocities, 
                              perturbation_start_t=perturbation_start_t)
                agent.bstep.reset(env.pro_gains)
                last_action = torch.zeros([1, 1, arg.ACTION_DIM])
                last_action_raw = last_action.clone()

                state = torch.cat([x[-arg.OBS_DIM:].view(1, 1, -1), last_action,
                           env.target_position_obs.view(1, 1, -1)], dim=2).to(arg.device)

                hiddenin_self = None
                    
                for t in range(arg.EPISODE_LEN):
                    if not cross_start_threshold and (last_action_raw.abs() > arg.TERMINAL_ACTION).any():
                        cross_start_threshold = True
                        
                    action, action_raw, hiddenout_self = agent.select_action(state, hiddenin_self, action_noise=None)
                    next_x, reached_target, relative_dist = env(x, action, t)
                    next_ox = agent.bstep(next_x)
                    next_state = torch.cat([next_ox.view(1, 1, -1), action,
                                            env.target_position_obs.view(1, 1, -1)], dim=2).to(arg.device)

                    is_stop = env.is_stop(x, action)

                    last_action_raw = action_raw
                    last_action = action
                    state = next_state
                    x = next_x
                    hiddenin_self = hiddenout_self

                    if is_stop and cross_start_threshold:
                        break

                #log stuff 
                rewarded_number_log += int(reached_target & is_stop)
                dist_log += relative_dist.item()
                tot_t += t

            self.data = self.data.append(
                        pd.DataFrame({'episode': [episode], 'task': [task],
                                      'reward_fraction': [rewarded_number_log / self.validation_size], 
                                      'error_distance': [dist_log * arg.LINEAR_SCALE / self.validation_size],
                                      'reward_rate': [rewarded_number_log / (tot_t * arg.DT)],
                                      'ordered_episode': [self.ordered_episode]}), 
                                      ignore_index=True)
            
        
    def validation_ekf(self, agent, episode):
        for task in self.task_sets:
            if 'gain' in task:
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
            elif 'noise' in task:
                arg = config.ConfigNoise()
                obs_noise = float(task.replace('noise', ''))
                arg.obs_noise_range = [obs_noise, obs_noise]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                perturbation_velocitiess = self.perturbation_velocitiess

            arg.device = agent.device
            env = Env(arg)

            rewarded_number_log = 0
            dist_log = 0
            tot_t = 0
            
            for target_position, perturbation_velocities, perturbation_start_t in zip(self.target_positions, 
                                                                                      perturbation_velocitiess,
                                                                                      self.perturbation_start_ts):
                cross_start_threshold = False
                x = env.reset(target_position=target_position, perturbation_velocities=perturbation_velocities, 
                              perturbation_start_t=perturbation_start_t)
                b, state = agent.bstep.reset(env.pro_gains, env.pro_noise_std, env.target_position)
                state = state.to(arg.device)
                last_action_raw = torch.zeros(1, arg.ACTION_DIM)

                for t in range(arg.EPISODE_LEN):
                    if not cross_start_threshold and (last_action_raw.abs() > arg.TERMINAL_ACTION).any():
                        cross_start_threshold = True
                        
                    action, action_raw = agent.select_action(state, action_noise=None)
                    next_x, reached_target, relative_dist = env(x, action, t)
                    next_ox = agent.bstep.observation(next_x)
                    next_b = agent.bstep(b, next_ox, action, env.perturbation_vt, env.perturbation_wt)
                    next_state = agent.bstep.b_reshape(next_b).to(arg.device)

                    is_stop = env.is_stop(x, action)

                    last_action_raw = action_raw
                    state = next_state
                    x = next_x
                    b = next_b

                    if is_stop and cross_start_threshold:
                        break

                #log stuff 
                rewarded_number_log += int(reached_target & is_stop)
                dist_log += relative_dist.item()
                tot_t += t

            self.data = self.data.append(
                        pd.DataFrame({'episode': [episode], 'task': [task],
                                      'reward_fraction': [rewarded_number_log / self.validation_size], 
                                      'error_distance': [dist_log * arg.LINEAR_SCALE / self.validation_size],
                                      'reward_rate': [rewarded_number_log / (tot_t * arg.DT)],
                                      'ordered_episode': [self.ordered_episode]}), 
                                      ignore_index=True)


# In[ ]:




