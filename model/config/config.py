import datetime
import torch
from torch.nn import LSTM
import numpy as np
from torch_optimizer import RAdam
from pathlib import Path


class ConfigCore():
    def __init__(self, data_path='./'):
        # network
        self.device = 'cuda:0'
        self.SEED_NUMBER = 0
        self.FC_SIZE = 300
        
        # RL
        self.GAMMA = 0.97
        self.TAU = 0.005
        self.POLICY_FREQ = 2
        self.policy_noise = 0.05
        self.policy_noise_clip = 0.1
        
        # optimzer
        self.optimzer = RAdam
        self.lr = 3e-4
        self.eps = 1.5e-4
        self.decayed_lr = 5e-5
        
        # environment
        self.STATE_DIM = 5
        self.ACTION_DIM = 2
        self.POS_DIM = 3
        self.OBS_DIM = 2
        self.TARGET_DIM = 2
        self.TERMINAL_ACTION = 0.1
        self.DT = 0.1 # s
        self.EPISODE_TIME = 3.5 # s
        self.EPISODE_LEN = int(self.EPISODE_TIME / self.DT)
        self.REWARD_SCALE = 10
        self.LINEAR_SCALE = 400 # cm/unit
        self.goal_radius_range = np.array([65, 65]) / self.LINEAR_SCALE
        self.initial_radius_range = np.array([100, 400]) / self.LINEAR_SCALE
        self.relative_angle_range = np.deg2rad([-35, 35])
        self.process_gain_default = torch.tensor([200 / self.LINEAR_SCALE, torch.deg2rad(torch.tensor(90.))])
        self.get_self_action = True
        self.target_fullon = False
        self.target_offT = 3 # steps
        
        # others
        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_path = data_path
        self.freeze_belief = False
        self.freeze_policy = False
        
        # For model-free belief
        self.BATCH_SIZE = 16
        self.MEMORY_SIZE = int(1e5)
        self.RNN = LSTM
        self.RNN_SIZE = 128
        
        # For model-based belief
        self.EKF_STATE_DIM = 13
        self.EKF_BATCH_SIZE = 256
        self.EKF_MEMORY_SIZE = int(1.6e6)
        
    def save(self):
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, self.data_path / f'{self.filename}_arg.pkl')
        
    def load(self, filename):
        self.__dict__ = torch.load(self.data_path / f'{filename}_arg.pkl')
        self.filename = filename

        
class ConfigNoise(ConfigCore):
    def __init__(self, data_path='./'):
        super().__init__(data_path)
        self.task = 'noise'
        self.process_gain_range = None
        self.pro_noise_range = [0.3, 0.3] # proportional to process gain
        self.obs_noise_range = [0, 1]
        self.perturbation_velocity_range = None
        self.perturbation_duration = None
        self.perturbation_std = None
        self.perturbation_start_t_range = None
        
        
class ConfigNoiseControl(ConfigNoise):
    def __init__(self, data_path='./', task='obs'):
        super().__init__(data_path)
        if task == 'obs':
            self.task = 'noise_control_obs'
            self.pro_noise_range = [0.8, 0.8]
            self.obs_noise_range = [0, 0]
        elif task == 'pro':
            self.task = 'noise_control_pro'
            self.pro_noise_range = [0, 0]
            self.obs_noise_range = [0.8, 0.8]
        else:
            raise ValueError('No such a task!')
        
        
class ConfigGain(ConfigCore):
    def __init__(self, data_path='./', gain_distribution='uniform', get_self_action=True,
                 exclude_gain=None):
        super().__init__(data_path)
        self.task = 'gain'
        self.gain_distribution = gain_distribution
        self.process_gain_range = [1, 1]
        self.exclude_gain = exclude_gain
        self.get_self_action = get_self_action
        self.pro_noise_range = [0.2, 0.2]
        self.obs_noise_range = [0.1, 0.1]
        self.perturbation_velocity_range = None
        self.perturbation_duration = None
        self.perturbation_std = None
        self.perturbation_start_t_range = None
        
        
class ConfigGainControl(ConfigGain):
    def __init__(self, data_path='./', pro_noise=0.2, obs_noise=0.1):
        super().__init__(data_path)
        self.task = 'gain_control'
        self.pro_noise_range = [pro_noise] * 2
        self.obs_noise_range = [obs_noise] * 2
        
        
class ConfigGainFreezeBelief(ConfigGain):
    def __init__(self, data_path='./'):
        super().__init__(data_path)
        self.freeze_belief = True
        
        
class ConfigGainFreezePolicy(ConfigGain):
    def __init__(self, data_path='./'):
        super().__init__(data_path)
        self.freeze_policy = True
        
        
class ConfigPerturb(ConfigCore):
    def __init__(self, data_path='./', exclude_perturb=None):
        super().__init__(data_path)
        self.task = 'perturbation'
        self.process_gain_range = None
        self.pro_noise_range = [0.2, 0.2]
        self.obs_noise_range = [0.1, 0.1]
        self.perturbation_velocity_range = np.hstack([np.array([-200, 200]) / self.LINEAR_SCALE, 
                                                      np.deg2rad([-120, 120])])
        self.perturbation_velocity_range_large = np.hstack([np.array([-200, 800]) / self.LINEAR_SCALE, 
                                                            np.deg2rad([-180, 180])])
        self.exclude_perturb = exclude_perturb
        self.perturbation_duration = 10 # steps
        self.perturbation_std = 2 # steps
        self.perturbation_start_t_range = [0, 11] # steps
        
        
class ConfigPerturbControl(ConfigPerturb):
    def __init__(self, data_path='./', pro_noise=0.2, obs_noise=0.1):
        super().__init__(data_path)
        self.task = 'perturbation_control'
        self.pro_noise_range = [pro_noise] * 2
        self.obs_noise_range = [obs_noise] * 2