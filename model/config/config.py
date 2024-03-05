import datetime
import torch
from torch.nn import LSTM
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from pathlib import Path


class ConfigCore():
    def __init__(self, data_path='./'):
        # network
        self.device = 'cuda:0'         # 'cuda:0' if uses gpu; otherwise 'cpu'
        self.SEED_NUMBER = 0           # seed number
        self.FC_SIZE = 300             # number of neurons in a fully connected layer (MLP)
            
        # RL
        self.GAMMA = 0.97              # discount factor in RL
        self.TAU = 0.005               # target networks update parameter (eq. (7) in the paper)
        self.POLICY_FREQ = 2           # actor update frequency; default: every 2 critic updates
        self.policy_noise = 0.05       # target policy noise std (eq. (8)/(12) in the paper)
        self.policy_noise_clip = 0.1   # target policy noise range (eq. (8)/(12) in the paper)
        
        # optimzer
        self.optimizer = Adam          # optimizer
        self.lr = 3e-4                 # learning rate
        self.eps = 1.5e-4              # eps in Adam
        self.decayed_lr = 5e-5         # decayed learning rate
        
        # environment
        self.STATE_DIM = 5             # dimension of agent's state: x, y, heading, v, w
        self.ACTION_DIM = 2            # dimension of agent's action: action_v, action_w
        self.POS_DIM = 3               # dimension of agent's position: x, y, heading
        self.OBS_DIM = 2               # dimension of agent's observation: observation_v, observation_w
        self.TARGET_DIM = 2            # dimension of target's position: target_x, target_y
        self.TERMINAL_ACTION = 0.1     # start/stop threshold
        self.DT = 0.1                  # discretization time step
        self.EPISODE_TIME = 3.5 # s    # max trial duration in seconds
        self.EPISODE_LEN = int(self.EPISODE_TIME / self.DT)   # max trial duration in steps
        self.REWARD_SCALE = 10
        self.LINEAR_SCALE = 400        # cm/unit
        self.goal_radius_range = np.array([65, 65]) / self.LINEAR_SCALE    # reward zone
        self.initial_radius_range = np.array([100, 400]) / self.LINEAR_SCALE   # range of target distance
        self.relative_angle_range = np.deg2rad([-35, 35])                      # range of target angle
        self.process_gain_default = torch.tensor([200 / self.LINEAR_SCALE, torch.deg2rad(torch.tensor(90.))]) # joystick gain
        self.target_fullon = False  # is target always visible; by default False
        self.target_offT = 3        # when target is invisible; by default target is visible in the first 300 ms
        
        # others
        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_path = data_path
        
        # For model-free belief
        self.BATCH_SIZE = 16               # batch size (number of trajectories) in each update
        self.MEMORY_SIZE = int(1e5)        # replay buffer size (number of trajectories)
        self.RNN = LSTM                    # RNN type
        self.RNN_SIZE = 128                # RNN hidden size
             
        # For model-based belief
        self.EKF_STATE_DIM = 13            # dimension of EKF state; belief mean+uncertainty
        self.EKF_BATCH_SIZE = 256          # batch size (number of transitions) in each update
        self.EKF_MEMORY_SIZE = int(1.6e6)  # replay buffer size (number of transitions)
        
    def save(self):                        # save config files
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, self.data_path / f'{self.filename}_arg.pkl')
        
    def load(self, filename):              # load config files
        self.__dict__ = torch.load(self.data_path / f'{filename}_arg.pkl')
        self.filename = filename
        
        
class ConfigGain(ConfigCore):              # specific parameters for the gain task
    def __init__(self, data_path='./', gain_distribution='uniform', exclude_gain=None):
        super().__init__(data_path)
        self.task = 'gain'
        self.process_gain_range = [1, 1]           # joystick gain range
        self.exclude_gain = exclude_gain           # excluded joystick gain range
        self.pro_noise_range = [0.2, 0.2]          # process noise std range
        self.obs_noise_range = [0.1, 0.1]          # observation noise std range
        self.perturbation_velocity_range = None    # no perturbations
        self.perturbation_duration = None
        self.perturbation_std = None
        self.perturbation_start_t_range = None
        
        
class ConfigGainControl(ConfigGain):             # specific parameters for the gain task manipulating noises
    def __init__(self, data_path='./', pro_noise=0.2, obs_noise=0.1):
        super().__init__(data_path)
        self.task = 'gain_control'
        self.pro_noise_range = [pro_noise] * 2   # process noise std range
        self.obs_noise_range = [obs_noise] * 2   # observation noise std range
        
            
class ConfigPerturb(ConfigCore):                 # specific parameters for the perturbation task
    def __init__(self, data_path='./', exclude_perturb=None):
        super().__init__(data_path)
        self.task = 'perturbation'
        self.process_gain_range = None           # do not manipulate joystick gain
        self.pro_noise_range = [0.2, 0.2]        # process noise std range
        self.obs_noise_range = [0.1, 0.1]        # observation noise std range
        # peaks for perturbation linear and angular velocities
        self.perturbation_velocity_range = np.hstack([np.array([-200, 200]) / self.LINEAR_SCALE, 
                                                      np.deg2rad([-120, 120])])
        # peaks for large perturbation linear and angular velocities
        self.perturbation_velocity_range_large = np.hstack([np.array([-200, 800]) / self.LINEAR_SCALE, 
                                                            np.deg2rad([-180, 180])])
        self.exclude_perturb = exclude_perturb    # exclude some perturbation linear and angular velocities
        self.perturbation_duration = 10           # perturbation duration 1 s
        self.perturbation_std = 2                 # decides the Gaussian shape of perturbations; fixed 
        self.perturbation_start_t_range = [0, 11] # perturbation start time
        
        
class ConfigPerturbControl(ConfigPerturb):    # specific parameters for the perturbation task manipulating noises
    def __init__(self, data_path='./', pro_noise=0.2, obs_noise=0.1):
        super().__init__(data_path)
        self.task = 'perturbation_control'
        self.pro_noise_range = [pro_noise] * 2   # process noise std range
        self.obs_noise_range = [obs_noise] * 2   # observation noise std range