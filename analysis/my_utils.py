import numpy as np
from numpy.random import default_rng
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from contextlib import contextmanager
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from astropy.convolution import convolve
import warnings
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression

import sys; sys.path.append('../model')
import config.config as config

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def deepcopy_pd(df, is_df=False):
    func = pd.DataFrame if is_df else pd.Series
    return func(deepcopy(df.to_dict()))


def param_prior_range():
    param = {'target_theta': [55, 125], 'target_dist': [100, 400],
             'perturb_vpeak': [-200, 200], 'perturb_wpeak': [-120, 120],
             'perturb_start_time_ori': [0, 1], 'SAMPLING_RATE': 1 / 833,
             'gain_v': [200, 400], 'gain_w': [90, 180]}
    
    param['target_x'] = param['target_dist'][1] * np.cos(np.deg2rad(param['target_theta']))[::-1]
    param['target_y'] = np.array(param['target_dist']) * np.array([np.sin(np.deg2rad(param['target_theta'][0])), 1])
    return param


def min_max_scaling(data, d_range=None):
    d_range = [data.min(), data.max()] if d_range is None else d_range
    
    return (data - d_range[0]) / (d_range[1] - d_range[0])


def min_max_scale_df(df):
    scaled_data = []
    for variable, data in df.iteritems():
        scaled_data.append(min_max_scaling(data, d_range=param_prior_range()[variable]))
        
    return pd.concat(scaled_data, axis=1)


def cartesian_prod(*args):
    return list(np.stack(np.meshgrid(*args), axis=-1).reshape(-1, len(args)))


def get_relative_r_ang(px, py, heading_angle, target_x, target_y):
    heading_angle = np.deg2rad(heading_angle)
    distance_vector = np.vstack([px - target_x, py - target_y])
    relative_r = np.linalg.norm(distance_vector, axis=0)
    
    relative_ang = heading_angle - (np.arctan2(py, px) - np.arctan2(target_y, target_x))
    return relative_r, relative_ang

        
@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['pdf.fonttype'] = '42'
    plt.rcParams["font.family"] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['mathtext.default'] = 'it'
    plt.rcParams['mathtext.fontset'] = 'custom'
    fig = plt.figure(dpi=dpi)
    yield fig
    plt.show()
    
    
def set_violin_plot(vp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-', hatch=r''):
    plt.setp(vp['bodies'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls, hatch=hatch)
    plt.setp(vp['cmins'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha)
    plt.setp(vp['cmaxes'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha)
    plt.setp(vp['cbars'], facecolor='None', edgecolor='None', 
             linewidth=linewidth, alpha=alpha)
    
    linecolor = 'k' if facecolor == 'None' else 'snow'
    if 'cmedians' in vp:
        plt.setp(vp['cmedians'], facecolor=linecolor, edgecolor=linecolor, 
                 linewidth=linewidth, alpha=alpha, ls=ls)
    if 'cmeans' in vp:
        plt.setp(vp['cmeans'], facecolor=linecolor, edgecolor=linecolor, 
                 linewidth=linewidth, alpha=alpha, ls=ls)
    
    
def filter_fliers(data, whis=1.5, return_idx=False):
    if not isinstance(data, list):
        data = [data]
    filtered_data = []; data_ides = []
    for value in data:
        Q1, Q2, Q3 = np.percentile(value, [25, 50, 75])
        lb = Q1 - whis * (Q3 - Q1); ub = Q3 + whis * (Q3 - Q1)
        filtered_data.append(value[(value > lb) & (value < ub)])
        data_ides.append(np.where((value > lb) & (value < ub))[0])
    if return_idx:
        return filtered_data, data_ides
    else:
        return filtered_data
    
    
def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def match_similar_trials(df, reference, variables=['target_x', 'target_y'], is_scale=False, is_sort=False, 
                         EPISODE_SIZE=2000, replace=False):
    df.reset_index(drop=True, inplace=True)
    reference.reset_index(drop=True, inplace=True)
    df_trials = df.loc[:, variables].copy()
    reference_trials = reference.loc[:, variables].copy()
    if is_scale:
        df_trials, reference_trials = map(min_max_scale_df, [df_trials, reference_trials])
    
    closest_df_indices = []
    for _, reference_trial in reference_trials.iterrows():
        distance = np.linalg.norm(df_trials - reference_trial, axis=1)
        closest_df_trial = df_trials.iloc[distance.argmin()]
        closest_df_indices.append(closest_df_trial.name)
        if not replace:
            df_trials.drop(closest_df_trial.name, inplace=True)
        
    matched_df = df.loc[closest_df_indices]
    matched_df.reset_index(drop=True, inplace=True)
    
    if is_sort:
        if is_scale:
            matched_df_trials = min_max_scale_df(matched_df.loc[:, variables].copy())
        else:
            matched_df_trials = matched_df.loc[:, variables].copy()
        chosen_indices = np.linalg.norm(matched_df_trials - reference_trials, axis=1).argsort()[:EPISODE_SIZE]
        return matched_df.iloc[chosen_indices], reference.iloc[chosen_indices]
    else:
        return matched_df

    
def config_colors():
    colors = {'modular_c': 'lightseagreen', 'holistic_c': 'salmon', 
              'moact_hocri_c': '#a84499', 'hoact_mocri_c': '#183319', 'EKF_c': 'k', 
              'actor1_c': 'palevioletred', 'actor2_c': '#6f7e8f', 'actor3_c': 'darkcyan',
              'monkB_c': 'darkblue', 'monkS_c': 'olive',
              'sensory_c': '#662f8e', 'belief_c': '#ff343c', 'motor_c': '#e867c7',
              'reward_c': 'rosybrown', 'unreward_c': 'dodgerblue', 
              'pos_perturb_c': 'peru', 'neg_perturb_c': 'plum'}
    return colors


def simulate_no_generalization(df_manip, df_normal, subject):
    arg = config.ConfigCore()
    assert subject in ['monkey', 'agent'], 'Subject should be "monkey" or "agent".'
    if subject == 'agent':
        LINEAR_SCALE = arg.LINEAR_SCALE
        ang_func = np.rad2deg
        DT = arg.DT
    else:
        LINEAR_SCALE = 1
        ang_func = lambda x: x
        DT = param_prior_range()['SAMPLING_RATE']
    
    sim_vs = []; sim_ws = []; sim_heads = []
    sim_xs = []; sim_ys = []
    sim_rs = []; sim_thetas = []
    relative_rs = []; relative_angs = []
    sim_action_v = []; sim_action_w = []
    sim_perturb_start_idx = []
    sim_target_x = []; sim_target_y = []
    for (_, trial_manip), (_, trial_normal) in zip(df_manip.iterrows(), df_normal.iterrows()):
        sim_v = trial_normal.action_v * trial_manip.gain_v * LINEAR_SCALE
        sim_w = trial_normal.action_w * ang_func(trial_manip.gain_w)
        
        perturb_start_idx = np.nan
        if 'perturb_vpeak' in trial_manip.index and trial_manip.perturb_vpeak != 0:  # a perturbation trial
            if subject == 'agent':
                perturb_start_idx = trial_manip.perturb_start_time
            else:
                perturb_start_idx = round(trial_manip.perturb_start_time / DT)
            
            trial_size = max(perturb_start_idx + trial_manip.perturb_v_gauss.size, sim_v.size)
            if trial_size > sim_v.size:
                sim_v = np.hstack([sim_v, np.zeros(trial_size - sim_v.size)])
                sim_w = np.hstack([sim_w, np.zeros(trial_size - sim_w.size)])
            
            sim_v[perturb_start_idx:perturb_start_idx + trial_manip.perturb_v_gauss.size] += trial_manip.perturb_v_gauss
            sim_w[perturb_start_idx:perturb_start_idx + trial_manip.perturb_w_gauss.size] += trial_manip.perturb_w_gauss
            
        if subject == 'agent':
            sim_v = np.hstack([0, sim_v])[:-1]
            sim_w = np.hstack([0, sim_w])[:-1]
            
        sim_head = np.cumsum(sim_w) * DT + 90
        if subject == 'agent':
            sim_head = np.hstack([90, sim_head])[:-1]
            
        sim_x = np.cumsum(sim_v * np.cos(np.deg2rad(sim_head))) * DT
        sim_y = np.cumsum(sim_v * np.sin(np.deg2rad(sim_head))) * DT
        if subject == 'agent':
            sim_x = np.hstack([0, sim_x])[:-1]
            sim_y = np.hstack([0, sim_y])[:-1]
            
        
        sim_r, sim_theta = cart2pol(sim_x, sim_y)
        relative_r, relative_ang = get_relative_r_ang(sim_x, sim_y, sim_head, 
                                                      trial_manip.target_x, 
                                                      trial_manip.target_y)
        sim_vs.append(sim_v); sim_ws.append(sim_w)
        sim_heads.append(sim_head)
        sim_xs.append(sim_x); sim_ys.append(sim_y)
        sim_rs.append(sim_r)
        sim_thetas.append(np.rad2deg(sim_theta))
        relative_rs.append(relative_r)
        relative_angs.append(np.rad2deg(relative_ang))
        sim_action_v.append(trial_normal.action_v)
        sim_action_w.append(trial_normal.action_w)
        sim_perturb_start_idx.append(perturb_start_idx)
        sim_target_x.append(trial_normal.target_x)
        sim_target_y.append(trial_normal.target_y)
        
    return df_manip.assign(sim_pos_v=sim_vs, sim_pos_w=sim_ws, sim_head_dir=sim_heads,
                           sim_pos_x=sim_xs, sim_pos_y=sim_ys, 
                           sim_pos_x_end=[x[-1] for x in sim_xs],
                           sim_pos_y_end=[y[-1] for y in sim_ys],
                           sim_pos_r=sim_rs, sim_pos_theta=sim_thetas,
                           sim_pos_r_end=[r[-1] for r in sim_rs],
                           sim_pos_theta_end=[theta[-1] for theta in sim_thetas],
                           sim_relative_radius=relative_rs, sim_relative_angle=relative_angs,
                           sim_relative_radius_end=[r[-1] for r in relative_rs],
                           sim_relative_angle_end=[ang[-1] for ang in relative_angs],
                           sim_action_v=sim_action_v, sim_action_w=sim_action_w,
                           sim_perturb_start_idx=sim_perturb_start_idx,
                           sim_target_x=sim_target_x, sim_target_y=sim_target_y)


def my_tickformatter(value, pos):
    if abs(value) > 0 and abs(value) < 1:
        value = str(value).replace('0.', '.')
    elif value == 0:
        value = 0
    elif int(value) == value:
        value = int(value)
    return value


def get_radial_error(dfs, is_sim=False, copyindf=True):
    if not isinstance(dfs, list):
        dfs = [dfs]
        
    errors = []
    rel_r_key = 'relative_radius_end'
    x_key = 'pos_x'; y_key = 'pos_y'
    
    if is_sim:
        rel_r_key, x_key, y_key = ['sim_' + key for key in [rel_r_key, x_key, y_key]]
        
    for df in dfs:        
        d1 = np.sqrt(df.target_x ** 2 + df.target_y ** 2)
        r1 = (df.target_x ** 2 + df.target_y ** 2) / (2 * df.target_x)
        radian1 = 2 * r1 * np.arcsin(d1 / (2 * r1))
        
        x_end = np.array([x[-1] for x in df[x_key]]); y_end = np.array([y[-1] for y in df[y_key]])
        d2 = np.sqrt(x_end ** 2 + y_end ** 2)
        r2 = (x_end ** 2 + y_end ** 2) / (2 * x_end + 1e-8)
        radian2 = 2 * r2 * np.arcsin(d2 / (2 * r2 + 1e-8))
        
        undershoot = radian2 < radian1
        error = df[rel_r_key].copy()
        error[undershoot] = - error[undershoot]
        errors.append(error.values)
        if copyindf:
            if is_sim:
                df['sim_relative_radius_end_shoot'] = error
            else:
                df['relative_radius_end_shoot'] = error
            
    return errors


def downsample(df, enable_noise=False):
    arg = config.ConfigCore(); interval=arg.DT
    df = df.copy()
    variables = ['action_v', 'action_w']
    eps = 1e-8
    if enable_noise:
        rng = default_rng(0)
        vnoise_std, wnoise_std = arg.process_gain_default.numpy() * arg.pro_noise_range[0]
        ovnoise_std, ownoise_std = arg.process_gain_default.numpy() * arg.obs_noise_range[0]
    
    avs = []; aws = []; vs = []; ws = []; heads = []; 
    xs = []; ys = []; ts = []; ovs = []; ows = []
    for _, trial in df.iterrows():
        t_max = my_ceil(trial.time.max(), 1)
        t_steps = np.arange(0, t_max + eps, interval)
        t_binned = np.digitize(trial.time, t_steps)

        trial_variables = np.vstack(trial[variables])
        new_trial = []
        for t in np.unique(t_binned):
            new_trial.append(trial_variables[:, t_binned == t].mean(axis=1))

        new_trial = np.vstack(new_trial).T
        av, aw = np.vsplit(new_trial, len(variables))
        av = av.flatten(); aw = aw.flatten()
        end_idx = np.where((abs(av) < arg.TERMINAL_ACTION)  & 
                           (abs(aw) < arg.TERMINAL_ACTION) == False)[0][-1] + 1

        av = av[:end_idx + 1]; aw = aw[:end_idx + 1]
        v = np.hstack([0, av * trial.gain_v])[:-1] / arg.LINEAR_SCALE
        w = np.deg2rad(np.hstack([0, aw * trial.gain_w])[:-1])
        ov = v.copy(); ow = w.copy()
        if enable_noise:
            v[1:] = rng.normal(v[1:], vnoise_std)
            w[1:] = rng.normal(w[1:], wnoise_std)
            ov = np.hstack([0, rng.normal(v[1:], ovnoise_std)])
            ow = np.hstack([0, rng.normal(w[1:], ownoise_std)])
            
        head = np.cumsum(w) * interval + np.pi / 2
        head = np.hstack([np.pi / 2, head])[:-1]
        x = np.cumsum(v * np.cos(head)) * interval
        y = np.cumsum(v * np.sin(head)) * interval
        x = np.hstack([0, x])[:-1]
        y = np.hstack([0, y])[:-1]
        
        avs.append(av); aws.append(aw); vs.append(v); ws.append(w);
        heads.append(head); xs.append(x); ys.append(y); ts.append(t_steps[:-1])
        ovs.append(ov), ows.append(ow)
        
    df = df.assign(action_v_ds=avs, action_w_ds=aws, 
                   pos_v_ds=vs, pos_w_ds=ws, head_dir_ds=heads,
                   pos_x_ds=xs, pos_y_ds=ys, time_ds=ts,
                   obs_v_ds=ovs, obs_w_ds=ows)
    return df


def repeat_element(*args, time=2):
    args = [[arg] * time for arg in args]
    args = [arg for arg_ in args for arg in arg_]
    return args


def draw_arena():
    arg = config.ConfigCore()
    # arc
    target_rel_ang = np.linspace(*arg.relative_angle_range)
    rel_phi = np.pi / 2 - target_rel_ang
    target_x = arg.initial_radius_range[1] * arg.LINEAR_SCALE * np.cos(rel_phi)
    target_y = arg.initial_radius_range[1] * arg.LINEAR_SCALE * np.sin(rel_phi)
    
    # radius
    k = target_y[0] / target_x[0]
    r = k * np.linspace(0, target_x[0])
    
    # three lines
    line1 = (np.linspace(0, target_x[0]), r)
    line2 = (-np.linspace(0, target_x[0]), r)
    line3 = (target_x, target_y)
    
    
    return line1, line2, line3


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_index_upper_lower_diag(array):
    indices_upper = np.ravel_multi_index(np.triu_indices(array.shape[0], k=1), array.shape)
    indices_lower = np.ravel_multi_index(np.tril_indices(array.shape[0], k=-1), array.shape)
    indices_diag = np.ravel_multi_index(np.diag_indices_from(array), array.shape)
    return indices_upper, indices_lower, indices_diag


def get_scatter_slope(x, y):
    model = LinearRegression(fit_intercept=False)
    model.fit(x.reshape(-1, 1), y)
    return model.coef_[0]