# Inductive biases of neural network modularity in spatial navigation

This is a project exploring neural architectures for deep reinforcement learning agents. See [paper](https://www.science.org/doi/10.1126/sciadv.adk1256).

All the codes have been tested on Windows machines with Anaconda and CUDA-capable GPUs. The following instructions allow users to run codes in this repo based on the Windows+CUDA GPU system that has been used. However, in general, MacOS or Linux machines, with or without GPUs, should also work with slight modifications in the setup.

## Setup

Follow these steps to set up the project:

### Download Repository
1. Clone or download this repository.
2. Navigate to the main project folder; it should contain two subfolders: `analysis` and `model`.

### Download Data
1. Download data `data.zip` from [DataDryad](https://datadryad.org/stash/share/lUrgNzqfqc-dBmWDYRUfgcTe4h8MsFMOrjYrDYnVIVc).
2. Unzip the downloaded data and move the 'data' folder into the project's main folder.
3. Confirm that your project's folder now contains three subfolders: `analysis`, `model`, and `data`.
4. Inside the `data` folder, you should find various subfolders, such as `agents`, `agents_all`, `agents_temp`, `analysis_data`, `monkey_data`, `training_curve`, and `training_curve_temp`.

### Set up the Python environment
1. Download and install the [Anaconda distribution](https://www.anaconda.com/download).
2. Once Anaconda is installed, activate the Anaconda Prompt. For Windows, click Start, search for Anaconda Prompt, and open it.
3. Create a new conda environment with Python 3.8.8. You can name it whatever you like; for example, `inductivebias`. Enter the following command:
   ```bash
   conda create -n inductivebias python=3.8.8
   ```
4. Activate the created environment. If the name is `inductivebias`, enter:
   ```bash
   conda activate inductivebias
   ```
5. Install Jupyter by entering:
   ```bash
   conda install anaconda::jupyter=1.0.0
   ```
6. Install PyTorch based on your system:
   - For Windows/Linux users with a CUDA GPU:
      ```bash
      conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
      ```
   - For Windows/Linux users without a CUDA GPU:
      ```bash
      conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
      ```
   - For OSX users:
      ```bash
      conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
      ```
7. Install scikit-learn with the command:
    ```bash
    conda install anaconda::scikit-learn=0.24.1
    ```
8. Install astropy with:
    ```bash
    conda install astropy=4.2.1
    ```
9. Install pandas using:
    ```bash
    conda install anaconda::pandas=1.2.4
    ```
10. Install Matplotlib with:
    ```bash
    conda install conda-forge::matplotlib=3.5.3
    ```
11. Install scipy by entering:
    ```bash
    conda install scipy=1.10.1
    ```

## Code running
After completing the setup process, follow the steps below to run the code.
1. Open a new Anaconda Prompt.
   
2. Activate the created environment where all dependent Python libraries are installed by entering the following command:
   ```bash
   conda activate inductivebias
   ```
   Replace `inductivebias` with the actual name of your environment if it's different.
   
3. Once the environment is activated, start Jupyter Notebook by entering the following command:
   ```bash
   jupyter notebook
   ```
   This will open a new window in your default browser. Navigate to the project folder that you downloaded earlier.

## Analysis code

### Code

All analysis results presented in the paper can be exactly reproduced by running the analysis code in the `analysis` folder. This folder contains 9 notebook files with the extension `.ipynb` for executing all analyses and creating the figures presented in the paper. Each notebook's name indicates which figure it contributes to. For example, `fig1&3_trained.ipynb` created all analysis panels in Figs. 1 and 3 in the paper. Additionally, there is a Python file called `my_utils.py`, which contains customized utility functions used in the notebooks.

To reproduce all analysis figure panels in the paper, open an analysis notebook and run it sequentially for each cell. Within each notebook, we have denoted which specific panel in the paper each generated figure corresponds to.

The flow of each notebook is typically as follows: 
- Monkey data. Load monkey data if the corresponding paper figure uses monkey data. 
- Agent data.
   - For each agent, inspecting the performance of all checkpoints on the task that will be analyzed, and load the checkpoint where the agent has the highest performance (see Agent Selection section in the paper Methods and below).
   -  Run the agent at the selected checkpoint to generate agent data.
- Analysis and plotting for each figure panel in the paper.

### Data

Monkey data are stored in the `data/monkey_data`. All training checkpoints for agents are stored in the `data/agents` and `data/agents_all`, where the former contains agents that underwent default training ($10^4$ trials after training phase I, see the Agent Training section in the paper's Methods), while the latter contains agents that underwent extensive training ($10^5$ trials after training phase I). Performance evaluation data for all checkpoints of the agents are stored in the `data/training_curve`.

Each notebook loads data from these folders. Depending on the task being evaluated, each agent is endowed with the neural parameters at the checkpoint that allow it to achieve the highest reward rate in this task, according to the performance evaluation data. Since the size of agents' all checkpoints is too huge ($\sim 273$ G), the data folder downloaded by users only contain checkpoints that are chosen by analysis notebooks ($\sim 3$ G). 

Analysis notebooks for Fig. 2 (`fig2_belief.ipynb`) and Fig. 7 (`fig7_1_belief.ipynb`) involve exhaustively running agents over various uncertainty conditions, a process that can take several hours. Consequently, I have stored the results of these runs in the `data/analysis_data`. Both notebooks are designed to load these stored results for further analysis and plotting. Within these notebooks, cells responsible for generating data in the `analysis_data` folder are commented out in blocks, accompanied by notes stating 
> Running the following cells takes a significant amount of time, so I've saved the results. We can now simply load those results for analysis.

Users still have the option to execute all these cells, and doing so will generate the exact same data as stored in `analysis_data`.

## Agent training code

### Notebook files

The `model` folder contains notebooks for training agents. We have three types of agents used in the paper: 1. The EKF agent using an Extended Kalman Filter (EKF) to construct belief (Fig. S2**a**), 2. The holistic-EKF agent using a holistic critic and an EKF actor (Fig. S3**c**), and 3. all other agents using RNNs for beliefs (Fig. 8**a**–**b**). Training each type of agent requires running the following notebooks: `training_EKF_TD3.ipynb`, `training_RNNEKF_TD3.ipynb`, and `training_RNN_TD3.ipynb`, respectively. During training, we periodically save checkpoints for each agent. After training, we execute `validating.ipynb` to evaluate performance of all saved checkpoints.

### Python files

There are other Python files (.py) in the `model` folder that are imported by notebooks. Specifically, `Actor1.py`, `Actor2.py` and `Actor3.py` define the corresponding actor architectures in Fig. 8**b**. The actor using the EKF for belief is defined in `ActorEKF.py`, as illustrated in Fig. S3**c**. On the other hand, `Critic1.py`, `Critic2.py`, `Critic3.py`, `Critic4.py`, and `Critic5.py` define the corresponding critic architectures in Fig. 8**a**. The three types of RL agents—EKF agent, holistic-EKF agent, and other agents using RNNs—are defined in `Agent_EKF.py`, `Agent_RNN_EKF.py`, and `Agent_RNN.py`, respectively. The task environment is defined in `Environment.py`. During training, we periodically evaluate the agents' performance to decide the training phase and anneal some training parameters (see Agent Training section in paper's Methods), a process that requires `validation.py`.

### Training parameters

There is a subfolder `model/config`, containing a file named `config.py`. This file defines default values for many hyperparameters. Please note that all `.py` files are not directly modified; instead, users can modify hyperparameters for each agent training in the `.ipynb` notebooks `training_EKF_TD3.ipynb`, `training_RNNEKF_TD3.ipynb`, and `training_RNN_TD3.ipynb`. Specifically, in the third cell of each notebook (after the subtitle **`specify parameters`**), users can modify training parameters. Each parameter is accompanied by comments explaining its meaning and possible arguments.

For `training_RNN_TD3.ipynb`, users can decide actor and critic architectures through the variables `actors` and `critics`. Possible arguments are commented in the notebook. Users can choose three actor architectures by setting `actors = ['Actor1']` or `['Actor2']` or `['Actor3']`. Users can choose five critic architectures by setting `critics = ['Critic1']` or `['Critic2']` or `['Critic3']` or `['Critic4']` or `['Critic5']`. Users can also choose training tasks: one task is the $1\times$ gain task with the default process and observation noise uncertainty by setting the variable `task=['gain']`. The other is the $1\times$ gain task with other uncertainty conditions by setting the variable `task=['gain_control']` and providing values for variables `pro_noise` (process noise SD) and `obs_noise` (observation noise SD). Users can also specify other parameters such as random seeds (`seeds`), total training trials (`TOTAL_EPISODE`), and the root folder path for saving agents' checkpoints (`folder_path`).

In `training_RNNEKF_TD3.ipynb`, the only possible actor architecture is `actorEKF`, therefore, it should always be `actors = ['ActorEKF']`. In the paper, we only used `Critic1` to develop the holistic-EKF agent, so we set `critics = ['Critic1']`. However, other critic architectures are also possible to use. 

In `training_EKF_TD3.ipynb`, users do not specify architectures through variables since there is only one architecture.

### Checkpoints saving

During training, agents' checkpoints are automatically saved in the root folder path, with the default location being `data/agents_temp`. For agents trained in the $1\times$ gain task with the default uncertainty, the path for the checkpoints is `data/agents_temp/<architecture>/gain/<seed_number>`. e.g., `data/agents_temp/Actor1Critic1/gain/seed0`. For agents trained in the $1\times$ gain task with other uncertainty conditions, the path for the checkpoints is `data/agents_temp/<architecture>/gain_control/<process_noise_SD>_<observation_noise_SD>/<seed_number>`. e.g., `data/agents_temp/Actor1Critic1/gain_control/0.4_0.1/seed2`. The saved checkpoints in each path have an extension of `.pth.tar`, along with a `.csv` file recording the performance evaluation during training to decide the training phase and a `.pkl` file storing hyperparameters for this training run. All these files in the same path have the same name with a format of `%Y%m%d-%H%M%S`. Users should make sure each path has only one training run, i.e., there cannot be more than one `.csv` or `.pkl` file. 

Users can also refer to `data/agents` and `data/agents_all` to understand the folder structure of agents’ stored checkpoints.

### Checkpoints evaluation

After training, `validating.ipynb` should be run. In this notebook, in the second cell, users can define parameters for evaluating all checkpoints. The variable `training_task` should be set to `'gain'` if the agent being evaluated is trained with the $1\times$ gain task and default uncertainties. If the agent is trained with the $1\times$ gain task and other uncertainties, set `training task` to `'gain_control'` and specify `pro_noise` (process noise SD) and `obs_noise` (observation noise SD) used in training. If `training task='gain'`, evaluate this agent on the gain task with higher gains or on the perturbation task by setting `testing task='gain'` or `testing task='perturb'`. If `training task='gain_control'`, evaluate this agent on the gain task with higher gains or on the perturbation task by setting `testing task='gain_control'` or `testing task='perturb_control'`. If the agent is trained with `extensive=True` and `TOTAL_EPISODE = 1e5` in the training notebooks, here also set `extensive=True`, otherwise set `extensive=False` (see Agent Selection section in paper Methods). 

Users should also specify the root folder for checkpoints, agent architecture and random seed to locate the checkpoints for the specific agent being evaluated. For example, by letting `progress_agents_path = Path(f'../data/agents_temp/')`, `agent_archs = ['Actor3Critic5',]`, `agent_seeds = [(1,),]`, and `training task='gain'`, the code will locate the agent's checkpoints being evaluated at `data/agents_temp/Actor3Critic5/gain/seed1`.

The variable `save_path` denotes the root path for storing the performance data after evaluating all checkpoints for this agent. By default, it is `data/training_curve_temp`. For agents with `testing task='gain'` or with `testing task='perturb'`, the path for saving the performance data is `data/training_curve_temp/gain` or `data/training_curve_temp/perturbation`. For agents with `testing task='gain_control'` or with `testing task='perturb_control'`, the path for saving the performance data is `data/training_curve_temp/gain_control/<process_noise_SD>_<observation_noise_SD>` or `data/training_curve_temp/perturbation_control/<process_noise_SD>_<observation_noise_SD>`. 

The performance data file has a name format `<architecture>_<seed_number>.csv`, e.g., `Actor1Critic1_seed0.csv`. Users can also refer to `data/training_curve` to understand the folder structure of the performance data for agents’ stored checkpoints.

## Analysis code with newly trained agents

The analysis notebooks in the `analysis` folder require both trained agents' checkpoints and the performance evaluation data for these checkpoints. These analysis notebooks, by default in the third cell, load downloaded checkpoints and performance evaluation data from `data/agents`/`data/agents_all` and `data/training_curve`. If users want to run analysis notebooks with their newly trained agents, in the third cell, they should change paths to `data/agents_temp` and `data/training_curve_temp` to locate newly trained agents' checkpoints and the performance data evaluating these checkpoints.

## Reproducibility

It is known for PyTorch that 
> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds (https://pytorch.org/docs/stable/notes/randomness.html#reproducibility).

In all code implementations, we consistently set random seeds to ensure reproducibility in the presence of randomness. By testing on two Windows machines, each equipped with a CUDA GPU, we found that using the same random seeds leads to exactly the same outcomes for model evaluation. Therefore, all results in the analysis notebooks that only involve model evaluation can be exactly reproduced. However, the outcomes across different machines differ for model training when requiring model update on GPUs. Therefore, training with the same seeds across machines does not result in the exact learned neural weights (see this [discussion](https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047) and this [discussion](https://github.com/pytorch/pytorch/issues/38219)). Nevertheless, the conclusions of the results still hold statistically when averaging across enough random seeds.

