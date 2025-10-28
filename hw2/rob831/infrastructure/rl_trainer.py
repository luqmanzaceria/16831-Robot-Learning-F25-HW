from collections import OrderedDict
import pickle
import os
import sys
import time
from multiprocessing import Pool, cpu_count
import copy

try:
    import gymnasium as gym
    from gymnasium import wrappers
    USING_GYMNASIUM = True
except ImportError:
    import gym
    from gym import wrappers
    USING_GYMNASIUM = False
import numpy as np
import torch
from rob831.infrastructure import pytorch_util as ptu

from rob831.infrastructure import utils
from rob831.infrastructure.logger import Logger
from rob831.infrastructure.action_noise_wrapper import ActionNoiseWrapper

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

# Global variable for worker initialization
_worker_env = None
_worker_policy = None


def _init_worker(env_name, seed_base, action_noise_std, policy_state):
    """Initialize worker process with environment and policy"""
    global _worker_env, _worker_policy

    # Create environment
    _worker_env = gym.make(env_name)
    if USING_GYMNASIUM:
        _worker_env.reset(seed=seed_base)
    else:
        _worker_env.seed(seed_base)

    # Add noise wrapper if needed
    if action_noise_std > 0:
        from rob831.infrastructure.action_noise_wrapper import ActionNoiseWrapper
        _worker_env = ActionNoiseWrapper(_worker_env, seed_base, action_noise_std)

    # Recreate policy from state dict
    # We'll pass the policy class and parameters to reconstruct it
    _worker_policy = policy_state


def _worker_sample_trajectory(args):
    """Worker function that uses pre-initialized environment"""
    global _worker_env, _worker_policy
    seed_offset, max_path_length = args

    # Reset with new seed for this trajectory
    if USING_GYMNASIUM:
        _worker_env.reset(seed=_worker_env.unwrapped.spec.kwargs.get('seed', 0) + seed_offset)

    # Sample trajectory
    return utils.sample_trajectory(_worker_env, _worker_policy, max_path_length)


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        if USING_GYMNASIUM:
            self.env.reset(seed=seed)
        else:
            self.env.seed(seed)

        # Add noise wrapper
        if params['action_noise_std'] > 0:
            self.env = ActionNoiseWrapper(self.env, seed, params['action_noise_std'])

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name']=='obstacles-rob831-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10

        # Number of parallel workers for trajectory collection (0 = disable parallelization)
        self.params['num_workers'] = self.params.get('num_workers', 0)
        self._worker_pool = None  # Persistent worker pool
        
        # Initialize logging flags
        self.log_video = False
        self.log_metrics = False

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.log_metrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(itr,
                                initial_expertdata, collect_policy,
                                self.params['batch_size'])
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            train_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, load_initial_expertdata, collect_policy, batch_size):
        if itr == 0:
            if load_initial_expertdata:
                paths = pickle.load(open(self.params['expert_data'], 'rb'))
                return paths, 0, None
            else:
                num_transitions_to_sample = self.params['batch_size_initial']
        else:
            num_transitions_to_sample = self.params['batch_size']

        print("\nCollecting data to be used for training...")
        if self.params['num_workers'] > 0:
            paths, envsteps_this_batch = self._sample_trajectories_parallel(
                collect_policy, num_transitions_to_sample, self.params['ep_len'])
        else:
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'])

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def _sample_trajectories_parallel(self, policy, min_timesteps_per_batch, max_path_length):
        """Parallel version of sample_trajectories using multiprocessing"""
        num_workers = self.params['num_workers']

        paths = []
        timesteps_this_batch = 0

        # Create pool once and reuse (only create if not exists or policy changed)
        if self._worker_pool is None:
            self._worker_pool = Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(self.params['env_name'], self.params['seed'],
                         self.params['action_noise_std'], policy)
            )

        # Estimate how many trajectories we need
        estimated_traj_needed = max(num_workers, int(min_timesteps_per_batch / (max_path_length / 2)))

        traj_idx = 0
        while timesteps_this_batch < min_timesteps_per_batch:
            # Determine batch size
            batch_size = min(num_workers * 2, estimated_traj_needed)

            # Prepare args (just seed offset and max_path_length)
            args_list = [(traj_idx + i, max_path_length) for i in range(batch_size)]

            # Collect trajectories in parallel
            batch_paths = self._worker_pool.map(_worker_sample_trajectory, args_list)

            # Add paths and count timesteps
            for path in batch_paths:
                paths.append(path)
                timesteps_this_batch += utils.get_pathlength(path)
                print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')

                if timesteps_this_batch >= min_timesteps_per_batch:
                    break

            traj_idx += batch_size

        return paths, timesteps_this_batch

    def __del__(self):
        """Clean up worker pool when trainer is destroyed"""
        if hasattr(self, '_worker_pool') and self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool.join()

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
