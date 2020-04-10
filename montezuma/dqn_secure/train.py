import os
import pickle
import click
import yaml
import numpy as np

import sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
OUTPUT_DIR = ROOT_DIR
sys.path.append(ROOT_DIR)

from lib.utils import Font
from dqn_secure.experiment import DQNExperiment
from lib.utils import ExperienceReplay
from environments.atari_env import AtariEnv

np.set_printoptions(suppress=True, linewidth=200, precision=2)
floatX = 'float32'


def run():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config_atari.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    print('\n')
    print(Font.bold + Font.red + 'Parameters ' + Font.end)
    for key in params:
        print(key, params[key])
    print('\n')

    # TODO: clean this for distribution, this is only a workaround for Colab, since it cannot take very large numpy array
    params['replay_max_size']=100000

    np.random.seed(seed=params['random_seed'])
    random_state = np.random.RandomState(params['random_seed'])

    if params['game_name'] == 'montezuma_revenge':
        ch_weights = [1, 0, 0]  # only using red
    elif params['game_name'] in ['frostbite', 'seaquest']:
        ch_weights = [0.5, 0.5, 0]
    elif params['game_name'] == 'asteroid':
        ch_weights = [0.3333, 0.3333, 0.3333]
    else:
        ch_weights = [0.5870, 0.2989, 0.1140]

    # TODO: maybe ch_weights helps
    env = AtariEnv(frame_skip=params['frame_skip'], repeat_action_probability=params['repeat_action_probability'],
                    game=params['game_name'])

    from dqn_secure.ai import AI  # we need to import after ALE due to tensorflow/cv2 conflicts
    rewards_list = []
    for ex in range(params['num_experiments']):
        print('\n')
        print(Font.bold + Font.red + '>>>>> Experiment ', ex, ' >>>>>' + Font.end)
        print('\n')
        replay_buffer_exploit = ExperienceReplay(max_size=params['replay_max_size'],
                                                 state_shape=env.observation_space.shape, action_dim=params['action_dim'],
                                                 reward_dim=params['reward_dim'])
        if params['use_exploit_btstrap_corr'] == True:
            btstrap_corr = [0, 100]
        else:
            btstrap_corr = []

        ai = AI(state_shape=env.observation_space.shape, nb_actions=env.action_space.n, action_dim=params['action_dim'],
                reward_dim=params['reward_dim'], no_network=False, history_len=params['history_len'],
                gamma=params['gamma'], learning_rate=params['learning_rate'],
                minibatch_size=params['minibatch_size'], update_freq=params['update_freq'],
                learning_frequency=params['learning_frequency'], ddqn=params['ddqn'], normalize=params['normalize'],
                replay_buffer=replay_buffer_exploit, bootstrap_corr=btstrap_corr, rng=random_state)

        if params['secure'] == True:
            # AI for explore (by definition): gamma=1.0 ; method=DDQN ; reward -1 for bad term, zero otherwise
            replay_buffer_explore = ExperienceReplay(max_size=params['replay_max_size'],
                                                     state_shape=env.observation_space.shape, action_dim=params['action_dim'],
                                                     reward_dim=params['reward_dim'])
            ai_explore = AI(state_shape=env.observation_space.shape, nb_actions=env.action_space.n, action_dim=params['action_dim'],
                            reward_dim=params['reward_dim'], no_network=False, history_len=params['history_len'],
                            gamma=1.0, learning_rate=params['learning_rate'], minibatch_size=params['minibatch_size'],
                            update_freq=params['update_freq'], learning_frequency=params['learning_frequency'],
                            ddqn=True, normalize=params['normalize'],
                            replay_buffer=replay_buffer_explore, bootstrap_corr=[-1, 0], rng=random_state)
        else:
            ai_explore = None
        if params['test']:  # note to pass correct folder name
            network_weights_dir = os.path.join(os.getcwd(), 'results', params['folder_name'])
            ai.load_weights(weights_file_path=network_weights_dir+'/q_network_weights.h5')
            if params['secure'] == True:
                ai_explore.load_weights(weights_file_path=network_weights_dir + '/q_explore_network_weights.h5')

        expt = DQNExperiment(env=env, ai=ai, ai_explore=ai_explore, episode_max_len=params['episode_max_len'],
                             history_len=params['history_len'], max_start_nullops=params['max_start_nullops'],
                             replay_min_size=params['replay_min_size'], epsilon=params['epsilon'],
                             annealing=params['annealing'], final_epsilon=params['final_epsilon'],
                             annealing_start=params['annealing_start'], annealing_steps=params['annealing_steps'],
                             folder_location=os.path.join(OUTPUT_DIR, params['folder_location']),
                             folder_name=params['folder_name'], use_expl_inc=params['use_expl_inc'],
                             testing=params['test'], score_window_size=100, rng=random_state,
                             q_threshold=params['q_threshold'], secure=params['secure'], max_secure=params['max_secure'],
                             ai_rewarding_buffer_size=params['exploit_rewarding_buffer_size'],
                             ai_explore_rewarding_buffer_size=params['explore_rewarding_buffer_size'],
                             exploration_learning_steps=params['exploration_learning_steps'])
        env.reset()
        if not params['test']:
            with open(expt.folder_name + '/config.yaml', 'w') as y:
                yaml.safe_dump(params, y)  # saving params for reference
            expt_folder_name = expt.folder_name
            rewards = expt.do_epochs(number=params['num_epochs'], is_learning=params['is_learning'],
                                     steps_per_epoch=params['steps_per_epoch'], is_testing=params['is_testing'],
                                     steps_per_test=params['steps_per_test'])
            rewards_list.append(rewards)
            with open(expt_folder_name + '/rewards_output.pkl', 'wb') as f:
                pickle.dump(rewards_list, f)
        else:
            if params['human']:
                expt.do_human_episode()
            else:
                expt.evaluate(number=5)


if __name__ == '__main__':
    run()
