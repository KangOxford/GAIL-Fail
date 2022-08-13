import argparse
import numpy as np
import random
import matplotlib
import seaborn as sns; sns.set()
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt

import baselines.common.plot_util as pu

# plt.rcParams['svg.fonttype'] = 'svgfont'
# plt.rcParams.update({'font.size': 22})
font_size = 14

matplotlib.rc('xtick', labelsize=font_size) 
matplotlib.rc('ytick', labelsize=font_size) 

def argsparser():
    parser = argparse.ArgumentParser("Plotting lines")
    parser.add_argument('--env_id', help='env', default='Hopper')
    parser.add_argument('--num_timesteps', help='timesteps to draw', type=int, default=int(2e6))
    parser.add_argument('--legend', help='legend', type=int, default=1)
    return parser.parse_args()

def main(args):
    env_name = args.env_id.split("-")[0]
    # dir_path = '../training/0107_baselines_gail_new_dataset/%s'%args.env_id
    # dir_path = '../training/0106_baselines_gail/%s'%args.env_id
    # dir_path = '../training/0108_baselines_trpo/%s'%args.env_id
    # dir_path = '../training/0109_deterministic_imitation/%s'%args.env_id
    # dir_path = '../training/0109_deterministic_imitation_monitor/%s'%args.env_id
    # dir_path = '../training/0110_deterministic_adaptive_sampling/%s'%args.env_id
    # dir_path = '../training/0111_d2_sampling/misc/%s'%args.env_id
    # dir_path = '../training/0111_d2_reward_negative/%s'%args.env_id
    # dir_path = '../training/0111_d2_prob/%s'%args.env_id
    # dir_path = '../plotting/d2_gail/%s'%args.env_id

    # log_path = '../training/0113_dac_incomplete_result'
    # log_path = '../training/0107_baselines_bc_new_dataset'
    # log_path = '../training/0113_d2_stable'
    # log_path = 'log'
    # log_path = '../training/0118_d2_mixed_sampling'
    # log_path = '../training/0113_dac_incomplete_result'
    # log_path = '../training/0118_d2_mixed_sampling'
    # log_path = '../training/0120_d2_complete'
    # log_path = '../training/0120_dac_baselines_ppo'
    # log_path = '../training/0122_d2_ppo_data'
    # log_path = '../training/0122_d2_sac_data_pure'
    # log_path = '../training/0127_d2_ppo_full_results'
    # log_path = '../training/0126_ablations_no_discriminator'

    # log_path = '../training/0125_dac_sac_data_pure'
    log_path = '../training/0122_d2_sac_data_pure'

    # select_envs = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    # select_keys = [
    #     'td3-20.txt',   'td3-15.txt',   'td3-10.txt',   'td3-5.txt',    'td3-1.txt', 
    #     'ddpg-20.txt',  'ddpg-15.txt',  'ddpg-10.txt',  'ddpg-5.txt',   'ddpg-1.txt', 
    #     'ppo-20.txt',   'ppo-15.txt',   'ppo-10.txt',   'ppo-5.txt',    'ppo-1.txt', 
    #     'sac-20.txt',   'sac-15.txt',   'sac-10.txt',   'sac-5.txt',    'sac-1.txt' ]

    # select_envs = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    # select_keys = ['ppo-20', 'ppo-15', 'ppo-10', 'ppo-5']

    # select_envs = ['Ant-v2']
    # select_keys = ['ppo-20.txt.with_pretrained.BC_iter_1000', 
    #                'ppo-20.txt.with_pretrained.BC_iter_2000',
    #                'ppo-20.txt.with_pretrained.BC_iter_4000']

    select_envs = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    select_keys = ['sac-20.txt', 'sac-15.txt', 'sac-10.txt', 'sac-05.txt']
    # select_keys = ['sac-20.txt', 'sac-15.txt', 'sac-10.txt']
    # select_keys = ['ppo-20.txt', 'ppo-15.txt', 'ppo-10.txt', 'ppo-5.txt', 'ppo-1.txt']

    mode = 'training'
    # mode = 'evaluation'

    if mode == 'evaluation':
        xy_fn = pu.evaluate_mean_returns
    else:
        xy_fn = pu.default_xy_fn

    for env in select_envs:
        dir_path = '%s/%s'%(log_path, env)
        results = pu.load_results(dir_path, enable_progress=True, enable_monitor=True)

        for key in select_keys:
            print('********************')
            print('Key: %s'%key)
            tmp_results = []
            for result in results:
                if key in result.dirname:
                    tmp_results.append(result)

            pu.plot_results(tmp_results, 
                            xy_fn=xy_fn,
                            # xy_fn=pu.length_fn,
                            # xy_fn=pu.bc_evaluation_xy_fn,
                            average_group=True, 
                            split_fn = lambda _ : '',
                            figsize=(8, 8),
                            # legend_outside=True,
                            smooth_step=3.0,
                            shaded_std=False)
            # plt.grid(which='major')
            plt.xlabel('Number of timesteps (%s)'%(env), fontsize=font_size)
            plt.ylabel('Returns', fontsize=font_size)
            # plt.ylabel('Episode length', fontsize=font_size)
            plt.tight_layout()
            params = {'legend.fontsize': font_size}
            plt.rcParams.update(params)
            fig_name = "%s/%s-%s-%s.pdf"%(log_path, mode, env, key)
            plt.savefig(fig_name)
            # plt.show()

if __name__ == '__main__':
    args = argsparser()
    main(args)
