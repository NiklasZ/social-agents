import json
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Creates plot of various events in an episode.

# TODO: write a proper argparse
eval_path = 'king_of_the_hill/evaluation/run-1677743405/checkpoint-range-100-1500-2023-03-04-16-08-31/checkpoint-1500'
policy_folder = 'policy0__VS__policy1'
metrics_path = f'{eval_path}/{policy_folder}/metrics.pkl'
checkpoint_no = 1500
ep_id = 0


def plot_single_step_events(m: dict):
    if len(m['zaps'][ep_id][0]) != 2:
        raise NotImplementedError('Function is only designed for 2 agents')

    player_0_zaps = [i for i, p in enumerate(m['zaps'][ep_id]) if len(p['player_0'])]  # Whenever player 0 zaps player 1
    player_1_zaps = [i for i, p in enumerate(m['zaps'][ep_id]) if len(p['player_1'])]  # Whenever player 0 zaps player 1
    player_0_paints = [i for i, p in enumerate(m['actions'][ep_id]) if p['player_0'] == 7]
    player_1_paints = [i for i, p in enumerate(m['actions'][ep_id]) if p['player_1'] == 7]
    player_0_sees_1 = [i for i, p in enumerate(m['visibilities'][ep_id]) if len(p['player_0'])]
    player_1_sees_0 = [i for i, p in enumerate(m['visibilities'][ep_id]) if len(p['player_1'])]
    player_0_dead = [i for i, p in enumerate(m['deaths'][ep_id]) if p['player_0']]
    player_1_dead = [i for i, p in enumerate(m['deaths'][ep_id]) if p['player_1']]
    player_0_captures = [i for i, p in enumerate(m['captures'][ep_id]) if p['player_0']]
    player_1_captures = [i for i, p in enumerate(m['captures'][ep_id]) if p['player_1']]

    data = [player_1_zaps, player_0_zaps, player_1_paints, player_0_paints, player_1_sees_0, player_0_sees_1,
            player_1_captures, player_0_captures, player_1_dead, player_0_dead]
    labels = ['blue player zaps red', 'red player zaps blue', 'blue paints', 'red paints', 'blue sees red',
              'red sees blue', 'blue captures hill', 'red captures hill', 'blue is knocked out', 'red is knocked out']

    fig = plt.figure(1)

    # set different colors for each set of positions
    colors1 = ['C{}'.format(i) for i in range(len(data))]

    # set different line properties for each set of positions
    # note that some overlap
    lineoffsets1 = np.array([0, -1, -4, -5, -8, -9, -12, -13, -15, -16])
    # linelengths1 = [5, 2, 1, 1, 3, 1.5]

    # create a horizontal plot
    ax = plt.eventplot(data, colors=colors1, lineoffsets=lineoffsets1)

    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('timesteps')
    plt.title(f'Single timestep events at checkpoint {checkpoint_no}')
    ax[0].axes.get_yaxis().set_visible(False)
    # plt.show()

    out_path = f'{eval_path}/ep{ep_id}_events'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Generated image to {out_path}')


if __name__ == "__main__":
    with open(metrics_path, 'rb') as metrics_str:
        metrics = pickle.load(metrics_str)
    plot_single_step_events(metrics)
