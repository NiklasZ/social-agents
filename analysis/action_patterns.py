import json
from typing import List
from matplotlib import pyplot as plt
from tqdm import tqdm
from king_of_the_hill.wandb_api import make_dir_if_not_exists

# Plots common action sequences that agents take.

eval_path = 'king_of_the_hill/evaluation/run-1663376683/checkpoint-1300-2022-11-08-14-34-10'
metrics_path = f'{eval_path}/metrics.json'

with open(metrics_path) as metrics_str:
    metrics = json.load(metrics_str)


def get_action_patterns(episodes: List[List[dict]]):
    player_ids = episodes[0][0].keys()
    # list of [sequence length] by {player: {action sequence:frequency}}. E.g:
    # 1st element contains frequency of all single actions
    # 2nd element contains frequency of all action pairs
    # 3rd element contains frequency of all action triples
    # etc.
    sequences = []
    seq_limit = 50

    episode_list = []  # [episodes] by [timesteps] by {player:action}

    for episode in episodes:
        actions_by_player = []
        for p in player_ids:
            actions_by_player.append([actions[p] for actions in episode])
        episode_list.append(actions_by_player)

    for seq_length in range(1, seq_limit + 1):
        print(f'\nLength {seq_length}...')
        seqs = {}  # dict of {(player_id,action_sequence):count}
        for episode in tqdm(episode_list):
            for id, action_trajectory in enumerate(episode):
                idx = 0
                while idx + seq_length <= len(action_trajectory):
                    key = (id, tuple(action_trajectory[idx:idx + seq_length]))
                    if key not in seqs:
                        seqs[key] = 0
                    seqs[key] += 1
                    idx += 1

        # If the action sequences are so rare as to be unique, it's not worthwhile to look at longer sequences.
        most_frequent = max(seqs.values())

        print(f'\nMost frequent sequence occurred {most_frequent} times ({most_frequent / len(episodes)} per episode)')
        # Drop all sequences that only occur once.
        # And also drop all sequences that don't occur often enough per episode.
        min_frequency_per_ep = 0.1
        reduced = {k: v for k, v in seqs.items() if
                   v != 1 and v / len(
                       episodes) > min_frequency_per_ep}

        if len(reduced) == 0:
            print('Skipping further lengths as no more sequences are frequent enough')
            break

        print(
            f'Dropped {(len(seqs) - len(reduced)) / len(seqs) * 100:.4f}% of sequences for '
            f'as they only occurred once or <= {min_frequency_per_ep} times per episode')
        sequences.append(reduced)

    action_key = (
        'N',  # NOOP
        'F',  # FORWARD
        'B',  # BACKWARD
        'SL',  # STEP LEFT
        'SR',  # STEP RIGHT
        'TL',  # TURN LEFT
        'TR',  # TURN RIGHT
        'Z',  # FIRE ZAP (paint)
    )

    def to_action_str(action_seq):
        return tuple(action_key[i] for i in action_seq)

    most_frequent_actions = {id: {} for id in player_ids}  # by player, by length, by frequency

    print('Finding most frequent actions...')
    for freqs in tqdm(sequences):
        for idx, player in enumerate(most_frequent_actions):
            player_actions_freqs = [(to_action_str(f[1]), round(v / len(episodes), 2)) for f, v in freqs.items() if
                                    f[0] == idx]
            if len(player_actions_freqs):
                seq_len = len(player_actions_freqs[0][0])
                most_frequent_actions[player][seq_len] = sorted(player_actions_freqs, key=lambda t: t[1], reverse=True)[
                                                         :8]
    with open(f'{eval_path}/action_frequencies.json', 'w') as convert_file:
        convert_file.write(json.dumps(most_frequent_actions))

    print('\nMost frequent actions per player:')
    for p in player_ids:
        print(f'\n=={p}==')
        for seq_len in most_frequent_actions[p]:
            print(f'Seq length {seq_len}')
            print(most_frequent_actions[p][seq_len])

    print('Generating plots...')
    colours = ['red', 'blue', -1]
    max_freq = 0
    chart_path = f'{eval_path}/charts/action_frequencies'
    make_dir_if_not_exists(chart_path)

    for p in player_ids:
        max_freq = max([t[1] for t in most_frequent_actions[p][1]] + [max_freq])

    for p, col in zip(player_ids, colours):
        for seq_len in tqdm(most_frequent_actions[p]):
            number_of_seqs = len(most_frequent_actions[p][seq_len])
            seq_strs = ['-'.join(t[0]) for t in most_frequent_actions[p][seq_len]]
            seq_freqs = [t[1] for t in most_frequent_actions[p][seq_len]]
            fig = plt.figure()
            plt.bar(seq_strs, seq_freqs, color=col)

            plt.xlabel("Sequences")
            plt.ylabel("Frequency Per Episode")
            plt.title(f'{number_of_seqs} Most Frequent Action Sequences of Length {seq_len} for {p}')
            fig.subplots_adjust(bottom=0.5)
            plt.xticks(rotation=90)
            ax = plt.gca()
            ax.set_ylim([0, max_freq])
            # plt.show()
            plt.savefig(f'{chart_path}/{p}-len-{seq_len}.png')
            plt.clf()


get_action_patterns(metrics['actions'])
