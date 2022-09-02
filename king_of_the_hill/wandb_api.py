import os
import shutil
from typing import Dict, Optional, List
import tqdm
import wandb
from king_of_the_hill.generate_config import get_alg_config


def get_api_key() -> str:
    folder = 'king_of_the_hill'
    folder_idx = __file__.index(folder) + len(folder)
    folder_path = __file__[:folder_idx]
    with open(f'{folder_path}/wandb_api_key.txt', mode='r') as f:
        api_key = f.read()
    return api_key


def make_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_run_folder(all_runs_dir: str) -> str:
    folder_names = [d for d in os.listdir(all_runs_dir) if os.path.isdir(f'{all_runs_dir}/{d}')]
    assert len(
        folder_names) == 1, f'Expected to find exactly 1 run folder in {all_runs_dir} but got {len(folder_names)}'
    return folder_names[0]


def sync_from_wandb(rllib_path: str,
                    api_key: str,
                    run_config: Dict,
                    sync_checkpoints: Optional[List[str]] = None) -> Dict:
    api = wandb.Api(api_key=api_key)
    run_path = f"{run_config['entity_name']}/{run_config['wandb_project_name']}/{run_config['run_id']}"
    run = api.run(run_path)
    combined_config = run.config
    run_config = combined_config['run_config'] | run_config  # merge configs
    config = get_alg_config(run_config)

    all_runs_dir = f"{rllib_path}/{run_config['rllib_run_name']}"
    make_if_not_exists(all_runs_dir)

    print('Downloading run state file...')
    state_file_names = [file.name for file in run.files() if 'zip' in file.name and 'checkpoint' not in file.name]
    assert len(state_file_names) == 1, f'Expected to find exactly 1 state zip but got {len(state_file_names)}'

    state_file = state_file_names[0]
    wandb.restore(state_file, run_path=run_path, root=rllib_path)
    shutil.unpack_archive(f'{rllib_path}/{state_file}', extract_dir=all_runs_dir)

    run_folder = find_run_folder(all_runs_dir)
    run_dir = f'{all_runs_dir}/{run_folder}'

    print('Downloading checkpoint files...')

    # Note this will probably get really slow if there are many checkpoints to download
    checkpoint_filter = (lambda x: x in sync_checkpoints) if sync_checkpoints else True
    checkpoint_file_names = [file.name for file in run.files() if 'checkpoint' in file.name and checkpoint_filter(file)]
    for name in tqdm.tqdm(checkpoint_file_names):
        wandb.restore(name, run_path=run_path, root=run_dir)
        checkpoint_dir = f'{run_dir}/{os.path.splitext(name)[0]}'
        make_if_not_exists(checkpoint_dir)
        shutil.unpack_archive(f'{checkpoint_dir}.zip', extract_dir=checkpoint_dir)
        os.remove(f'{checkpoint_dir}.zip')

    return config
