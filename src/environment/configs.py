from environment.clean_up.maps import CLEANUP_MAPS
from environment.coins.config import COIN_DEFAULT_CONFIG
from environment.coins.maps import COINS_MAPS
from environment.territory_open.config import TERRITORY_OPEN_DEFAULT_CONFIG
from src.environment.coins.training_metrics import MetricsCallback as CoinsMetricsCallback
from environment.prisoners_dilemma_arena.maps import PRISONERS_DILEMMA_IN_THE_MATRIX_ARENA_MAPS
from environment.territory_rooms.maps import TERRITORY_ROOMS_MAPS
from src.environment.allelopathic_harvest_open.maps import ALLELOPATHIC_HARVEST_OPEN_MAPS
from src.environment.king_of_the_hill.config import KING_OF_THE_HILL_DEFAULT_CONFIG
from src.environment.king_of_the_hill.evaluation_metrics import KOTH_EVAL_METRICS_FUNCS
from src.environment.king_of_the_hill.maps import KING_OF_THE_HILL_MAPS
from src.environment.king_of_the_hill.training_metrics import MetricsCallback as KOTHMetricsCallback
from src.environment.predator_and_prey_open.maps import PREDATOR_AND_PREY_OPEN_MAPS
from src.environment.territory_open.maps import TERRITORY_OPEN_MAPS
from src.environment.territory_open.training_metrics import MetricsCallback as TerroMetricsCallback
from src.deepmind.meltingpot.configs.substrates.paintball__king_of_the_hill import get_config as get_config_koth
from src.deepmind.meltingpot.configs.substrates.allelopathic_harvest__open import \
    get_config as get_config_allelopathic_open
from src.deepmind.meltingpot.configs.substrates.predator_prey__open import get_config as get_config_predator_prey_open
from src.deepmind.meltingpot.configs.substrates.territory__open import get_config as get_config_territory_open
from src.deepmind.meltingpot.configs.substrates.prisoners_dilemma_in_the_matrix__arena import \
    get_config as get_config_prisoners_dilemma_in_the_matrix_arena
from src.deepmind.meltingpot.configs.substrates.clean_up import \
    get_config as get_config_clean_up
from src.deepmind.meltingpot.configs.substrates.territory__rooms import \
    get_config as get_config_territory_rooms
from src.deepmind.meltingpot.configs.substrates.coins import get_config as get_config_coins

# Contains custom arguments to feed to the environment constructor
DEFAULT_ENV_SETTINGS = {
    'allelopathic_harvest_open': None,
    'cleanup': None,
    'coins': COIN_DEFAULT_CONFIG,
    'king_of_the_hill': KING_OF_THE_HILL_DEFAULT_CONFIG,
    'predator_and_prey_open': None,
    'prisoners_dilemma_arena': None,
    'territory_open': TERRITORY_OPEN_DEFAULT_CONFIG,
    'territory_rooms': None,
}

# Aliases for lazy command typing
ENV_ALIASES = {
    'aho': 'allelopathic_harvest_open',
    'allelopathic_harvest_open': 'aho',
    'clean': 'cleanup',
    'cleanup': 'clean',
    'coins': 'coins',
    'king_of_the_hill': 'koth',
    'koth': 'king_of_the_hill',
    'papo': 'predator_and_prey_open',
    'predator_and_prey_open': 'papo',
    'prisa': 'prisoners_dilemma_arena',
    'prisoners_dilemma_arena': 'pris',
    'teroom': 'territory_rooms',
    'territory_open': 'terro',
    'territory_rooms': 'teroom',
    'terro': 'territory_open',
}

# For calculating evaluation metrics that are specific to an environment.
stub_dict = {'episode_step': lambda *args, **kwargs: {}, 'episode_end': lambda *args, **kwargs: {}}
ENV_EVAL_METRIC_FUNCS = {
    'allelopathic_harvest_open': stub_dict,
    'cleanup': stub_dict,
    'coins': stub_dict,
    'king_of_the_hill': KOTH_EVAL_METRICS_FUNCS,
    'predator_and_prey_open': stub_dict,
    'prisoners_dilemma_arena': stub_dict,
    'territory_open': stub_dict,
    'territory_rooms': stub_dict,
}

# For calculating training metrics that are specific to an environment.
ENV_TRAINING_METRICS_CALLBACKS = {
    'allelopathic_harvest_open': None,
    'cleanup': None,
    'coins': CoinsMetricsCallback,
    'king_of_the_hill': KOTHMetricsCallback,
    'predator_and_prey_open': None,
    'prisoners_dilemma_arena': None,
    'territory_open': TerroMetricsCallback,
    'territory_rooms': None,
}

# Maps created for each environment.
PLAYABLE_MAPS = {
    'allelopathic_harvest_open': ALLELOPATHIC_HARVEST_OPEN_MAPS,
    'cleanup': CLEANUP_MAPS,
    'coins': COINS_MAPS,
    'king_of_the_hill': KING_OF_THE_HILL_MAPS,
    'predator_and_prey_open': PREDATOR_AND_PREY_OPEN_MAPS,
    'prisoners_dilemma_arena': PRISONERS_DILEMMA_IN_THE_MATRIX_ARENA_MAPS,
    'territory_open': TERRITORY_OPEN_MAPS,
    'territory_rooms': TERRITORY_ROOMS_MAPS,
}

# Melting pot's own environment config generation functions.
MELTING_POT_CONFIG_FUNCS = {
    'allelopathic_harvest_open': get_config_allelopathic_open,
    'cleanup': get_config_clean_up,
    'coins': get_config_coins,
    'king_of_the_hill': get_config_koth,
    'predator_and_prey_open': get_config_predator_prey_open,
    'prisoners_dilemma_arena': get_config_prisoners_dilemma_in_the_matrix_arena,
    'territory_open': get_config_territory_open,
    'territory_rooms': get_config_territory_rooms,
}
