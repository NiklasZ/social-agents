from src.environment.allelopathic_harvest_open.maps import ALLELOPATHIC_HARVEST_OPEN_MAPS
from src.environment.king_of_the_hill.config import KING_OF_THE_HILL_DEFAULT_CONFIG
from src.environment.king_of_the_hill.evaluation_metrics import KOTH_EVAL_METRICS_FUNCS
from src.environment.king_of_the_hill.maps import KING_OF_THE_HILL_MAPS
from src.environment.king_of_the_hill.training_metrics import MetricsCallback as KOTHMetricsCallback
from src.environment.predator_and_prey_open.maps import PREDATOR_AND_PREY_OPEN_MAPS
from src.deepmind.meltingpot.configs.substrates.paintball__king_of_the_hill import get_config as get_config_koth
from src.deepmind.meltingpot.configs.substrates.allelopathic_harvest__open import get_config as get_config_allelopathic_open
from src.deepmind.meltingpot.configs.substrates.predator_prey__open import get_config as get_config_predator_prey_open

# Contains custom arguments to feed to the environment constructor
DEFAULT_ENV_SETTINGS = {
    'king_of_the_hill': KING_OF_THE_HILL_DEFAULT_CONFIG,
    'allelopathic_harvest_open': None,
    'predator_and_prey_open': None
}

# Aliases for lazy command typing
ENV_ALIASES = {
    'koth': 'king_of_the_hill',
    'king_of_the_hill': 'koth',
    'aho': 'allelopathic_harvest_open',
    'allelopathic_harvest_open': 'aho',
    'papo': 'predator_and_prey_open',
    'predator_and_prey_open': 'papo'
}

# For calculating evaluation metrics that are specific to an environment.
ENV_EVAL_METRIC_FUNCS = {
    'king_of_the_hill': KOTH_EVAL_METRICS_FUNCS,
    'allelopathic_harvest_open': {'episode_step': lambda *args, **kwargs: {},
                                  'episode_end': lambda *args, **kwargs: {}},
    'predator_and_prey_open': {'episode_step': lambda *args, **kwargs: {},
                               'episode_end': lambda *args, **kwargs: {}}
}

# For calculating training metrics that are specific to an environment.
ENV_TRAINING_METRICS_CALLBACKS = {
    'king_of_the_hill': KOTHMetricsCallback,
    'allelopathic_harvest_open': None,
    'predator_and_prey_open': None
}

# Maps created for each environment.
PLAYABLE_MAPS = {
    'king_of_the_hill': KING_OF_THE_HILL_MAPS,
    'allelopathic_harvest_open': ALLELOPATHIC_HARVEST_OPEN_MAPS,
    'predator_and_prey_open': PREDATOR_AND_PREY_OPEN_MAPS,
}

# Melting pot's own environment config generation functions.
MELTING_POT_CONFIG_FUNCS = {
    'king_of_the_hill': get_config_koth,
    'allelopathic_harvest_open': get_config_allelopathic_open,
    'predator_and_prey_open': get_config_predator_prey_open
}
