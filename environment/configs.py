from environment.allelopathic_harvest_open.maps import ALLELOPATHIC_HARVEST_OPEN_MAPS
from environment.king_of_the_hill.config import KING_OF_THE_HILL_DEFAULT_CONFIG
from environment.king_of_the_hill.evaluation_metrics import KOTH_EVAL_METRICS_FUNCS
from environment.king_of_the_hill.maps import KING_OF_THE_HILL_MAPS
from environment.king_of_the_hill.training_metrics import MetricsCallback as KOTHMetricsCallback
from environment.predator_and_prey_open.maps import PREDATOR_AND_PREY_OPEN_MAPS

DEFAULT_ENV_SETTINGS = {
    'king_of_the_hill': KING_OF_THE_HILL_DEFAULT_CONFIG,
    'allelopathic_harvest_open': None,
    'predator_and_prey_open': None
}

ENV_ALIASES = {
    'koth': 'king_of_the_hill',
    'king_of_the_hill': 'koth',
    'aho': 'allelopathic_harvest_open',
    'allelopathic_harvest_open': 'aho',
    'papo': 'predator_and_prey_open',
    'predator_and_prey_open': 'papo'
}

ENV_EVAL_METRIC_FUNCS = {
    'king_of_the_hill': KOTH_EVAL_METRICS_FUNCS,
    'allelopathic_harvest_open': {'episode_step': lambda *args, **kwargs: {},
                                  'episode_end': lambda *args, **kwargs: {}},
    'predator_and_prey_open': {'episode_step': lambda *args, **kwargs: {},
                               'episode_end': lambda *args, **kwargs: {}}
}

ENV_TRAINING_METRICS_CALLBACKS = {
    'king_of_the_hill': KOTHMetricsCallback,
    'allelopathic_harvest_open': None,
    'predator_and_prey_open': None
}

PLAYABLE_MAPS = {
    'king_of_the_hill': KING_OF_THE_HILL_MAPS,
    'allelopathic_harvest_open': ALLELOPATHIC_HARVEST_OPEN_MAPS,
    'predator_and_prey_open': PREDATOR_AND_PREY_OPEN_MAPS,
}
