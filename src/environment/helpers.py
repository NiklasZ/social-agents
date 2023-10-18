from typing import List

from src.environment.configs import MELTING_POT_CONFIG_FUNCS


def validate_environment_roles(env_name: str, roles: List[str]):
    valid_roles = MELTING_POT_CONFIG_FUNCS[env_name]()['valid_roles']
    for role in roles:
        if role not in valid_roles:
            raise Exception(f"Invalid role '{role}' provided for environment {env_name}. Valid roles are {valid_roles}")
