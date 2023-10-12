from collections import defaultdict
from copy import deepcopy
from typing import Callable, List
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO

from evaluation.arg_parser import supported_activation_layers


class ActivationExtractor:
    def __init__(self):
        self.activations = {}
        self.hooks = []
        self.current_player = None

    def set_current_player(self, player: str):
        self.current_player = player

    def clear_activations(self):
        self.activations = {}

    def register_activation_hooks(self, agent: PPO, policy_name: str, layer_aliases: List[str]):
        """
        Adds hooks to save activations from all neural network layers in this class.
        For more details on how this works, see:
        https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
        https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
        :return:
        """

        layer_names = [supported_activation_layers[l][0] for l in layer_aliases]
        layer_indices = [supported_activation_layers[l][1] for l in layer_aliases]

        model = agent.get_policy(policy_name).model

        for name, index, alias in zip(layer_names, layer_indices, layer_aliases):
            def save_activations(index: int, alias: str) -> Callable:
                def save(model: torch.nn.Module, input, output):
                    # RNNs will output a tuple containing the same tensor twice, so we need to pick one.
                    if type(model).__name__ in ['RNN', 'LSTM'] and isinstance(index, int):
                        # Need to pick the hidden layer activation, rather than the cell state
                        # (LSTM case. With RNN it doesn't really matter)
                        activation = output[1][index].cpu().detach().numpy().squeeze()
                    elif type(model).__name__ in ['SlimFC']:
                        activation = output.cpu().detach().numpy().squeeze()
                    else:
                        raise NotImplementedError("Other activation layers not supported. Please implement")

                    concatable = np.expand_dims(activation, axis=0)

                    # Set up dict path
                    if self.current_player not in self.activations:
                        self.activations[self.current_player] = {}  # Create activations list
                    if alias not in self.activations[self.current_player]:
                        self.activations[self.current_player][alias] = concatable
                    else:
                        self.activations[self.current_player][alias] = np.concatenate(
                            [self.activations[self.current_player][alias], concatable], axis=0)

                return save

            layer = getattr(model, name)
            hook = layer.register_forward_hook(save_activations(index, alias))
            self.hooks.append(hook)

    def deregister_activation_hooks(self):
        """
        Removes stored activation hooks
        :return:
        """
        for hook in self.hooks:
            hook.remove()


def has_trainable_parameters(module: torch.nn.Module) -> bool:
    """
    Checks if a module has any trainable parameters
    :param module:
    :return:
    """
    return any(p.requires_grad for p in module.parameters())


def filter_children_by_trainable_parameters(module: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Filters a module's children by those with trainable parameters
    :param module:
    :return:
    """
    return [child for child in module.children() if has_trainable_parameters(child)]


class NeuronScorer:
    """
    This is an implementation of a neuron scorer with the purpose of finding "dormant" i.e near-zero activation neurons.
    The idea of neurons becoming dormant during training and lowering the capability to learn is from the paper:
    'The Dormant Neuron Phenomenon in Deep Reinforcement Learning' by Sokar et al. (2023)
    paper: https://arxiv.org/pdf/2302.12902.pdf, code: https://github.com/google/dopamine/tree/master/dopamine/labs/redo

    It should be noted that at the time of writing, the agents do not seem to exhibit this increase in dormant neurons.
    There could be many reasons for this, but we will monitor for increased dormancy in future experiments just in case.
    """

    def __init__(self, scoring_batch_size: int = 256):
        # Dict indexed as follows: {player_name: {layer_name: {layer_property: activations}}}
        # layer_property is needed as some trainable layers are have multiple trainable weights that need to be scored separately
        self.activations = {}
        # Dict indexed as follows: {player_name: {layer_name: {layer_property: activation scores}}}
        self.activation_scores = {}
        self.scoring_batch_size = scoring_batch_size
        self.hooks = []
        self.current_player = None

        # [child for child in model.children() if any(p.requires_grad for p in child.parameters())]

    def set_current_player(self, player: str):
        self.current_player = player

    def clear_activations(self):
        self.activations = {}

    def register_activation_hooks(self, module: torch.nn.Module, player: str, prefix=''):
        if player not in self.activations:
            self.activations[player] = {}

        # Iterate over all possible children
        for idx, child in enumerate(filter_children_by_trainable_parameters(module)):
            new_prefix = f'{prefix}.{idx}' if prefix else f'{idx}'
            layer_type_name = type(child).__name__
            known_layer_types = ['LSTM', 'SlimFC', 'SlimConv2d']
            # Layers that are composed of multiple trainable layers
            if layer_type_name == 'Sequential':
                self.register_activation_hooks(child, player, new_prefix)
            # Unknown trainable layer types
            elif layer_type_name not in known_layer_types:
                raise Exception(f'Unhandled activation registration for trainable layer type: {layer_type_name}')
            # Known layers composed of trainable parameters, and activation function and miscellaneous operations
            # We apply the activation hooks at this point as the original paper also measures for dormant neurons
            # After the activation function.
            else:
                layer_id = f'{prefix}_{str(child)}'
                self.activations[player][layer_id] = defaultdict(list)

                def store_activations(layer_alias: str) -> Callable:

                    def activation_hook(model: torch.nn.Module, input, output):
                        layer_type_name = type(model).__name__
                        # Note that these activations are always taken from a module's output, so the activation function
                        # and other operations are already applied.
                        if layer_type_name == 'LSTM':
                            self.activations[player][layer_alias]['hidden_state'].append(
                                output[1][0].cpu().detach().numpy().squeeze())
                            self.activations[player][layer_alias]['cell_state'].append(
                                output[1][1].cpu().detach().numpy().squeeze())
                        elif layer_type_name == 'SlimFC':
                            self.activations[player][layer_alias]['default'].append(
                                output.cpu().detach().numpy().squeeze())
                        elif layer_type_name == 'SlimConv2d':
                            # Note we transpose the activations to put the spatial dimension last, which is easier to work with when calculating scores.
                            output_transposed = output.permute(*list(reversed(range(output.ndim))))
                            self.activations[player][layer_alias]['default'].append(
                                output_transposed.cpu().detach().numpy().squeeze())
                        else:
                            raise Exception(f'Unsupported layer type: {layer_type_name}')

                        # Check if we have a sufficient batch of activations to score
                        should_batch = True
                        for p in self.activations:  # player
                            for l in self.activations[p]:  # layer
                                for t in self.activations[p][l]:  # layer type
                                    activations = self.activations[p][l][t]
                                    # Note that we need to check for layers with no activations recorded, as the value function
                                    # is not necessarily called during evaluation, which means the activation hook may
                                    # not be called for some layers.
                                    if len(activations) < self.scoring_batch_size and len(activations) != 0:
                                        should_batch = False
                        if should_batch:
                            self.calculate_scores()

                    return activation_hook

                # Make a copy of the activations dict to store the scores as they have an identical structure
                self.activation_scores = deepcopy(self.activations)
                hook = child.register_forward_hook(store_activations(layer_id))
                self.hooks.append(hook)

    def calculate_scores(self):
        """
        Calculates neuron scores for all stored activations
        @return:
        """
        for p in self.activations:  # player
            for l in self.activations[p]:  # layer
                for t in self.activations[p][l]:  # layer type
                    # Stack collected activations into a tensor get magnitude
                    activations = np.abs(np.stack(self.activations[p][l][t]))
                    # Take mean over all dimensions except the spatial dimension (defining the # of neurons) at the end.
                    dims = list(range(activations.ndim - 1))
                    scores = np.mean(activations, axis=tuple(dims))
                    # Normalise score by mean score across all neurons
                    self.activation_scores[p][l][t].append(scores / (np.mean(scores) + 1e-9))

                    # Reset activations for next scoring
                    self.activations[p][l][t] = []

    def deregister_activation_hooks(self):
        """
        Removes stored activation hooks
        :return:
        """
        for hook in self.hooks:
            hook.remove()
