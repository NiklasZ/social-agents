from typing import Callable, List
import numpy as np
from ray.rllib.algorithms.ppo import PPO
from king_of_the_hill.arg_parser import supported_activation_layers


class ActivationExtractor:
    def __init__(self):
        self.activations = {}
        self.hooks = []
        self.current_player = None

    def set_current_player(self, player: str):
        self.current_player = player

    def clear_activations(self):
        self.activations = {}

    def register_activation_hooks(self, agent: PPO, layer_aliases: List[str]):
        """
        Adds hooks to save activations from all neural network layers in this class.
        For more details on how this works, see:
        https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
        https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
        :return:
        """

        layer_names = [l.split('_')[0] for l in layer_aliases]
        layer_indices = [supported_activation_layers[l] for l in layer_aliases]

        for policy_name in agent.config['multiagent']['policies']:
            model = agent.get_policy(policy_name).model

            for name, index, alias in zip(layer_names, layer_indices, layer_aliases):

                def save_activations(index: int, alias: str) -> Callable:
                    def save(model, input, output):
                        # RNNs will output a tuple containing the same tensor twice, so we need to pick one.
                        if type(model).__name__ in ['RNN', 'LSTM'] and isinstance(index, int):
                            # Need to pick the hidden layer activation, rather than the cell state
                            # (LSTM case. With RNN it doesn't really matter)
                            activation = output[1][index].cpu().detach().numpy().squeeze()
                        else:
                            # self.activations[name].append(output.cpu().detach())
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
                # self.activations[player_name] = {}  # Create activations list
                # We access the internal model of the RLlib module here. Not great, but it's the only thing that works.
                # hook = layer._model.register_forward_hook(save_activations_for(name))
                hook = layer.register_forward_hook(save_activations(index, alias))
                self.hooks.append(hook)

    def deregister_activation_hooks(self):
        """
        Removes stored activation hooks
        :return:
        """
        for hook in self.hooks:
            hook.remove()
