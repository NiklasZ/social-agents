from typing import Callable, List, Tuple
import numpy as np

from ray.rllib.algorithms.ppo import PPO


class ActivationExtractor:
    def __init__(self):
        self.activations = {}
        self.hooks = []
        self.current_player = None

    def set_current_player(self, player: str):
        self.current_player = player

    def clear_activations(self):
        self.activations = {}

    def register_activation_hooks(self, agent: PPO, layer_name: str):
        """
        Adds hooks to save activations from all neural network layers in this class.
        For more details on how this works, see:
        https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
        https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
        :return:
        """
        for policy_name in agent.config['multiagent']['policies']:
            model = agent.get_policy(policy_name).model

            def save_activations() -> Callable:
                def save(model, input, output):
                    # RNNs will output a tuple containing the same tensor twice, so we need to pick one.
                    if type(model).__name__ in ['RNN', 'LSTM']:
                        # Need to pick the hidden layer activation, rather than the cell state
                        # (LSTM case. With RNN it doesn't really matter)
                        activation = output[0].cpu().detach().numpy().squeeze()
                    else:
                        # self.activations[name].append(output.cpu().detach())
                        raise NotImplementedError("Other activation layers not supported. Please implement")

                    concatable = np.expand_dims(activation, axis=0)

                    if self.current_player not in self.activations:
                        self.activations[self.current_player] = concatable  # Create activations list
                    else:
                        self.activations[self.current_player] = np.concatenate(
                            [self.activations[self.current_player], concatable], axis=0)
                return save

            layer = getattr(model, layer_name)
            # self.activations[player_name] = {}  # Create activations list
            # We access the internal model of the RLlib module here. Not great, but it's the only thing that works.
            # hook = layer._model.register_forward_hook(save_activations_for(name))
            hook = layer.register_forward_hook(save_activations())
            self.hooks.append(hook)

    def deregister_activation_hooks(self):
        """
        Removes stored activation hooks
        :return:
        """
        for hook in self.hooks:
            hook.remove()
