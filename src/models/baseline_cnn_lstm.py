from typing import Tuple, List, Dict
import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import deprecation_warning
from ray.rllib.utils.annotations import override
import torch
from ray.rllib.utils.torch_utils import one_hot
from ray.util import log_once
from torch import nn, TensorType

"""
This is a baseline implementation of a CNN-LSTM model for the melting pot environment.
It is also possible to skip this implementation by using rllib's config, but this gives a bit more control.
"""


class BaselineCNNLSTM(RecurrentNetwork, torch.nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name
    ):
        torch.nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.input_shape = np.flip(
            obs_space.shape)  # Note: melting pot provides the observation as (H, W, C) but torch expects (C, H, W)
        self.lstm_state_size = 256
        self.cnn_output_size = 64
        self.lstm_input_size = self.cnn_output_size + self.action_space.n

        # Set to give model access to previous action
        self.view_requirements['model_prev_action'] = ViewRequirement(
            data_col=SampleBatch.ACTIONS, space=self.action_space, shift=-1
        )
        cnn = nn.Sequential()
        # First CNN layer
        cnn.add_module('conv1', nn.Conv2d(3, 16, (8, 8), stride=8))
        cnn.add_module('conv1_act', nn.ReLU())
        # Second CNN layer
        cnn.add_module('conv2', nn.Conv2d(16, 128, (11, 11), stride=1))
        cnn.add_module('conv2_act', nn.ReLU())
        cnn_out_shape = cnn(torch.zeros(*self.input_shape)).shape

        # Post CNN FC layers
        cnn.add_module('post_cnn_flatten', nn.Flatten())
        post_cnn_flatten_shape = torch.squeeze(torch.zeros(*cnn_out_shape)).shape[0]
        cnn.add_module('post_cnn_fc1', nn.Linear(post_cnn_flatten_shape, 64))
        cnn.add_module('post_cnn_fc1_act', nn.ReLU())
        cnn.add_module('post_cnn_fc2', nn.Linear(64, self.cnn_output_size))
        cnn.add_module('post_cnn_fc2_act', nn.ReLU())
        self.cnn = cnn

        # LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_state_size, batch_first=True)
        self.post_lstm_fc1 = nn.Linear(self.lstm_state_size, 256)
        self.post_lstm_fc1_act = nn.ReLU()

        # Action and value branches
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None


    @override(RecurrentNetwork)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """This is a one-to-one copy of the implementation from RecurrentNetwork, with the
        exception that we also get the previous action as an input."""
        # Creating a __init__ function that acts as a passthrough and adding the warning
        # there led to errors probably due to the multiple inheritance. We encountered
        # the same error if we add the Deprecated decorator. We therefore add the
        # deprecation warning here.
        if log_once("recurrent_network_tf"):
            deprecation_warning(
                old="ray.rllib.models.torch.recurrent_net.RecurrentNetwork"
            )
        flat_inputs = input_dict["obs_flat"].float()
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        # Add previous action as input
        prev_a = input_dict[SampleBatch.PREV_ACTIONS]
        prev_a = one_hot(prev_a.float(), self.action_space)

        output, new_state = self.forward_rnn(inputs, state, seq_lens, prev_action=prev_a)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(ModelV2)
    def get_initial_state(self) -> List[TensorType]:
        # Place hidden states on same device as model.
        h = torch.zeros(self.lstm_state_size)
        c = torch.zeros(self.lstm_state_size)
        return [h, c]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens, prev_action: TensorType, **kwargs) -> Tuple[
        TensorType, List[TensorType]]:
        """
        @param prev_action: one-hot tensor of last action
        @param inputs: tensor of shape (B x T x H x W x C), where:
         B is the batch dimension, T is the time dimension, H is the observation height, W is the observation width,
         and C is the number of channels
        @param state: a tuple of (B x h) where h is the hidden state size
        @param seq_lens:
        @return:
        """
        # The input is reshaped to condense the B and T dimensions into one and swap the C dimension, to make the torch
        # CNN happy -> (B*T x C x H x W)
        x = inputs.reshape(-1, *self.input_shape)
        out = self.cnn(x)
        # Concatenate on the previous action
        out = torch.concat([out, prev_action], dim=1)
        # However, the LSTM does care about the B and T dimensions, so now we need to restore them.
        out, [h, c] = self.lstm(out.reshape(*inputs.shape[:2], -1),
                                [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        self._features = self.post_lstm_fc1_act(self.post_lstm_fc1(out))
        action_out = self.action_branch(torch.squeeze(self._features))
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
