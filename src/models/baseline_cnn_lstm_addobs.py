from typing import Tuple, List, Dict
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
import numpy as np
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
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.original_space = getattr(obs_space, "original_space", obs_space)
        # rgb = self.original_space['RGB']

        # self.flattened_input_space = flatten_space(self.original_space)
        # Note: melting pot provides the observation as (H, W, C) but torch expects (C, H, W)
        self.rgb_shape = self.original_space['RGB'].shape
        self.rgb_shape = [self.rgb_shape[2], self.rgb_shape[0], self.rgb_shape[1]]
        self.lstm_state_size = 256
        self.cnn_output_size = 64
        self.lstm_input_size = self.cnn_output_size + self.action_space.n
        for name in self.original_space:
            if name != 'RGB':
                self.lstm_input_size += int(np.prod(self.original_space[name].shape))

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
        cnn_out_shape = cnn(torch.zeros(*self.rgb_shape)).shape

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

        orig_obs = input_dict[SampleBatch.OBS]
        # Lets check if orig_obs is flattened
        if (not isinstance(orig_obs, dict)) and type(self.original_space).__name__ == 'Dict':
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.original_space, tensorlib="torch"
            )
        processed_inputs = []
        # flat_inputs = input_dict["obs_flat"].float()
        for name in self.original_space:
            if name == 'RGB':
                processed_inputs.append(self.cnn(orig_obs[name].float().permute(0, 3, 1, 2)))
            else:
                tmp = orig_obs[name].float()
                processed_inputs.append(tmp.view(tmp.size(0), -1))
        prev_a = one_hot(input_dict[SampleBatch.PREV_ACTIONS].float(), self.action_space)
        processed_inputs = torch.cat(processed_inputs+[prev_a], dim=1)
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            processed_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        # Add previous action as input
        prev_a = input_dict[SampleBatch.PREV_ACTIONS]
        prev_a = one_hot(prev_a.float(), self.action_space)

        # output, new_state = self.forward_rnn(inputs, state, seq_lens, prev_action=prev_a)
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
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
    # def forward_rnn(self, inputs, state, seq_lens, prev_action: TensorType, **kwargs) -> Tuple[
    #     TensorType, List[TensorType]]:
    #     """
    #     @param prev_action: one-hot tensor of last action
    #     @param inputs: tensor of shape (B x T x H x W x C), where:
    #      B is the batch dimension, T is the time dimension, H is the observation height, W is the observation width,
    #      and C is the number of channels
    #     @param state: a tuple of (B x h) where h is the hidden state size
    #     @param seq_lens:
    #     @return:
    #     """
    #     # The input is reshaped to condense the B and T dimensions into one and swap the C dimension, to make the torch
    #     # CNN happy -> (B*T x C x H x W)
    #     x = inputs.reshape(-1, *self.rgb_shape)
    #     out = self.cnn(x)
    #     # Concatenate on the previous action
    #     out = torch.concat([out, prev_action], dim=1)
    #     # However, the LSTM does care about the B and T dimensions, so now we need to restore them.
    #     out, [h, c] = self.lstm(out.reshape(*inputs.shape[:2], -1),
    #                             [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
    #     self._features = self.post_lstm_fc1_act(self.post_lstm_fc1(out))
    #     action_out = self.action_branch(torch.squeeze(self._features))
    #     return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def forward_rnn(self, inputs, state, seq_lens):
        # self._features, [h, c] = self.lstm(inputs, [y0[None, :], y1[None, :]])
        self._features, [h, c] = self.lstm(inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        # batch_size, seq_len, hidden_size = self._features.size()
        # if self.img_aux_loss:
        #     x = self._features.contiguous().view(batch_size * seq_len, hidden_size)
        #     self.aux_image = self.auxDecoder(x).permute(0, 2, 3, 1)
        action_out = self.action_branch(self._features.squeeze(0))
        return action_out, [h.squeeze(0), c.squeeze(0)]