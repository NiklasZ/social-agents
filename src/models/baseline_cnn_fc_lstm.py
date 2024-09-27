import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import deprecation_warning
from ray.rllib.utils.annotations import override
import torch
import tree  # pip install dm_tree
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.util import log_once
from torch import nn, TensorType
from ray.rllib.models.torch.misc import SlimConv2d, same_padding


class BaselineCNNFCLSTM(RecurrentNetwork, nn.Module):
  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    nn.Module.__init__(self)
    super().__init__(obs_space, action_space, num_outputs, model_config, name)
    # Define import model parameters. TODO: Move these to config
    self.lstm_state_size = 256
    self.cnn_output_size = 128
    self.fcnet_hiddens = 64
    self.post_preprocess_size = 0
    self.conv_filters = [[16, [8, 8], 8], [128, [8, 8], 1]]

    self.obs_size = get_preprocessor(obs_space)(obs_space).size
    self.original_space = getattr(obs_space, "original_space", obs_space)
    self.flattened_input_space = flatten_space(self.original_space)
    # self.input_shape = np.flip(obs_space.shape)  # (H, W, C) to (C, H, W)

    # Define view requirements
    self.view_requirements['model_prev_action'] = ViewRequirement(
      data_col=SampleBatch.ACTIONS, space=self.action_space, shift=-1
    )

    # Initialize CNN and FC layers
    self.flatten, self.cnns = {}, {}
    for i, component in enumerate(self.flattened_input_space):
      if len(component.shape) != 3:
        flat_shape = int(np.product(component.shape))
        self.flatten[i] = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(flat_shape, self.fcnet_hiddens)),
          ('ReLU1', nn.ReLU()),
          ('fc2', nn.Linear(self.fcnet_hiddens, self.fcnet_hiddens)),
          ('ReLU2', nn.ReLU()),
        ]))
        self.post_preprocess_size += self.fcnet_hiddens
      else:
        self._build_cnn(component, i)

    self.lstm_input_size = self.post_preprocess_size + self.action_space.n
    self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_state_size, batch_first=True)
    self.post_lstm_fc1 = nn.Linear(self.lstm_state_size, 256)
    self.post_lstm_fc1_act = nn.ReLU()

    # Action and value branches
    self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
    self.value_branch = nn.Linear(self.lstm_state_size, 1)
    self._features = None

  def _build_cnn(self, component, index):
    """
    Constructs the CNN layers using traditional PyTorch layers instead of SlimConv2d.

    Args:
        component: The observation component (an image with shape [H, W, C]).
        index: The index of the observation component.

    Returns:
        None: The layers are assigned to self.cnns[index].
    """

    layers = []
    in_channels = component.shape[2]  # The number of channels in the input image
    in_size = component.shape[:2]  # The height and width of the input image

    # Define your CNN architecture
    for out_channels, kernel_size, stride in self.conv_filters[:-1]:
      # padding = self._compute_padding(in_size, kernel_size, stride)  # Calculate padding
      padding = [0,0]
      layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
      layers.append(nn.ReLU())
      in_channels = out_channels  # Update in_channels for the next layer
      in_size = self._compute_output_size(in_size, kernel_size, stride, padding)  # Update the output size

    # Add the last Conv2d layer
    out_channels, kernel_size, stride = self.conv_filters[-1]
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    layers.append(nn.ReLU())

    # Flatten the output from the CNN to feed into the LSTM or fully connected layers
    layers.append(nn.Flatten())

    # Assign the sequential CNN model to self.cnns for the corresponding component
    self.cnns[index] = nn.Sequential(*layers)

    # Add post cnn fc layers
    out_channels, kernel_size, stride = self.conv_filters[-1]
    out_size = self._compute_output_size(in_size, kernel_size, stride, padding)
    post_cnn_flatten_shape = out_channels * out_size[0] * out_size[1]
    self.cnns[index].add_module('post_cnn_fc1', nn.Linear(post_cnn_flatten_shape, self.cnn_output_size))
    self.cnns[index].add_module('post_cnn_fc1_act', nn.ReLU())

    # Move the CNN to the appropriate device
    # self.cnns[index].to(self.device)

    # Update the post-preprocess size to match the final output channels of the CNN
    self.post_preprocess_size += out_channels

  def _compute_padding(self, input_size, kernel_size, stride):
    """Helper function to compute the padding for 'same' padding with multi-dimensional kernel size."""
    pad_h = max((input_size[0] - 1) * stride + kernel_size[0] - input_size[0], 0)
    pad_w = max((input_size[1] - 1) * stride + kernel_size[1] - input_size[1], 0)
    return (pad_h // 2, pad_w // 2)

  def _compute_output_size(self, input_size, kernel_size, stride, padding):
    """Helper function to compute the output size after a Conv2d layer with multi-dimensional kernel size."""
    output_height = (input_size[0] + 2 * padding[0] - kernel_size[0]) // stride + 1
    output_width = (input_size[1] + 2 * padding[1] - kernel_size[1]) // stride + 1
    return output_height, output_width

  @override(RecurrentNetwork)
  def forward(self, input_dict, state, seq_lens):
    """Forward pass through the network. We first process the inputs via CNN/FC layers,
    then pass through the LSTM, and finally through the action and value branches."""

    orig_obs = input_dict[SampleBatch.OBS]
    # Lets check if orig_obs is flattened
    if (not isinstance(orig_obs, dict)) and type(self.original_space).__name__ == 'Dict':
      orig_obs = restore_original_dimensions(
                  input_dict[SampleBatch.OBS], self.original_space, tensorlib="torch"
              )
    processed_inputs = []

    # Iterate through the observation components
    for idx, obs_component in enumerate(tree.flatten(orig_obs)):
      # obs_component = obs_component.float().to(self.device)
      obs_component = obs_component.float()

      # If observation is not an image (non-3D), process it through FC layers
      if idx in self.flatten:
        # if len(obs_component) ==140:
        #   raise ValueError("obs_component is 140")
        if (len(obs_component.shape) == 1):
          # obs_component = torch.reshape(obs_component, (input_dict[SampleBatch.OBS].shape[0], -1))
          obs_component = obs_component.view(len(input_dict), -1)
        processed_inputs.append(self.flatten[idx](obs_component))
      else:
        # If it's an image (3D), process it through the CNN layers
        img_inputs = obs_component.permute(0, 3, 1, 2)  # (B, C, H, W) format for CNN
        processed_inputs.append(self.cnns[idx](img_inputs))
    # Concatenate all processed inputs (from both CNN and FC layers)
    concat_inputs = torch.cat(processed_inputs, dim=1)

    # Add previous action as additional input
    # prev_a = one_hot(input_dict[SampleBatch.PREV_ACTIONS].float(), self.action_space).to(self.device)
    prev_a = one_hot(input_dict[SampleBatch.PREV_ACTIONS].float(), self.action_space)
    concat_inputs = torch.cat([concat_inputs, prev_a], dim=1)

    # Use add_time_dimension to handle sequence lengths
    lstm_inputs = add_time_dimension(
      concat_inputs,
      seq_lens=seq_lens,
      framework="torch",
      time_major=self.model_config.get("_time_major", False)
    )

    # Pass the concatenated input through the LSTM
    lstm_out, [h, c] = self.lstm(lstm_inputs, [state[0].unsqueeze(0), state[1].unsqueeze(0)])

    # Apply the fully connected layer post-LSTM
    self._features = self.post_lstm_fc1_act(self.post_lstm_fc1(lstm_out))

    # Pass the LSTM output to the action and value branches
    action_out = self.action_branch(self._features.view(-1, self.lstm_state_size))  # Ensures correct reshape
    return action_out, [h.squeeze(0), c.squeeze(0)]

  @override(ModelV2)
  def get_initial_state(self):
    h, c = torch.zeros(self.lstm_state_size), torch.zeros(self.lstm_state_size)
    return [h, c]

  @override(ModelV2)
  def value_function(self):
    return self.value_branch(self._features).view(-1)

@override(RecurrentNetwork)
def forward_rnn(self, inputs, state, seq_lens, **kwargs):
  """
  Forward pass through the LSTM.
  Since `prev_a` is already concatenated into the inputs in the forward function,
  we don't need to handle it separately here.

  Args:
      inputs: Tensor of shape (B x T x input_size), where:
          B is the batch dimension, T is the time dimension, and input_size is the size of the concatenated input.
      state: A tuple of (h, c), where h is the hidden state and c is the cell state.
      seq_lens: A tensor containing the sequence lengths for each sample in the batch.

  Returns:
      action_out: The output after passing through the action branch.
      new_state: The updated LSTM state (h, c).
  """

  # Pass the inputs through the LSTM
  lstm_out, [h, c] = self.lstm(inputs, [state[0].unsqueeze(0), state[1].unsqueeze(0)])

  # Apply the fully connected layer post-LSTM
  self._features = self.post_lstm_fc1_act(self.post_lstm_fc1(lstm_out))

  # Pass the LSTM output to the action branch
  action_out = self.action_branch(self._features.view(-1, self.lstm_state_size))  # Reshape for action output

  # Return the action output and the updated LSTM state
  return action_out, [h.squeeze(0), c.squeeze(0)]

