from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.torch_utils import one_hot, flatten_inputs_to_1d_tensor
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.misc import SlimFC
import numpy as np

torch, nn = try_import_torch()


class BaselineCNNFCLSTM_test(RecurrentNetwork, nn.Module):
  def __init__(
          self,
          obs_space,
          action_space,
          num_outputs,
          model_config,
          name,
          **kwargs,
  ):

    nn.Module.__init__(self)
    super().__init__(obs_space, action_space, num_outputs, model_config, name)

    custom_model_config = {
      "lstm_cell_size": 256,
      "lstm_use_prev_action": True,
      "lstm_use_prev_reward": True,
      "lambda_rec": 1e-3,
      "lambda_inp": 1e-3,
      "lambda_img": 1e-3,
      "device": 'cpu',
      "conv_filters": [[16, [8, 8], 8], [128, [11, 11], 1]],
      "aux_loss": 'L1loss',
      "fcnet_hiddens": [256],
      'device': 'cpu',

    }
    custom_model_config |= model_config["custom_model_config"]
    self.model_config = custom_model_config
    self.cell_size = custom_model_config["lstm_cell_size"]
    self.time_major = custom_model_config.get("_time_major", False)
    self.use_prev_action = custom_model_config["lstm_use_prev_action"]
    self.use_prev_reward = custom_model_config["lstm_use_prev_reward"]
    self.lambda_rec = custom_model_config["lambda_rec"]
    self.lambda_inp = custom_model_config["lambda_inp"]
    self.lambda_img = custom_model_config["lambda_img"]
    self.device = custom_model_config.get("device", 'cpu')
    self.conv_filters = custom_model_config.get("conv_filters")
    self.aux_loss = self.model_config.get('aux_loss', 'L1loss')
    # Build the Module from fc + RNN + 2xfc (action + value outs).
    self.complex_input_net = ComplexInputNetwork(obs_space, action_space, [], custom_model_config, name)
    raw_input_dims = {name: int(np.product(component.shape)) for name, component in self.complex_input_net.obs_space.items()}
    # Calculate cumulative sum of raw input dimensions
    cumulative_sum = np.cumsum(list(raw_input_dims.values()))
    # Find the beginning and ending indices for 'RGB' in the flattened inputs
    start_index = 0 if 'RGB' not in self.complex_input_net.obs_space else cumulative_sum[np.argmax(np.array(list(self.complex_input_net.obs_space.keys())) == 'RGB')]
    end_index = start_index + raw_input_dims.get('RGB', 0)
    self.RGB_loc = [start_index, end_index]
    # The output is defined a little differently in LSTM
    self.interm_num_outputs = self.complex_input_net.num_outputs
    # Here, the action_dim is defined a little differently in LSTMWrapper. We need to test the effectiveness
    self.action_dim = self.action_space.n

    # Add prev-action/reward nodes to input to LSTM.
    if self.use_prev_action:
      self.interm_num_outputs += self.action_dim
    if self.use_prev_reward:
      self.interm_num_outputs += 1

    # Define actual LSTM layer (with num_outputs being the nodes coming from the wrapped (underlying) layer).
    self.lstm = nn.LSTM(self.interm_num_outputs, self.cell_size, batch_first=not self.time_major)
    self.add_module("lstm", self.lstm)
    self.num_outputs = num_outputs

    self.action_branch = nn.Linear(self.cell_size, num_outputs)
    self.value_branch = nn.Linear(self.cell_size, 1)

    # Add an auxiliary layer
    layers = []
    in_channels = self.cell_size
    layers.append(nn.Unflatten(dim=1, unflattened_size=(in_channels, 1, 1)),)
    num_conv = len(self.conv_filters)
    for i, (out_channels, kernel, stride) in enumerate(self.conv_filters[::-1]):
      if i == num_conv - 1:
        out_channels = 3
      layers.append(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride)
      )
      layers.append(nn.ReLU(inplace=True))  # You can use ReLU or other activation functions
      in_channels = out_channels
    self.auxDecoder = nn.Sequential(*layers)
    if self.aux_loss == 'L1loss':
      self.aux_loss_fn = nn.L1Loss()

    # Adapted from LSTMWrapper
    # __sphinx_doc_begin__
    # Add prev-a/r to this model's view, if required.
    if custom_model_config["lstm_use_prev_action"]:
      self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
        SampleBatch.ACTIONS, space=self.action_space, shift=-1
      )
    if custom_model_config["lstm_use_prev_reward"]:
      self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
        SampleBatch.REWARDS, shift=-1
      )
    # __sphinx_doc_end__


  @override(ModelV2)
  def forward(self, input_dict, state, seq_lens):
    inputs, _ = self.complex_input_net(input_dict, [], None)

    # Concat. prev-action/reward if required.
    prev_a_r = []
    if self.model_config.get("lstm_use_prev_action", False):
      prev_a = input_dict[SampleBatch.PREV_ACTIONS]
      if self.model_config.get("_disable_action_flattening", False):
        prev_a_r.append(flatten_inputs_to_1d_tensor(prev_a, spaces_struct=self.action_space_struct, time_axis=False))
      else:
        if isinstance(self.action_space, (Discrete, MultiDiscrete)):
          prev_a = one_hot(prev_a.float(), self.action_space)
        else:
          prev_a = prev_a.float()
        prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))
    # Prev rewards.
    if self.model_config["lstm_use_prev_reward"]:
      prev_a_r.append(torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(), [-1, 1]))

    # Concat prev. actions + rewards to the "main" input.
    if prev_a_r:
      wrapped_out = torch.cat([inputs] + prev_a_r, dim=1)

    # Push everything through our LSTM.
    input_dict["obs_flat"] = wrapped_out

    self.time_major = self.model_config.get("_time_major", False)
    inputs = add_time_dimension(
      wrapped_out,
      seq_lens=seq_lens,
      framework="torch",
      time_major=self.time_major,
    )

    ## TODO Corerct the following
    output, new_state = self.forward_rnn(inputs, state, seq_lens)
    output = torch.reshape(output, [-1, self.num_outputs])
    # import matplotlib.pyplot as plt
    # plt.subplot(2,1,1)
    # plt.imshow(self.aux_image.detach().cpu().numpy()[-1].astype(int))
    # plt.title(f'reconstructed' )
    # plt.axis('off')
    # plt.subplot(2,1,2)
    # plt.imshow(input_dict['obs']['RGB'][-1].cpu().numpy().astype(int))
    # plt.title('current obs')
    # # plt.subplot(3,1,3)
    # # plt.imshow(input_dict['new_obs'].cpu().numpy()[-1,1:-1].reshape(88,88,3).astype(int))
    # # plt.title('next obs')
    # plt.axis('off')
    # plt.show()
    return output, new_state

  @override(RecurrentNetwork)
  def forward_rnn(self, inputs, state, seq_lens):
    """Feeds `inputs` (B x T x ..) through the Gru Unit.
    Returns the resulting outputs as a sequence (B x T x ...).
    Values are stored in self._cur_value in simple (B) shape (where B
    contains both the B and T dims!).
    Returns:
        NN Outputs (B x T x ...) as sequence.
        The state batches as a List of two items (c- and h-states).
    """
    x = inputs
    y0 = torch.unsqueeze(state[0], 0)
    y1 = torch.unsqueeze(state[1], 0)

    # max_seq_len = inputs.shape[0]
    # time_major = self.model_config["_time_major"]
    # if time_major and max_seq_len > 1:
    #     inputs = torch.nn.utils.rnn.pack_padded_sequence(
    #         inputs, seq_lens,
    #         batch_first=not time_major, enforce_sorted=False)
    self._features, [h, c] = self.lstm(x, [y0, y1])
    batch_size, seq_len, hidden_size = self._features.size()
    x = self._features.contiguous().view(batch_size * seq_len, hidden_size)
    self.aux_image = self.auxDecoder(x).permute(0,2,3,1)
    # Re-apply paddings.
    # if time_major and max_seq_len > 1:
    #     self._features, _ = torch.nn.utils.rnn.pad_packed_sequence(
    #         self._features,
    #         batch_first=not time_major)
    action_out = self.action_branch(self._features)
    return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


  @override(ModelV2)
  def get_initial_state(self):
    # Place hidden states on same device as model.
    h = [
      self.action_branch.weight.new(1, self.cell_size).zero_().squeeze(0),
      self.action_branch.weight.new(1, self.cell_size).zero_().squeeze(0),
    ]
    return h


  @override(ModelV2)
  def custom_loss(self, policy_loss, loss_inputs):
    # import matplotlib.pyplot as plt
    # for i in range(50, 61):
    #   reconstruct_image = loss_inputs['obs'][i, 1:-1].reshape(88,88,3)
    #   reconstruct_image = reconstruct_image.numpy().astype(int)
    #   plt.imshow(reconstruct_image); plt.title(f'obs {i}'); plt.show()
    #   reconstruct_image = loss_inputs['new_obs'][i, 1:-1].reshape(88,88,3)
    #   reconstruct_image = reconstruct_image.numpy().astype(int)
    #   plt.imshow(reconstruct_image); plt.title(f'new obs {i}'); plt.show()
    # Recurrent matrix norm regularization
    l2_lambda_rec = self.lambda_rec
    V_norm = torch.tensor(0.).to(self.device)
    V_norm += torch.norm(self.lstm.weight_hh_l0).to(self.device)

    # Input matrix norm regularization
    l2_lambda_inp = self.lambda_inp
    W_norm = torch.tensor(0.).to(self.device)
    W_norm += torch.norm(self.lstm.weight_ih_l0).to(self.device)

    self.l2_loss = l2_lambda_rec * V_norm
    self.l2_loss_inp = l2_lambda_inp * W_norm
    self.original_loss = policy_loss

    assert self.l2_loss.requires_grad, "l2 loss no gradient"
    assert self.l2_loss_inp.requires_grad, "l2 loss no gradient"

    num_batch, width, height, num_color = self.aux_image.shape
    self.auxiliary_loss = self.aux_loss_fn(self.aux_image, loss_inputs['new_obs'][:,self.RGB_loc[0]:self.RGB_loc[1]].view(-1, width, height, num_color))

    custom_loss = self.l2_loss + self.l2_loss_inp + self.auxiliary_loss * self.lambda_img
    # depending on input add loss
    total_loss = [p_loss + custom_loss for p_loss in policy_loss]

    # total_loss = [p_loss + aux_loss for aux_loss in policy_loss]
    return total_loss

  def metrics(self):
    metrics = {
      "weight_loss": self.l2_loss.item(),
      # TODO Nguyen figure out if good or not
      "original_loss": self.original_loss[0].item(),
      "auxiliary_loss": self.auxiliary_loss.item(),
    }
    # you can print them to command line here. with Torch models its somehow not reported to the logger
    # print(metrics)

  @override(ModelV2)
  def value_function(self):
    assert self._features is not None, "must call forward() first"
    return torch.reshape(self.value_branch(self._features), [-1])