# class LinearProjector(nn.Module):

#     def __init__(self, visual_hidden_size, llm_hidden_size):
#         super().__init__()
#         self.layers = nn.Linear(visual_hidden_size, llm_hidden_size)
#         self.use_activation_checkpointing = False

#     def gradient_checkpointing_enable(self):
#         self.use_activation_checkpointing = True

#     def gradient_checkpointing_disable(self):
#         self.use_activation_checkpointing = False

#     def enable_input_require_grads(self):

#         def make_inputs_require_grad(module, input, output):
#             output.requires_grad_(True)

#         self.layers.register_forward_hook(make_inputs_require_grad)

#     def forward(self, x):
#         if self.use_activation_checkpointing and self.training:
#             layer_outputs = torch.utils.checkpoint.checkpoint(self.layers, x)
#         else:
#             layer_outputs = self.layers(x)
#         return layer_outputs

# class MLPProjector(nn.Module):

#     def __init__(self, visual_hidden_size, llm_hidden_size, depth=2):
#         super().__init__()
#         modules = [nn.Linear(visual_hidden_size, llm_hidden_size)]
#         for _ in range(1, depth):
#             modules.append(nn.GELU())
#             modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
#         self.layers = nn.Sequential(*modules)
#         self.use_activation_checkpointing = False

#     def gradient_checkpointing_enable(self):
#         self.use_activation_checkpointing = True

#     def gradient_checkpointing_disable(self):
#         self.use_activation_checkpointing = False

#     def enable_input_require_grads(self):

#         def make_inputs_require_grad(module, input, output):
#             output.requires_grad_(True)

#         self.layers.register_forward_hook(make_inputs_require_grad)

#     def forward(self, x):
#         if self.use_activation_checkpointing and self.training:
#             layer_outputs = torch.utils.checkpoint.checkpoint(self.layers, x)
#         else:
#             layer_outputs = self.layers(x)
#         return layer_outputs
