# msfid_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class InceptionV3MultiFeature(nn.Module):
    def __init__(self, layers=['Mixed_5d', 'Mixed_6e', 'Mixed_7c']):
        super().__init__()
        self.model = models.inception_v3(pretrained=True, aux_logits=True, transform_input=True)
        self.model.eval()
        self.outputs = {}
        self.layers = layers

        # Register forward hooks for intermediate layers
        for name in layers:
            block = dict(self.model.named_children())[name]
            block.register_forward_hook(self._get_hook(name))

    def _get_hook(self, name):
        def hook(_, __, output):
            self.outputs[name] = output.detach()
        return hook

    def forward(self, x):
        self.outputs.clear()
        _ = self.model(x)
        return [self.outputs[name] for name in self.layers]
