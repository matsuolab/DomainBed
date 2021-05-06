import torch
import timm


class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLPMixer(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = getattr(timm.models.mlp_mixer, hparams['backbone'])
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
