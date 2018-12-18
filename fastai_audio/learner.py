from fastai.torch_core import *
from fastai.train import Learner

from fastai.callbacks.hooks import num_features_model
from fastai.vision import create_body, create_head
from fastai.vision.learner import cnn_config, _resnet_split

__all__ = ['create_cnn']


# copied from fastai.vision.learner, omitting unused args,
# and adding channel summing of first convolutional layer
def create_cnn(data, arch, pretrained=True, sum_channel_weights=True, **kwargs):
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)

    # sum up the weights of in_channels axis, to reduce to single input channel
    # Suggestion by David Gutman
    # https://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/2
    if sum_channel_weights:
        first_conv_layer = body[0]
        first_conv_weights = first_conv_layer.state_dict()['weight']
        assert first_conv_weights.size(1) == 3 # RGB channels dim
        summed_weights = torch.sum(first_conv_weights, dim=1, keepdim=True)
        first_conv_layer.weight.data = summed_weights
        first_conv_layer.in_channels = 1

    nf = num_features_model(body) * 2
    head = create_head(nf, data.c, None, 0.5)
    model = nn.Sequential(body, head)
    learn = Learner(data, model, **kwargs)
    learn.split(meta['split'])
    if pretrained:
        learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)
    return learn
