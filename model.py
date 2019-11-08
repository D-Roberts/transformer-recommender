"""
Alibaba transformer based recommender.

"""

from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock, HybridSequential, LeakyReLU
from mxnet import gluon

_SEQ_LEN = 32


class Rec(HybridBlock):
    def __init__(self, **kwargs):
        super(Rec, self).__init__(**kwargs)
        with self.name_scope():
            self.features = HybridSequential()

            # TODO: add other features and positional encoding.
            self.features.add(nn.Embedding(input_dim=_SEQ_LEN,
                                           output_dim=16))

            # Transformer layer
            self.features.add(nn.Dense(2, activation='relu'))

            self.output = HybridSequential()
            # final leaky rely layers; dimensions were 1024, 512, 256
            self.output.add(nn.Dense(8))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(4))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(2))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(1))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Constructor
def ali_rec(**kwargs):
    net = Rec(**kwargs)
    return net