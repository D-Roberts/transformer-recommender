"""
Alibaba transformer based recommender.

"""

from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock, HybridSequential, LeakyReLU
from mxnet import gluon
import gluonnlp as nlp

from gluonnlp.model.seq2seq_encoder_decoder import _get_attention_cell

_SEQ_LEN = 32

# TODO:
# -add transformer layers; prolly start with transformer block

# be patient with yourself; this is a lot of stuff you havent worked with the transformer before
# _ yes what I need to look at is the transformer encoder portion of the gluon nlp encoder decoder model.
# plan: replicate that code here
# - first try to import classes and use the Encoder class directly - goal how to import

# Ok need to backoff and to understand the transformer encoder and the code
# start first with the use of the encoder
# only encoder portion for the alibaba rec in the transformer layer. Just do one transformer layer for now.
# It consists of initial item and position embeddings; after that: multi-head attention; then add and norm;
# then a feed forward layer; then add and norm; then the output of this thransformer layer goes into the next
# leaky rely layers.

# TODO: today - add a multi-head attention cell to the HybridSequential.DONE. Separate from Hyrbid though. DONE
# TODO: next transformer layer: add and norm


class Rec(HybridBlock):
    def __init__(self, **kwargs):
        super(Rec, self).__init__(**kwargs)
        with self.name_scope():
            self.features = HybridSequential()

            # TODO: add other features and positional encoding.
            self.features.add(nn.Embedding(input_dim=_SEQ_LEN,
                                           output_dim=16))


            self.cell = _get_attention_cell('multi_head',
                                                      units=16,
                                                      scaled=True,
                                                      dropout=0.0,
                                                      num_heads=4,
                                                      use_bias=False)

            # TODO



            self.output = HybridSequential()
            # final leaky rely layers; dimensions were 1024, 512, 256
            self.output.add(nn.Dense(8))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(4))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(2))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(1))

    def hybrid_forward(self, F, x, mask=None):
        x = self.features(x)
        x, attn_w = self.cell(x, x, x, mask)
        x = self.output(x)
        return x


# Constructor
def ali_rec(**kwargs):
    net = Rec(**kwargs)
    return net