"""
Barebones BST transformer based recommender.
Not all hyperparameter values were disclosed in the published article. Toy values used instead.
Best on Python3.8.
"""

import mxnet as mx
import numpy as np
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock, HybridSequential, LeakyReLU

from transformer_blocks import _get_attention_cell, PositionwiseFFN

_SEQ_LEN = 32
_OTHER_LEN = 32
_EMB_DIM = 32
_NUM_HEADS = 8
_DROP = 0.2
_UNITS = 32

def _position_encoding_init(max_length, dim):
    """Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    return position_enc

def _position_encoding_init_BST(max_length, dim):
    """For the BST recommender, the positional embedding takes the time of item being clicked as
    input feature and calculates the position value of item vi as p(vt) - p(vi) where
    p(vt) is recommending time and p(vi) is time the user clicked on item vi

    """
    # Assume position_enc is the p(vt) - p(vi) fed as input
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    return position_enc

class Rec(HybridBlock):
    """Alibaba transformer based recommender"""
    def __init__(self, **kwargs):
        super(Rec, self).__init__(**kwargs)
        with self.name_scope():
            self.otherfeatures = nn.Embedding(input_dim=_OTHER_LEN,
                                               output_dim=_EMB_DIM)
            self.features = nn.Embedding(input_dim=_SEQ_LEN,
                                           output_dim=_EMB_DIM)
            # Transformer layers
            # Multi-head attention with base cell scaled dot-product attention
            # Use b=1 self-attention blocks per article recommendation
            self.cell = _get_attention_cell('multi_head',
                                            units=_UNITS,
                                            scaled=True,
                                            dropout=_DROP,
                                            num_heads=_NUM_HEADS,
                                            use_bias=False)
            self.proj = nn.Dense(units=_UNITS,
                                 use_bias=False,
                                 bias_initializer='zeros',
                                 weight_initializer=None,
                                 flatten=False
                                 )
            self.drop_out_layer = nn.Dropout(rate=_DROP)
            self.ffn = PositionwiseFFN(hidden_size=_UNITS,
                                       use_residual=True,
                                       dropout=_DROP,
                                       units=_UNITS,
                                       weight_initializer=None,
                                       bias_initializer='zeros',
                                       activation='leakyrelu'
                                       )
            self.layer_norm = nn.LayerNorm(in_channels=_UNITS)
            # Final MLP layers; BST dimensions in the article were 1024, 512, 256
            self.output = HybridSequential()
            self.output.add(nn.Dense(8))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(4))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(2))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(1))

    def _arange_like(self, F, inputs, axis):
        """Helper function to generate indices of a range"""
        if F == mx.ndarray:
            seq_len = inputs.shape[axis]
            arange = F.arange(seq_len, dtype=inputs.dtype, ctx=inputs.context)
        else:
            input_axis = inputs.slice(begin=(0, 0, 0), end=(1, None, 1)).reshape((-1))
            zeros = F.zeros_like(input_axis)
            arange = F.arange(start=0, repeat=1, step=1,
                              infer_range=True, dtype=inputs.dtype)
            arange = F.elemwise_add(arange, zeros)
            # print(arange)
        return arange

    def _get_positional(self, weight_type, max_length, units):
        if weight_type == 'sinusoidal':
            encoding = _position_encoding_init(max_length, units)
        elif weight_type == 'BST':
            # BST position fed as input
            encoding = _position_encoding_init_BST(max_length, units)
        else:
            raise ValueError('Not known')
        return mx.nd.array(encoding)

    def hybrid_forward(self, F, x, x_other, mask=None):
        # The manually engineered features
        x1 = self.otherfeatures(x_other)

        # The transformer encoder
        steps = self._arange_like(F, x, axis=1)
        x = self.features(x)
        position_weight = self._get_positional('BST', _SEQ_LEN, _UNITS)
        # add positional embedding
        positional_embedding = F.Embedding(steps, position_weight, _SEQ_LEN, _UNITS)
        x = F.broadcast_add(x, F.expand_dims(positional_embedding, axis=0))
        # attention cell with dropout
        out_x, attn_w = self.cell(x, x, x, mask)
        out_x = self.proj(out_x)
        out_x = self.drop_out_layer(out_x)
        # add and norm
        out_x = x + out_x
        out_x = self.layer_norm(out_x)
        # ffn
        out_x = self.ffn(out_x)

        # concat engineered features with transformer representations
        out_x = mx.ndarray.concat(out_x, x1)
        # leakyrelu final layers
        out_x = self.output(out_x)
        return out_x
