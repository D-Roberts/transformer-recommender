"""
Alibaba transformer based recommender.

"""

from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock, HybridSequential, LeakyReLU
from mxnet import gluon
import gluonnlp as nlp

from gluonnlp.model.seq2seq_encoder_decoder import _get_attention_cell
from gluonnlp.model.transformer import PositionwiseFFN

_SEQ_LEN = 32

def _position_encoding_init(max_length, dim):
    """Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    print(position_enc)
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

    return position_enc


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

            encoding = _position_encoding_init(_SEQ_LEN, 16)

            # TODO: add projection layer: Done
            self.proj = nn.Dense(units=16,
                                 use_bias=False,
                                 bias_initializer='zeros',
                                 weight_initializer=None,
                                 flatten=False
                                 )
            # self.Leaky_relu_layer = LeakyReLU(0.1)
            self.drop_out_layer = nn.Dropout(rate=0.1)
            #
            # # TODO: add pointwise feedforward network: Done. Use it now on outputs. Done. Recheck after layer complete
            # # for correct order and additions.
            self.ffn = PositionwiseFFN(hidden_size=16,
                                        use_residual=True,
                                                             dropout=0.1,
                                                             units=16,
                                                             weight_initializer=None,
                                                             bias_initializer='zeros')

            self.layer_norm = nn.LayerNorm(in_channels=16)

            self.output = HybridSequential()
            # final leaky rely layers; dimensions were 1024, 512, 256
            self.output.add(nn.Dense(8))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(4))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(2))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(1))



            # TODO: add position embedding; do sin and cos right now but alibaba does something else:
             # TODO: likely ffn does the positional encoding; check in basetransformer cell where ffn(output) what
            # will ffn do with the argument if it will combine it with the positional stuff; if yes, then until
            # I change the type of positional encoding, the sinusoidal stuff is already added from importing
            # PositionFFN
            # TODO: add other features
            # TODO: fix the transformer encoder to be like in the ALIbaba paper
            # TODO: push omline
            # TODO: change position encoding like in alibaba paper; likely the entire POsitionFFN class must be changed.
            # TODO: feed movie data from Google

    def hybrid_forward(self, F, x, mask=None):
        x = self.features(x)

        # attention cell
        out_x, attn_w = self.cell(x, x, x, mask)
        # project
        out_x = self.proj(out_x)

        # dropout as in nmt

        x = self.drop_out_layer(out_x)

        # add and norm
        x = x + out_x
        x = self.layer_norm(x)

        # ffn
        x = self.ffn(x)
        x = self.output(x)

        return x
    # TODO: fix the layer norm

# Constructor
def ali_rec(**kwargs):
    net = Rec(**kwargs)
    return net