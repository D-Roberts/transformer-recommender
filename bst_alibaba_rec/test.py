"""
Bare bones test for alibaba rec codes.

"""

import mxnet as mx
from BST_rec import Rec


_SEQ_LEN = 32
_BATCH = 1
ctx = mx.cpu()

def _tst_module(net, x):
    net.initialize()
    net.collect_parameters()
    net(x)
    mx.nd.waitall()

def test():
    x = mx.random.uniform(shape=(_BATCH, _SEQ_LEN), ctx=ctx)
    net = Rec()
    _tst_module(net, x)

def main():
    import nose
    nose.runmodule()