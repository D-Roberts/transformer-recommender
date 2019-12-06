"""
Bare bones test for alibaba rec.

"""

import mxnet as mx
from bst_alibaba_rec import BST_rec

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
    net = BST_rec.Rec()
    _tst_module(net, x)

def main():
    import nose
    nose.runmodule()