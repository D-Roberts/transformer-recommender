"""

BST recommender.
"""

import mxnet as mx
from mxnet import gluon
import numpy as np
import random
from mxnet import autograd as ag

from bst_alibaba_rec import BST_rec

np.random.seed(100)
ctx = mx.cpu()
mx.random.seed(100)
random.seed(100)


_SEQ_LEN = 32
_BATCH = 1
ctx = mx.cpu()


def generate_sample():
    """Generate toy X and y. One target item.
    """
    X = mx.random.uniform(shape=(100, 64))

    y = mx.random.uniform(shape=(100, 1))
    y = y > 0.5
    # Data loader
    d = gluon.data.ArrayDataset(X, y)
    return gluon.data.DataLoader(d, _BATCH, last_batch='keep')

def train():
    train_data = generate_sample()
    optimizer = mx.optimizer.Adam()

    # Binary classification problem; predict if user clicks target item
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    net = BST_rec.Rec()
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer)
    train_metric = mx.metric.Accuracy()

    epochs = 1

    for epoch in range(epochs):
        train_metric.reset()

        for x, y in train_data:
            with ag.record():
                output = net(x[:,:32], x[:, 32:])
                l = loss(output, y).sum()

        l.backward()
        trainer.step(_BATCH)
        train_metric.update(y, output)



def main():
    train()


if __name__ == '__main__':
    main()



