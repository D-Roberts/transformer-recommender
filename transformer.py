"""

Replicate NVIDIA transformer implementation to be used with alibaba recommender.
"""

import mxnet as mx
from mxnet import gluon, nd
import numpy as np
import gluonnlp as nlp
import random
from mxnet import autograd as ag

import model

np.random.seed(100)
ctx = mx.cpu()
mx.random.seed(100)
random.seed(100)


_SEQ_LEN = 32
_BATCH = 1
ctx = mx.cpu()

# TODO:
# - READ ALIBABA PAPER TO UNDERSTAND THE MODEL
# - Code the model class with a test with random generated data similar to quo vadis. This will be the train file.
# - GET THE DATA FROM GOOGLE MOVIE REC AND GET IT READY FOR THE MODEL.


# TODO (small):
# - set up the train file - bare bones (see quo vadis)(DONE); start with artificially generated data (done)
# - next add embedding layer; and last three leaky relus layers

def generate_sample():
    """Generate X and y."""
    X = mx.random.uniform(shape=(100, 32))
    y = mx.random.uniform(shape=(100, 1))
    y = y > 0.5
    # Data loader for a simple MLP to start with
    d = gluon.data.ArrayDataset(X, y)
    return gluon.data.DataLoader(d, _BATCH, last_batch='keep')

def train():
    train_data = generate_sample()
    optimizer = mx.optimizer.Adam()
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    net = model.Rec()
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer)
    train_metric = mx.metric.Accuracy()

    epochs = 1

    for epoch in range(epochs):
        train_metric.reset()

        for x, y in train_data:
            with ag.record():
                output = net(x)
                l = loss(output, y).sum()

        l.backward()
        trainer.step(_BATCH)
        train_metric.update(y, output)


def main():
    train()

if __name__ == '__main__':
    main()



