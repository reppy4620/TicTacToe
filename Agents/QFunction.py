
import chainer
import chainer.functions as F
import chainer.links as L

import chainerrl


class QFunction(chainer.Chain):

    def __init__(self):
        super().__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv2 = L.Convolution2D(64, 128, 3, 1, 1)

            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(128)

            self.fc1 = L.Linear(128 * 3 * 3, 512)
            self.fc_bn1 = L.BatchNormalization(512)

            self.fc2 = L.Linear(512, 256)
            self.fc_bn2 = L.BatchNormalization(256)

            self.out = L.Linear(256, 9)

    def __call__(self, state, test=False):
        h = F.relu(self.bn1(self.conv1(state)))
        h = F.relu(self.bn2(self.conv2(h)))

        h = F.relu(self.fc_bn1(self.fc1(h)))
        h = F.relu(self.fc_bn2(self.fc2(h)))
        h = self.out(h)
        return chainerrl.action_value.DiscreteActionValue(h)
