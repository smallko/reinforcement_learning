#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import parl
from parl import layers  # 封装了 paddle.fluid.layers 的API


class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 512
        hid2_size = 512
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        #self.fc3 = layers.fc(size=act_dim, act=None)

        #self.conv1 = layers.conv2d(
        #    num_filters=32, filter_size=5, stride=1, padding=2, act='relu')
        #self.conv2 = layers.conv2d(
        #    num_filters=32, filter_size=5, stride=1, padding=2, act='relu')
        #self.conv3 = layers.conv2d(
        #    num_filters=64, filter_size=4, stride=1, padding=1, act='relu')
        #self.conv4 = layers.conv2d(
        #    num_filters=64, filter_size=3, stride=1, padding=1, act='relu')

        self.fc1_adv = layers.fc(size=512, act='relu')
        self.fc2_adv = layers.fc(size=act_dim)
        self.fc1_val = layers.fc(size=512, act='relu')
        self.fc2_val = layers.fc(size=1)

    def value(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        #Q = self.fc3(h2)

        As = self.fc2_adv(self.fc1_adv(out))
        V = self.fc2_val(self.fc1_val(out))
        Q = As + (V - layers.reduce_mean(As, dim=1, keep_dim=True))

        return Q
