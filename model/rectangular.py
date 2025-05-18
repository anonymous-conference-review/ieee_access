from collections import OrderedDict

import math
import numpy as np
import torch
from torch import nn
from torch.nn import init

import lightning.pytorch as pl


class Concatenate(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, data):
        outputs = []
        for layer in self.layers:
            outputs.append(layer(data))
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return f'layers=({", ".join([repr(layer) for layer in self.layers])})'


class Rectangular(nn.Sequential, pl.LightningModule):
    def __init__(self, depth, width, input_shape, output_shape, activation='relu', kernel_size=3, use_flattening=False,
                 use_batchnorm=False, dropout_rate=0, n_maxpool=None, maxpool_mode=None, skip_connections=None,
                 init='kaiming'):
        print(f'<{skip_connections}>', type(skip_connections))
        self._dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.use_flattening = use_flattening
        self.skip_connections = skip_connections

        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.width = width
        self.depth = depth
        self.init = init
        self.n_maxpool = n_maxpool
        if self.n_maxpool is not None:
            if maxpool_mode is None:
                self.maxpool_mode = 'log'
            else:
                self.maxpool_mode = maxpool_mode

        layers = self.create_layers()
        layers = self.add_top(layers)
        super().__init__(layers)
        self.save_hyperparameters()

        if self.init != 'kaiming':
            self.initialize()

    def initialize(self):
        for layer in self:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if self.init == 'kaiming':
                    layer.reset_parameters()
                elif self.init == 'zero':
                    init.zeros_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)
                elif self.init == 'glorot':
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)
                else:
                    raise ValueError(f'Unknown init scheme {self.init}')

    def add_top(self, layers):
        if len(self.input_shape) > 1:  # conv
            if self.use_flattening:
                features = nn.Sequential(layers)
                if torch.cuda.is_available():
                    dtype = torch.cuda.FloatTensor
                else:
                    dtype = torch.FloatTensor
                dummy_x = torch.rand(2, *self.input_shape).type(dtype)
                outputs = features(dummy_x)
                flattened_input_shape = torch.prod(torch.tensor(outputs.shape[1:], requires_grad=False))
                layers['Flattening'] = nn.Flatten(1, -1)
                layers[f'Linear-{self.depth}'] = nn.Linear(flattened_input_shape, self.output_shape)
            else:
                global_pool = Concatenate([nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.AdaptiveMaxPool2d((1, 1))])

                # global_pool = nn.AdaptiveAvgPool2d((1, 1))
                layers[f'ConcatPool'] = global_pool
                layers['Flattening'] = nn.Flatten(1, -1)
                layers[f'Linear-{self.depth}'] = nn.Linear(self.width * 2, self.output_shape)
        else:  # dense
            layers[f'Linear-{self.depth}'] = nn.Linear(self.width, self.output_shape)

        return layers

    def create_layers(self):
        if self.activation not in {'relu', 'gelu'}:
            raise ValueError(f'Unknown activation {self.activation}')
        if len(self.input_shape) > 1 and self.n_maxpool:
            if self.maxpool_mode == 'log':
                maxpool_layers = [self.depth // 2]
                for _ in range(self.n_maxpool - 1):
                    maxpool_layers.append(maxpool_layers[-1] // 2)

                maxpool_layers = self.depth - np.array(maxpool_layers)
                # Adjust position due to the inclusion of the previous maxpool layers
                maxpool_layers += np.arange(len(maxpool_layers))
            elif self.maxpool_mode == 'linear':
                maxpool_layers = np.linspace(0, self.depth, num=self.n_maxpool)
            else:
                raise ValueError(f'Unknown mode {self.maxpool_mode}')
        else:
            maxpool_layers = []

        layers = OrderedDict()
        for d in range(self.depth + len(maxpool_layers)):
            if d in maxpool_layers:
                layers[f'MaxPool2d-{d}'] = nn.MaxPool2d(2)
            else:
                if len(self.input_shape) > 1:
                    layer = nn.Conv2d(in_channels=self.width if d > 0 else self.input_shape[0], out_channels=self.width,
                                      kernel_size=self.kernel_size, padding=1)
                    layers[f'Conv2d-{d}'] = layer
                else:
                    layer = nn.Linear(in_features=self.width if d > 0 else self.input_shape[0], out_features=self.width)
                    layers[f'Linear-{d}'] = layer

                if self.use_batchnorm:
                    if len(self.input_shape) > 1:
                        layers[f'BN-{d}'] = (nn.BatchNorm2d(self.width))
                    else:
                        layers[f'BN-{d}'] = nn.BatchNorm1d(self.width)
                if self.dropout_rate:
                    layers[f'Dropout-{d}'] = nn.Dropout(self.dropout_rate)

                if self.activation == 'relu':
                    layers[f'ReLU-{d}'] = nn.ReLU()
                elif self.activation  == 'gelu':
                    layers[f'GELU-{d}'] = nn.GELU()


        return layers

    def forward(self, input_):
        relu_count = 0
        residual = None
        for module in self:
            # for name, module in list(self.named_modules())[1:]:

            input_ = module(input_)
            if self.skip_connections:
                if isinstance(module, nn.ReLU):
                    if relu_count == 0:
                        residual = input_
                    if relu_count % self.skip_connections == 0:
                        input_ = input_ + residual
                        residual = input_

                    relu_count += 1

        return input_

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._dropout_rate = value
        dropout_layers = 0
        for module in self:
            # for name, module in list(self.named_modules())[1:]:
            if isinstance(module, nn.Dropout):
                dropout_layers += 1
                module.p = self.dropout_rate

        if dropout_layers == 0:
            raise ValueError('Setting dropout rate but no dropout layers found')
