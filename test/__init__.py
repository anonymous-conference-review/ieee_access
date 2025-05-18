import torch
import torch.nn.functional as F
from lightning import seed_everything
from torch import nn
from torch.nn import Sequential, GELU


class TestNetwork(nn.Module):
    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear0 = nn.Linear(2, 4)
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(4, 4)
        self.linear4 = nn.Linear(4, 4)
        self.linear5 = nn.Linear(4, 4)
        self.linear6 = nn.Linear(4, 4)
        self.linear7 = nn.Linear(4, 4)
        self.linear8 = nn.Linear(4, 4)
        self.linear9 = nn.Linear(4, 4)
        self.linear10 = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)
        # self.output.bias.data = torch.tensor([1], dtype=torch.float)

        self.seed = seed
        if seed:
            seed_everything(seed)

    def forward(self, x):
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        x = F.relu(x)
        x = self.linear7(x)
        x = F.relu(x)
        x = self.linear8(x)
        x = F.relu(x)
        x = self.linear9(x)
        x = F.relu(x)
        x = self.linear10(x)
        x = F.relu(x)
        x = self.output(x)
        return x


def get_moons_2():
    #  model.add_unit(weight=[-1, 1], bias=-0.7, node_id=1, activation='relu')
    #         model.add_unit(weight=[-1, -1], bias=0.8, node_id=2, activation='relu')
    #         model.add_unit(weight=[1, -1], bias=-1.2, node_id=3, activation='relu')
    #
    #         model.add_unit([-10, 1, 1], 0, dependencies=[1, 2, 3], node_id=4)

    layer_1 = nn.Linear(2, 3, bias=True)
    layer_2 = nn.Linear(3, 1, bias=True)
    layer_1.weight.data = torch.tensor([[-1.3, 0.7], [-1.1, -0.9], [1.3, -0.7]], dtype=torch.float)
    layer_1.bias.data = torch.tensor([-0.6, 0.9, -1.7], dtype=torch.float)
    layer_2.weight.data = torch.tensor([[-10, 1, 1]], dtype=torch.float)
    layer_2.bias.data = torch.tensor([0], dtype=torch.float)
    moons_2_model = Sequential(layer_1, nn.ReLU(), layer_2)

    return moons_2_model


def get_moons_6():
    layer_1 = nn.Linear(2, 3, bias=True)
    layer_2 = nn.Linear(3, 3, bias=True)
    layer_3 = nn.Linear(3, 3, bias=True)
    layer_4 = nn.Linear(3, 3, bias=True)
    layer_5 = nn.Linear(3, 3, bias=True)
    layer_6 = nn.Linear(3, 1, bias=True)

    forward_weights = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    forward_biases = [0, 0, 0]
    layer_1.weight.data = torch.tensor([[-1.3, 0.7], [-1.1, -0.9], [1.3, -0.7]], dtype=torch.float)
    layer_1.bias.data = torch.tensor([-0.6, 0.9, -1.7], dtype=torch.float)
    layer_2.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_2.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_3.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_3.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_4.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_4.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_5.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_5.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_6.weight.data = torch.tensor([[-7, 1, 2]], dtype=torch.float)
    layer_6.bias.data = torch.tensor([0], dtype=torch.float)

    moons_6_model = Sequential(layer_1, nn.ReLU(), layer_2, nn.ReLU(), layer_3, nn.ReLU(), layer_4, nn.ReLU(), layer_5,
                               nn.ReLU(), layer_6)

    return moons_6_model

def get_moons_scaled():
    layer_1 = nn.Linear(2, 3, bias=True)
    layer_2 = nn.Linear(3, 3, bias=True)
    layer_3 = nn.Linear(3, 3, bias=True)
    layer_4 = nn.Linear(3, 3, bias=True)
    layer_5 = nn.Linear(3, 3, bias=True)
    layer_6 = nn.Linear(3, 1, bias=True)

    forward_weights = [[2, 0, 0], [0, 0.5, 0], [0, 0, 2]]
    forward_biases = [0, 0, 0]
    layer_1.weight.data = torch.tensor([[-1.3, 0.7], [-1.1, -0.9], [1.3, -0.7]], dtype=torch.float)
    layer_1.bias.data = torch.tensor([-0.6, 0.9, -1.7], dtype=torch.float)
    layer_2.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_2.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_3.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_3.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_4.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_4.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_5.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_5.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_6.weight.data = torch.tensor([[-7, 1, 2]], dtype=torch.float)
    layer_6.bias.data = torch.tensor([0], dtype=torch.float)

    moons_scaled_model = Sequential(layer_1, nn.ReLU(), layer_2, nn.ReLU(), layer_3, nn.ReLU(), layer_4, nn.ReLU(), layer_5,
                               nn.ReLU(), layer_6)

    return moons_scaled_model

def get_moons_affine():
    layer_1 = nn.Linear(2, 3, bias=True)
    layer_2 = nn.Linear(3, 3, bias=True)
    layer_3 = nn.Linear(3, 3, bias=True)
    layer_4 = nn.Linear(3, 3, bias=True)
    layer_5 = nn.Linear(3, 3, bias=True)
    layer_6 = nn.Linear(3, 1, bias=True)

    forward_weights = [[0, 2, 0], [0.5, 0, 0], [0, 0, 2]]
    forward_biases = [1, 0.5, 2]
    layer_1.weight.data = torch.tensor([[-1.3, 0.7], [-1.1, -0.9], [1.3, -0.7]], dtype=torch.float)
    layer_1.bias.data = torch.tensor([-0.6, 0.9, -1.7], dtype=torch.float)
    layer_2.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_2.bias.data = torch.tensor(forward_biases , dtype=torch.float)
    layer_3.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_3.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_4.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_4.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_5.weight.data = torch.tensor(forward_weights, dtype=torch.float)
    layer_5.bias.data = torch.tensor(forward_biases, dtype=torch.float)
    layer_6.weight.data = torch.tensor([[-7, 1, 2]], dtype=torch.float)
    layer_6.bias.data = torch.tensor([-34], dtype=torch.float)

    moons_affine_model = Sequential(layer_1, nn.ReLU(), layer_2, nn.ReLU(), layer_3, nn.ReLU(), layer_4, nn.ReLU(), layer_5,
                               nn.ReLU(), layer_6)

    return moons_affine_model


def get_random_alive_network(num_layers=5, input_dim=2, width=4, activation='relu', output_activation=None, attempts=5):
    if attempts <= 0:
        raise ValueError('Attempts must be greater than 0')
    network = None
    for _ in range(attempts):
        network = get_random_network(num_layers, input_dim, width, activation, output_activation)
        is_dead = is_network_dead(network, input_dim=input_dim)
        if not is_dead:
            return network
    print('Failed to generate a non-dead network')
    return network


def is_network_dead(network, input_dim=2):
    x = torch.randn(10, input_dim)
    y = network(x)
    unique = torch.unique(y)
    return len(unique) == 1


def get_random_network(num_layers=5, input_dim=2, output_dim=1, width=4, activation='relu', output_activation=None):
    layers = []
    for i in range(num_layers - 1):
        if i == 0:
            layers.append(nn.Linear(input_dim, width))
        else:
            layers.append(nn.Linear(width, width))
        layers.append(get_activation(activation))
    layers.append(nn.Linear(width, output_dim))
    if output_activation:
        layers.append(get_activation(output_activation))

    network = Sequential(*layers)
    network.width = width
    network.num_layers = num_layers
    return network


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softmax':
        return nn.Softmax()
    elif activation == 'gelu':
        return GELU()
    else:
        raise ValueError(f'Unknown activation {activation}')
