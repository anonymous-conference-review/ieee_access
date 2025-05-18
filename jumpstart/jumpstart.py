from functools import partial

import torch
import torch.nn.functional as F


def compute_max_preact_average(preact, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    dim = tuple(set(range(preact.ndim)) - {1 - ax})
    max_preact = preact.amax(dim=dim)
    return max_preact


def compute_min_preact_average(preact, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    dim = tuple(set(range(preact.ndim)) - {1 - ax})
    min_preact = preact.amin(dim=dim)
    return min_preact


def compute_max_preact_single(preact, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    x_max = preact.max(dim=ax)[0]

    # Since pytorch can't compute max across multiple dimensions, flatten the rest and aggregate
    max_preact = x_max.view(x_max.shape[0], -1).max(dim=1)[0]

    return max_preact


def compute_min_preact_single(preact, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    x_min = preact.min(dim=ax)[0]

    # Since pytorch can't compute max across multiple dimensions, flatten the rest and aggregate
    min_preact = x_min.view(x_min.shape[0], -1).min(dim=1)[0]

    return min_preact


def compute_max_preact_random(preact, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    dim = tuple(set(range(preact.ndim)) - {1 - ax})
    with torch.no_grad():
        max_ = preact.amax(dim=dim, keepdim=True)
        argmaxes = (preact == max_).nonzero()
        chosen = []
        for i in range(max_.shape[1 - ax]):
            item_maxes = argmaxes[argmaxes[:, 1 - ax] == i, ax]
            idx = torch.randint(item_maxes.shape[0], (1,))
            chosen.append(item_maxes[idx])

    chosen = torch.tensor([chosen]) if ax == 0 else torch.tensor([chosen]).T
    max_preact = torch.gather(preact, ax, chosen).flatten()

    return max_preact


def compute_min_preact_random(preact, ax):
    # Get max across the desired dimension (0 -> unit, 1 -> point)
    dim = tuple(set(range(preact.ndim)) - {1 - ax})

    with torch.no_grad():
        min_ = preact.amin(dim=dim, keepdim=True)
        argmines = (preact == min_).nonzero()
        chosen = []
        for i in range(min_.shape[1 - ax]):
            col_mines = argmines[argmines[:, 1 - ax] == i, ax]
            idx = torch.randint(col_mines.shape[0], (1,))
            chosen.append(col_mines[idx])

    chosen = torch.tensor([chosen]) if ax == 0 else torch.tensor([chosen]).T
    min_preact = torch.gather(preact, ax, chosen).flatten()

    return min_preact


class JumpstartRegularization:
    def __init__(self, model=None, sign_balance=0.5, jr_mode='full', aggr='balanced', tie_breaking='single',
                 skip_batchnorm=True, skip_last=True, skip_downsample=False):
        """

        :param model: the model whose preactivations are captured and used to compute the jumpstart loss
        :param sign_balance: the balance between positive and negative constraints so they don't cancel out with zero init
        :param jr_mode: either full, unit, point, unit_positive, point_positive, positive
        :param aggr: legacy: mean, sum, norm. default: balanced -> mean of unit and point separately
        :param tie_breaking:
        :param skip_batchnorm:
        :param skip_last:
        """

        self.skip_downsample = skip_downsample
        self.aggr = aggr
        self.jr_mode = jr_mode
        if self.jr_mode == 'full':
            self.compute_constraint_loss = self.compute_full
        elif self.jr_mode == 'unit':
            self.compute_constraint_loss = self.compute_unit
        elif self.jr_mode == 'point':
            self.compute_constraint_loss = self.compute_point
        elif self.jr_mode == 'unit_positive':
            self.compute_constraint_loss = self.compute_unit_positive
        elif self.jr_mode == 'point_positive':
            self.compute_constraint_loss = self.compute_point_positive
        elif self.jr_mode == 'positive':
            self.compute_constraint_loss = self.compute_positive
        elif self.jr_mode == 'all_but_unit_positive':
            self.compute_constraint_loss = self.compute_all_but_unit_positive
        elif self.jr_mode == 'all_but_unit_negative':
            self.compute_constraint_loss = self.compute_all_but_unit_negative
        elif self.jr_mode == 'all_but_point_positive':
            self.compute_constraint_loss = self.compute_all_but_point_positive
        elif self.jr_mode == 'all_but_point_negative':
            self.compute_constraint_loss = self.compute_all_but_point_negative

        else:
            raise ValueError(f'Unknown constraint mode {self.jr_mode}')

        self.sign_balance = sign_balance
        self.tie_breaking = tie_breaking
        self.skip_last = skip_last
        self.skip_batchnorm = skip_batchnorm

        self._model = None
        self.positive_unit_losses = {}
        self.negative_unit_losses = {}
        self.positive_point_losses = {}
        self.negative_point_losses = {}
        self.hook_handlers = {}

        if self.tie_breaking == 'average':
            self.compute_max_preact = compute_max_preact_average
            self.compute_min_preact = compute_min_preact_average
        elif self.tie_breaking == 'single':
            self.compute_max_preact = compute_max_preact_single
            self.compute_min_preact = compute_min_preact_single
        elif self.tie_breaking == 'random':
            self.compute_max_preact = compute_max_preact_random
            self.compute_min_preact = compute_min_preact_random
        else:
            raise ValueError(f'Unknown backprop tie breaking mode {self.tie_breaking}')

        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        # Avoid setting the model twice
        if model is not self.model:
            self._model = model
            print(f'Setting Jumpstart loss hooks with {self.aggr} aggregation and {self.tie_breaking} backprop mode')
            self.positive_unit_losses.clear()
            self.negative_unit_losses.clear()
            self.positive_point_losses.clear()
            self.negative_point_losses.clear()
            names = []
            named_modules = self.model.named_modules()
            if self.skip_last:
                print('Skipping last layer')
                named_modules = list(named_modules)[:-1]
            if self.skip_batchnorm:
                print('Skipping Batchnorm layers')
            if self.skip_downsample:
                print('Skipping downsample layers')
            for name, module in named_modules:
                if hasattr(module, 'weight'):
                    is_bn = isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                    is_downsample = 'downsample' in name
                    if not (is_bn and self.skip_batchnorm) and not (is_downsample and self.skip_downsample):
                        if name in names:
                            raise ValueError(f'Repeated layer name {name}')
                        names.append(name)
                        # print(f'Registering layer {name} of type {type(module).__name__}')
                        handler = module.register_forward_hook(partial(self.hook, name=name))
                        self.hook_handlers[name] = handler

    def hook(self, module, input_, output, name=None):
        """
        It is called at each forward and computes the appropriate losses according to self.mode
        positive unit: Eq. 3 from the paper
        negative unit: Eq. 4 from the paper
        positive point: Eq. 5 from the paper
        positive point: Eq. 6 from the paper

        ax: 0 -> unit, 1 -> point
        modes: full, unit, point, unit_positive, point_positive, positive

        :param module:
        :param input_:
        :param output:
        :return:
        """
        if name is None:
            name = module
        self.positive_unit_losses[name] = self.compute_positive_constraint(output, 0)
        self.negative_unit_losses[name] = self.compute_negative_constraint(output, 0)
        self.positive_point_losses[name] = self.compute_positive_constraint(output, 1)
        self.negative_point_losses[name] = self.compute_negative_constraint(output, 1)

    def compute_positive_constraint(self, preact, ax):
        max_preact = self.compute_max_preact(preact, ax)
        return F.relu(1 - max_preact)

    def compute_negative_constraint(self, preact, ax):
        min_preact = self.compute_min_preact(preact, ax)
        return F.relu(1 + min_preact)

    def aggregate_unit_point_losses(self, unit, point):
        if self.aggr == 'balanced':
            loss = unit.mean() + point.mean()
        elif self.aggr == 'norm':
            loss = torch.cat((unit, point)).norm()
        elif self.aggr == 'mean':
            loss = torch.cat((unit, point)).mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def aggregate_unit_or_point_losses(self, unit_or_point):
        if self.aggr == 'balanced':
            loss = unit_or_point.mean()
        elif self.aggr == 'norm':
            loss = unit_or_point.norm()
        elif self.aggr == 'mean':
            loss = unit_or_point.mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_full(self):
        positive_unit = torch.cat(list(self.positive_unit_losses.values()))
        negative_unit = torch.cat(list(self.negative_unit_losses.values()))
        positive_point = torch.cat(list(self.positive_point_losses.values()))
        negative_point = torch.cat(list(self.negative_point_losses.values()))

        unit = self.sign_balance * positive_unit + (1 - self.sign_balance) * negative_unit
        point = self.sign_balance * positive_point + (1 - self.sign_balance) * negative_point

        if self.aggr == 'balanced':
            loss = unit.mean() + point.mean()
        elif self.aggr == 'norm':
            loss = torch.cat((unit, point)).norm()
        elif self.aggr == 'mean':
            loss = torch.cat((unit, point)).mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_unit(self):
        positive_unit = torch.cat(list(self.positive_unit_losses.values()))
        negative_unit = torch.cat(list(self.negative_unit_losses.values()))
        unit = self.sign_balance * positive_unit + (1 - self.sign_balance) * negative_unit
        if self.aggr == 'balanced':
            loss = unit.mean()
        elif self.aggr == 'norm':
            loss = unit.norm()
        elif self.aggr == 'mean':
            loss = unit.mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_point(self):
        positive_point = torch.cat(list(self.positive_point_losses.values()))
        negative_point = torch.cat(list(self.negative_point_losses.values()))
        point = self.sign_balance * positive_point + (1 - self.sign_balance) * negative_point
        if self.aggr == 'balanced':
            loss = point.mean()
        elif self.aggr == 'norm':
            loss = point.norm()
        elif self.aggr == 'mean':
            loss = point.mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_unit_positive(self):
        positive_unit = torch.cat(list(self.positive_unit_losses.values()))
        unit = self.sign_balance * positive_unit
        if self.aggr == 'balanced':
            loss = unit.mean()
        elif self.aggr == 'norm':
            loss = unit.norm()
        elif self.aggr == 'mean':
            loss = unit.mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_point_positive(self):
        positive_point = torch.cat(list(self.positive_point_losses.values()))
        point = self.sign_balance * positive_point
        if self.aggr == 'balanced':
            loss = point.mean()
        elif self.aggr == 'norm':
            loss = point.norm()
        elif self.aggr == 'mean':
            loss = point.mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_positive(self):
        positive_unit = torch.cat(list(self.positive_unit_losses.values()))
        positive_point = torch.cat(list(self.positive_point_losses.values()))
        unit = self.sign_balance * positive_unit
        point = self.sign_balance * positive_point

        if self.aggr == 'balanced':
            loss = unit.mean() + point.mean()
        elif self.aggr == 'norm':
            loss = torch.cat((unit, point)).norm()
        elif self.aggr == 'mean':
            loss = torch.cat((unit, point)).mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_all_but_unit_positive(self):
        negative_unit = torch.cat(list(self.negative_unit_losses.values()))
        positive_point = torch.cat(list(self.positive_point_losses.values()))
        negative_point = torch.cat(list(self.negative_point_losses.values()))

        unit = (1 - self.sign_balance) * negative_unit
        point = self.sign_balance * positive_point + (1 - self.sign_balance) * negative_point

        if self.aggr == 'balanced':
            loss = unit.mean() + point.mean()
        elif self.aggr == 'norm':
            loss = torch.cat((unit, point)).norm()
        elif self.aggr == 'mean':
            loss = torch.cat((unit, point)).mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_all_but_unit_negative(self):
        positive_unit = torch.cat(list(self.positive_unit_losses.values()))
        positive_point = torch.cat(list(self.positive_point_losses.values()))
        negative_point = torch.cat(list(self.negative_point_losses.values()))

        unit = self.sign_balance * positive_unit
        point = self.sign_balance * positive_point + (1 - self.sign_balance) * negative_point

        if self.aggr == 'balanced':
            loss = unit.mean() + point.mean()
        elif self.aggr == 'norm':
            loss = torch.cat((unit, point)).norm()
        elif self.aggr == 'mean':
            loss = torch.cat((unit, point)).mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_all_but_point_positive(self):
        positive_unit = torch.cat(list(self.positive_unit_losses.values()))
        negative_unit = torch.cat(list(self.negative_unit_losses.values()))
        negative_point = torch.cat(list(self.negative_point_losses.values()))

        unit = self.sign_balance * positive_unit + (1 - self.sign_balance) * negative_unit
        point = (1 - self.sign_balance) * negative_point

        if self.aggr == 'balanced':
            loss = unit.mean() + point.mean()
        elif self.aggr == 'norm':
            loss = torch.cat((unit, point)).norm()
        elif self.aggr == 'mean':
            loss = torch.cat((unit, point)).mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    def compute_all_but_point_negative(self):
        positive_unit = torch.cat(list(self.positive_unit_losses.values()))
        negative_unit = torch.cat(list(self.negative_unit_losses.values()))
        positive_point = torch.cat(list(self.positive_point_losses.values()))

        unit = self.sign_balance * positive_unit + (1 - self.sign_balance) * negative_unit
        point = self.sign_balance * positive_point

        if self.aggr == 'balanced':
            loss = unit.mean() + point.mean()
        elif self.aggr == 'norm':
            loss = torch.cat((unit, point)).norm()
        elif self.aggr == 'mean':
            loss = torch.cat((unit, point)).mean()
        else:
            raise ValueError(f'Unknown aggr {self.aggr}')
        return loss

    @property
    def loss(self):
        return self.compute_constraint_loss()

    def __repr__(self):
        return f"JumpstartRegularization(model={self.model}, {len(self.hook_handlers)} layers," \
               f" sign_balance={self.sign_balance}, " \
               f"jr_mode='{self.jr_mode}', aggr='{self.aggr}', tie_breaking='{self.tie_breaking}', " \
               f"skip_batchnorm={self.skip_batchnorm}, skip_last={self.skip_last}, " \
               f"skip_downsample={self.skip_downsample})"