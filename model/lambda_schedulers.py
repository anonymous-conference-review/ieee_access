import math


class CosineAnnealingLambda:
    def __init__(self, base_lambda, T_max, lambda_min=0, verbose=False):
        self.T_max = T_max
        self.base_lambda = base_lambda
        self.lambda_min = lambda_min
        self.verbose = verbose
        self.model = None

    @property
    def last_epoch(self):
        return self.model.current_epoch

    def get_lambda(self):
        return self.lambda_min + (self.base_lambda - self.lambda_min) * \
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2

