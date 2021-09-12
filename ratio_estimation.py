import hypothesis as h
import torch
import numpy as np

from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import TensorDataset



class RatioEstimator(BaseRatioEstimator):

    def __init__(self, denominator):
        random_variables = {
            "configs": (1,),
            "inputs": (1,),
            "outputs": (1,)}
        Class = build_ratio_estimator("mlp", random_variables, denominator=denominator)
        activation = torch.nn.SELU
        trunk = [128] * 3
        r = Class(activation=activation, trunk=trunk)
        super(RatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, configs, inputs, outputs, **kwargs):
        configs = configs - 45
        configs = configs / 5
        return self._r.log_ratio(configs=configs, inputs=inputs, outputs=outputs)


class RatioEstimatorTypeI(RatioEstimator):

    def __init__(self):
        denominator = "inputs|outputs,configs"
        super(RatioEstimatorTypeI, self).__init__(denominator)


class RatioEstimatorTypeII(RatioEstimator):

    def __init__(self):
        denominator = "inputs,outputs|configs"
        super(RatioEstimatorTypeII, self).__init__(denominator)


class DatasetTrain(NamedDataset):

    def __init__(self):
        inputs = np.load("data/train/inputs.npy")
        configs = np.load("data/train/configs.npy")
        outputs = np.load("data/train/outputs.npy")
        inputs = TensorDataset(torch.from_numpy(inputs))
        configs = TensorDataset(torch.from_numpy(configs))
        outputs = TensorDataset(torch.from_numpy(outputs))
        super(DatasetTrain, self).__init__(
            configs=configs,
            inputs=inputs,
            outputs=outputs)


class DatasetTest(NamedDataset):

    def __init__(self):
        inputs = np.load("data/test/inputs.npy")
        configs = np.load("data/test/configs.npy")
        outputs = np.load("data/test/outputs.npy")
        inputs = TensorDataset(torch.from_numpy(inputs))
        configs = TensorDataset(torch.from_numpy(configs))
        outputs = TensorDataset(torch.from_numpy(outputs))
        super(DatasetTest, self).__init__(
            configs=configs,
            inputs=inputs,
            outputs=outputs)
