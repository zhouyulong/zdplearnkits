# _*_ coding: utf-8 _*_
# @Time     : 2022/12/1 16:16
# @Author   : Yu Loong
# @File     : fdutils.py

from typing import Dict, List, Any, Tuple, AnyStr, KeysView, Optional
import torch
from torch import nn
import time


# 1
class Accumulator():
    """
    collecting metrics in experiment
    """
    def __init__(self, names: List[Any]):
        self.accumulator = {}
        if not isinstance(names, list):
            raise Exception(f'type error, expected list but got {type(names)}')
        for name in names:
            self.accumulator[name] = list()

    def __getitem__(self, item) -> List[Any]:
        if item not in self.accumulator.keys():
            raise Exception(f'key error, {item} is not in accumulator')
        return self.accumulator[item]

    def add(self, name: AnyStr, val: Any):
        self.accumulator[name].append(val)

    def add_name(self, name: AnyStr):
        if name in self.accumulator.keys():
            raise Exception(f'{name} is  already in accumulator.keys')
        self.accumulator[name] = list()

    def gets(self) -> Dict[AnyStr, Any]:
        return self.accumulator

    def get_item(self, name: AnyStr) -> List[Any]:
        if name not in self.accumulator.keys():
            raise Exception(f'key error, {name} is not in accumulator')
        return self.accumulator[name]

    def clear(self):
        self.accumulator.clear()

    def get_names(self) -> KeysView:
        return self.accumulator.keys()


# 2
def display_model_layers(module: nn.Module, data_size: Tuple[int, int, int, int]) -> None:
    tx = torch.normal(0, 1, size=data_size)
    accumulator = Accumulator(['name', 'input_shape', 'output_shape', 'layer'])

    for layer in module:
        in_shape = tx.shape
        tx = layer(tx)
        out_shape = tx.shape
        accumulator.add('name', layer.__class__.__name__)
        accumulator.add('input_shape', in_shape)
        accumulator.add('output_shape', out_shape)
        accumulator.add('layer', layer.__str__())
    max_length_name = max(accumulator['name'], key=len).__len__()
    max_length_layer = max(accumulator['layer'], key=len).__len__()
    max_length = len(module)
    print('\nindex'.center(5), 'name'.center(max_length_name), '\t', 'layer'.center(max_length_layer), '\t', 'input shape'.center(40), '\toutput shape'.center(50), '\n')
    for index in range(max_length):
        print(str(index+1).center(5),accumulator['name'][index].ljust(max_length_name), '\t', accumulator['layer'][index].ljust(max_length_layer),
              '\t', f'input shape\t:{accumulator["input_shape"][index]}'.ljust(40), f'\toutput shape:\t{accumulator["output_shape"][index]}'.ljust(50))


# 3 timekeeping
class Timer:
    def __init__(self) -> None:
        self.__times = list()
        self.start()

    def start(self):
        self.start_time = time.time()

    def cut(sell) -> float:
        """
        return the time used for epoch
        """
        self.cut_time = time.time()-self.start_time()
        self.__times.append(self.cut_time)
        return self.cut_time

    def sum(self) -> float:
        return sum(self.__times)

    def avg(self) -> float:
        return torch.tensor(self.__times, dtype=torch.float).mean().item()

    # display cumsum for times
    def accumulate(self) -> float:
        return torch.tensor(self.__times, dtype=torch.float).cumsum(dim-0).tolist()
