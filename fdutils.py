# _*_ coding: utf-8 _*_
# @Time     : 2022/12/1 16:16
# @Author   : Yu Loong
# @File     : fdutils.py

from typing import Dict, List, Any, Union, AnyStr, KeysView, Optional


# c1
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
