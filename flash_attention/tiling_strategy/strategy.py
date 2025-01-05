# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
from abc import ABCMeta
from abc import abstractmethod
from collections import namedtuple

TilingPara = namedtuple("TilingPara", "Br last_Br Bc last_Bc Tr Tc")


class TilingStrategy(metaclass=ABCMeta):
    """Tiling strategy interface. All implementations should be defined in this module,
    otherwise, the UT will fail.
    """

    _strategies = {}

    def __init__(self, Nq, N, head_dim) -> None:
        super().__init__()
        self.Nq = Nq
        self.N = N
        self.Br = None
        self.last_Br = None
        self.Bc = None
        self.last_Bc = None
        self.Tr = None
        self.Tc = None
        self.d = head_dim

    def __init_subclass__(cls, **kwargs):
        TilingStrategy._strategies[cls.strategy_name()] = cls

    @classmethod
    @abstractmethod
    def strategy_name(cls):
        raise NotImplemented

    @classmethod
    def from_strategy_name(cls, stgy_name: str):
        stgy_clz = TilingStrategy._strategies.get(stgy_name)
        if stgy_clz is None:
            raise Exception(f"Strategy:{stgy_name} not supported")

        return stgy_clz

    @abstractmethod
    def tiling(self) -> TilingPara:
        raise NotImplemented

    def gen_tiling_para(self) -> TilingPara:
        return TilingPara(self.Br, self.last_Br, self.Bc, self.last_Bc, self.Tr, self.Tc)

