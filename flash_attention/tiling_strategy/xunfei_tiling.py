# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
from tiling_strategy.strategy import TilingPara
from tiling_strategy.strategy import TilingStrategy


class XunfeiTiling(TilingStrategy):
    """A tiling strategy implementation for xunfei ipt model shape"""

    @classmethod
    def strategy_name(cls):
        return "xunfei"

    def tiling(self) -> TilingPara:
        self.Br = min(128, self.Nq)
        self.Bc = min(128, self.N)

        self.Tr = self.Nq // self.Br
        self.Tc = self.N // self.Bc

        if self.Nq % self.Br != 0:
            self.last_Br = self.Nq - self.Tr * self.Br
            self.Tr += 1
        else:
            self.last_Br = self.Br
        if self.N % self.Bc != 0:
            self.last_Bc = self.N - self.Tc * self.Bc
            self.Tc += 1
        else:
            self.last_Bc = self.Bc

        return self.gen_tiling_para()