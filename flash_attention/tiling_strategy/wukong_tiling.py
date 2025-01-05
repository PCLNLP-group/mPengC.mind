# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
from tiling_strategy.strategy import TilingPara
from tiling_strategy.strategy import TilingStrategy


class WukongTiling(TilingStrategy):
    """A tiling strategy implementation for wukonghuahua model shape"""

    @classmethod
    def strategy_name(cls):
        return "wukong"

    def tiling(self) -> TilingPara:
        """
        反向的空间分布待详细分析
        N = (4096, 1024, 256, 64) 或 77
        Nq = (4096, 1024, 256, 64)
        d = dv = (40, 80, 160， 160)
        """
        if self.N <= 77:  # [77, 64]
            # cross-attention or self-attention of (64, 64, 160)
            self.Bc = self.N
            self.Tc = self.N // self.Bc
            if self.d <= 80:  # [40, 80]
                # 内存瓶颈为在ub中对P*V结果[Br, dv]进行cast
                # ub: 512 * 80 * 6 // 1024 = 240KB
                self.Br = min(self.Nq, 64)
                self.Tr = self.Nq // self.Br
            else:
                # ub: dv = 160， 256 * 160 * 6 // 1024 = 240KB
                self.Br = min(self.Nq, 64)
                self.Tr = self.Nq // self.Br
        else:
            # self-attention
            if self.N == 256:
                self.Bc = 64
                self.Tc = 1
                # 内存瓶颈为在ub中对Q*K的结果[Br, Bc]进行cast
                # ub: 128 * 256 * 6 // 1024 = 192KB
                self.Br = 64
                self.Tr = self.Nq // self.Br
            else:
                self.Bc = 64
                self.Tc = self.N // self.Bc
                # ub: 64 * 512 * 6 // 1024 = 192KB
                self.Br = 64
                self.Tr = self.Nq // self.Br

        self.last_Br = self.Br
        self.last_Bc = self.Bc

        return self.gen_tiling_para()