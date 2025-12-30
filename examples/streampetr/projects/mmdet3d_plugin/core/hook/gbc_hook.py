# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook

def parse_r(r: int, n: int, layers: int):
    if r <= 0 or n <= 0:
        rs = [0 for _ in range(layers)]
    else:
        r_per_layer = r // n
        if r_per_layer % 2 == 1:
            # 保证 rk 是偶数
            r_per_layer -= 1
        rs = [r_per_layer for _ in range(n - 1)]
        rs.append(r - sum(rs))
        for _ in range(layers - n):
            rs.append(0)
    return rs

@HOOKS.register_module()
class GBCHook(Hook):
    """Set runner's epoch information to the model."""

    def __init__(
            self,
            warmup_iters:int,
            heating_iters:int,
            r:int,
            n:int,
            k:int,
            layers:int,
            log_interval:int=1000
        ):
        self.warmup_iters = max(warmup_iters, 1)
        self.heating_iters = max(heating_iters, 0)
        self.r = r
        self.n = n
        self.k = k
        self.layers = layers
        self.log_interval = log_interval

    def before_train_iter(self, runner):
        i = runner.iter + 1
        if i == 1:
            runner.model.module.pts_bbox_head.transformer.decoder._tgtg_info["r"] = parse_r(0, self.n, self.layers)
            print("[GBC] Start warming up, set r = 0.")

        if self.warmup_iters < i <= self.warmup_iters + self.heating_iters:
            r = int((i - self.warmup_iters) / self.heating_iters * self.r)
            runner.model.module.pts_bbox_head.transformer.decoder._tgtg_info["r"] = parse_r(r, self.n, self.layers)
            if (i - self.warmup_iters) % self.log_interval == 0:
                print(f"[GBC] Heating, set r = {r}.")

        if i == self.warmup_iters + self.heating_iters + 1:
            r = self.r
            runner.model.module.pts_bbox_head.transformer.decoder._tgtg_info["r"] = parse_r(r, self.n, self.layers)
            print(f"[GBC] Heating done, set r = {r}.")
