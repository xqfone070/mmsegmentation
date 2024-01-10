import numpy as np
from typing import Iterator
import torch
import logging
import math
from mmengine.dataset.sampler import InfiniteSampler
from mmseg.registry import DATA_SAMPLERS
from mmengine.logging import print_log


@DATA_SAMPLERS.register_module()
class ClassBalanceInfiniteSampler(InfiniteSampler):
    def __init__(self,
                 balance_classes: list,
                 ratio: float,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

        print_log('ClassBalanceInfiniteSampler start to get balanced_indices', logger='current', level=logging.WARNING)
        image_flags = self.get_image_flags(balance_classes)

        self.balanced_indices = self.balance(image_flags, ratio)
        print_log('ClassBalanceInfiniteSampler finished to get balanced_indices', logger='current', level=logging.WARNING)

    def get_image_flags(self, balance_classes):
        image_flags = []
        for i in range(len(self.dataset)):
            cat_ids = self.dataset.get_cat_ids(i)
            flag = False
            for cat in cat_ids:
                if cat in balance_classes:
                    flag = True

            image_flags.append(flag)

        return image_flags

    def balance(self, image_flags, target_ratio):
        total_num = len(image_flags)
        balance_num = np.count_nonzero(image_flags)
        other_num = len(image_flags) - balance_num

        cur_ratio = balance_num / total_num
        if cur_ratio > target_ratio:
            target_balance_num = balance_num
            target_total_num = math.ceil(target_balance_num / target_ratio)
            target_other_num = target_total_num - target_balance_num
        else:
            target_other_num = other_num
            target_total_num = math.ceil(target_other_num / (1 - target_ratio))
            target_balance_num = target_total_num - target_other_num

        balance_indices = [i for i in range(total_num) if image_flags[i]]
        other_indices = [i for i in range(total_num) if not image_flags[i]]

        target_balance_indices = self.choice_samples(balance_indices, target_balance_num)
        target_other_indices = self.choice_samples(other_indices, target_other_num)

        result = []
        result.extend(target_balance_indices)
        result.extend(target_other_indices)
        return result

    @staticmethod
    def choice_samples(indices, num):
        result = []
        total_num = len(indices)
        while num >= total_num:
            result.extend(indices)
            num -= total_num

        if num > 0:
            step = int(total_num / num)
            lst = list(range(0, num * step, step))
            result.extend(lst)

        return result

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                np.random.shuffle(self.balanced_indices)

            yield from self.balanced_indices
