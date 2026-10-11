import contextlib 
import math 
import os 
from collections.abc import Callable, Iterable 
from typing import cast 


import torch 
import torch.distributed._functional_collectives as funcol 
import torch.distributed.distributed_c10d as c10d 
from loguru import logger 
from torch import distributed as dist 
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor 
from torch.distributed.tensor.experimental import context_parallel 
from torch.distributed.tensor.experimental._attention import context_parallel 

from torch.distributed.tensor.experimental._attention import (
    _cp_options, 
    set_rotate_method
)


from torch.distributed.tensor.parallel import loss_parallel
from torchfeather.config import TORCH_DTYPE_MAP
from torchfeather.distributed.parallel_dims import ParallelDims 


def get_train_context(
    enable_loss_parallel: bool, 
    enable_compiled_autograd: bool
) -> Callable[..., contextlib.AbstractContextManager]:
    @contextlib.contextmanager 
    def context(cp_context: contextlib.AbstractContextManager | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(loss_parallel())
            
            if enable_compile_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )
            
            if cp_context:
                stack.enter_context(cp_context)
            
            yield 
    
    return context 