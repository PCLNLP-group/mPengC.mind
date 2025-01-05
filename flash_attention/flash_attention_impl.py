from __future__ import absolute_import

import os
import sys

from mindspore.ops import DataType
from mindspore.ops import TBERegOp
from mindspore.ops import op_info_register

sys.path.append(os.path.dirname(__file__))

from flash_attention_fwd import flash_attention
from flash_attention_bwd import flash_attention_grad

kernel_name = "flash_attention"

cus_flash_atten_op_info = TBERegOp("FlashAttentionPrimitive") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("flash_attention.so") \
    .compute_cost(10) \
    .kernel_name(kernel_name) \
    .attr("prev_block_num", "required", "int", "all", "65536") \
    .attr("next_block_num", "required", "int", "all", "65536") \
    .attr("high_precision", "required", "bool", "all", "false") \
    .attr("tiling_stgy_name", "required", "str", "all", "xunfei") \
    .input(0, "q", False, "required", "all") \
    .input(1, "k", False, "required", "all") \
    .input(2, "v", False, "required", "all") \
    .input(3, "attn_mask", False, "optional", "all") \
    .input(4, "dropout_mask", False, "optional", "all") \
    .input(5, "alibi_mask", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "l", False, "required", "all") \
    .output(2, "m", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_Default,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_Default,
                  DataType.F16_Default) \
    .dtype_format(DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_Default,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F32_Default,
                  DataType.F16_Default) \
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(cus_flash_atten_op_info)
def flash_attention_impl(query, key, value, attn_mask, dropout_mask, alibi_mask, y, l,
                         m, prev_block_num, next_block_num, high_precision, tiling_stgy_name):
    flash_attention(query, key, value, attn_mask, dropout_mask, alibi_mask,
                    y, l, m, prev_block_num, next_block_num,
                    high_precision=high_precision,
                    kernel_name=kernel_name,
                    tiling_stgy_name=tiling_stgy_name)


kernel_name = "flash_attention_grad"

cus_flash_atten_grad_op_info = TBERegOp("FlashAttentionGradPrimitive") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("flash_attention_grad.so") \
    .compute_cost(10) \
    .kernel_name(kernel_name) \
    .attr("prev_block_num", "required", "int", "all", "65536") \
    .attr("next_block_num", "required", "int", "all", "65536") \
    .attr("high_precision", "required", "bool", "all", "false") \
    .attr("tiling_stgy_name", "required", "str", "all", "xunfei") \
    .input(0, "q", False, "required", "all") \
    .input(1, "k", False, "required", "all") \
    .input(2, "v", False, "required", "all") \
    .input(3, "o", False, "required", "all") \
    .input(4, "do", False, "required", "all") \
    .input(5, "l", False, "required", "all") \
    .input(6, "m", False, "required", "all") \
    .input(7, "attn_mask", False, "optional", "all") \
    .input(8, "dropout_mask", False, "optional", "all") \
    .input(9, "alibi_mask", False, "optional", "all") \
    .output(0, "dq", False, "required", "all") \
    .output(1, "dk", False, "required", "all") \
    .output(2, "dv", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_FracNZ,
                  DataType.F16_Default,
                  DataType.F16_FracNZ,
                  DataType.F32_FracNZ,
                  DataType.F32_FracNZ,
                  DataType.F32_FracNZ,) \
    .dtype_format(DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F16_FracNZ,
                  DataType.F32_Default,
                  DataType.F16_Default,
                  DataType.F16_FracNZ,
                  DataType.F16_Default,
                  DataType.F16_FracNZ,
                  DataType.F32_FracNZ,
                  DataType.F32_FracNZ,
                  DataType.F32_FracNZ) \
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(cus_flash_atten_grad_op_info)
def flash_attention_grad_impl(q, k, v, o, dout, l, m, attn_mask, dropout_mask, alibi_mask,
                              dq, dk, dv, prev_block_num, next_block_num,
                              high_precision, tiling_stgy_name="xunfei"):
    flash_attention_grad(q, k, v, o, dout, l, m, attn_mask, dropout_mask, alibi_mask,
                         dq, dk, dv, prev_block_num, next_block_num,
                         high_precision=high_precision,
                         kernel_name=kernel_name,
                         tiling_stgy_name=tiling_stgy_name)
