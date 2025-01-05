# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
from tbe import tik

BLOCK_NUM = 16
FP16 = "float16"
INT8 = "int8"
INT32 = "int32"
FP32 = "float32"
REPEAT_SZ = 128
BLK_STRIDE = 1
REPEAT_STRIDE = 8
TRANS_CUBE_TGT = 8
FP16_MIN_VAL = -65504.0
MASK_FILL_VALUE = -10000.0
GM = tik.scope_gm
L1 = tik.scope_cbuf
L1OUT = tik.scope_cbuf_out
UB = tik.scope_ubuf
L0A = tik.scope_ca
L0B = tik.scope_cb
L0C = tik.scope_cc
DTYPE_SIZE = {
    "int8": 1,
    "float16": 2,
    "int16": 2,
    "float32": 4,
}
