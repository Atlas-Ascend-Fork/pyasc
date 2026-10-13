.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Kernel API
==========

.. currentmodule:: asc.experimental.asctile

.. autosummary::
    :toctree: generated
    :nosignatures:

    LocalTensor
    GlobalTensor
    TensorLocation


Programming model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    global_tensor
    block_idx
    block_num
    sub_block_idx
    sub_block_num
    cv_strategy


Iterators
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    range
    static_range


Creation operations
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    cast
    concat
    full
    full_like
    zeros
    zeros_acc
    zeros_like


Memory operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    copy
    copy_in
    copy_out
    gather
    scatter


Arithmetic operations
---------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    bitwise_and
    bitwise_not
    bitwise_or
    bitwise_xor
    div
    equal
    greater
    greater_equal
    left_shift
    less
    less_equal
    maximum
    minimum
    mul
    negative
    not_equal
    right_shift
    sub


Math operations
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    ceil
    cos
    cosh
    erf
    exp
    exp2
    floor
    layer_norm
    log
    log2
    pow
    relu
    rms_norm
    rsqrt
    sin
    sinh
    softmax
    sqrt
    tan
    tanh


Matrix multiplication operations
--------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    matmul
    matmul_acc


Indexing operations
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    mask
    where


Reduction operations
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    reduce_min
    reduce_max
    reduce_sum
    reduce_prod


Shape manipulation operations
-----------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast_shapes
    broadcast_tensors
    broadcast_to
    expand_dims
    ravel
    reshape
    squeeze
    transpose


Atomic operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_max
    atomic_min


Debug operations
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    device_assert
    device_print
    static_assert
    static_print


Low-level tweaking operations
-----------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    empty
    empty_like
    inline
    inline_vf
    from_asc
    to_asc


Utility functions
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    ceildiv
