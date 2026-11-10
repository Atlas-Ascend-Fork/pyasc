# asc.lib.host.MatmulApiTiling.set_fix_split

#### MatmulApiTiling.set_fix_split(self: libhost.MatmulApiTilingBase, base_m_in: SupportsInt = -1, base_n_in: SupportsInt = -1, base_k_in: SupportsInt = -1) → int

设置固定的baseM、baseN、baseK，单位为元素个数。

**对应的Ascend C函数原型**

```c++
int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1)
```

**参数说明**

- base_m_in：固定的baseM，默认值为-1，表示由Tiling函数计算。
- base_n_in：固定的baseN，默认值为-1，表示由Tiling函数计算。
- base_k_in：baseK配置值，默认值为-1，具体取值限制遵循配套CANN版本的SetFixSplit接口约束。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**约束说明**

- base_m\*base_n个输出元素所占的存储空间大小不能超过L0C Buffer大小，即base_m \* base_n \* sizeof(C_TYPE) <= L0CSize。
- base_m需要小于等于single_m按16个元素向上对齐后的值（如ceil(single_m/16)\*16），base_n需要小于等于single_n以C0_size个元素向上对齐的值，其中single_m为单核内M轴长度，singleN为单核内N轴长度，half/bfloat16_t数据类型的C0_size为16，float数据类型的C0_size为8，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。例如single_m=12，则base_m需要小于等于16，同时base_m需要满足分形对齐的要求，所以base_m只能取16；如果base_m取其他超过16的值，获取Tiling将失败。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MatmulApiTiling(ascendc_platform)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(1024, 1024, 1024)
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_fix_split(16, 16, -1)  # 设置固定的base_m, bakse_n
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
