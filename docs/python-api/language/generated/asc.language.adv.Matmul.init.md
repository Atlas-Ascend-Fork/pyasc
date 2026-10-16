# asc.language.adv.Matmul.init

#### Matmul.init(tiling: TCubeTiling) → None

灵活的自定义Matmul模板参数配置。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr)
```

**参数说明**

- tiling: Matmul Tiling参数。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm)
mm.init(tiling)
```
