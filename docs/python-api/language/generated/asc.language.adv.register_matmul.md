# asc.language.adv.register_matmul

### asc.language.adv.register_matmul(pipe: [TPipe](../fwk.md#asc.language.fwk.TPipe), workspace: GlobalAddress, matmul: [Matmul](../adv.md#asc.language.adv.Matmul), tiling: TCubeTiling | None = None) → None

主要用于初始化Matmul对象。

**对应的Ascend C函数原型**

```c++
REGIST_MATMUL_OBJ(tpipe, workspace, ...)
```

**参数说明**

- pipe: TPipe对象。
- workspace: 系统workspace指针。
- matmul: 待初始化的Matmul对象。
- tiling: 可选的Matmul Tiling参数；未传入时，通过Matmul.init单独设置。

**约束说明**

- 在分离模式中，本接口必须在init_buffer接口前调用。
- 在程序中，最多支持定义4个Matmul对象。
- 当代码中只有一个Matmul对象时，本接口可以不传入tiling参数，通过init接口单独传入tiling参数。

**调用示例**

```python
pipe = asc.TPipe()
# 推荐：初始化单个matmul对象，传入tiling参数
asc.adv.register_matmul(pipe, workspace, mm, tiling)
# 初始化单个matmul对象，未传入tiling参数。注意，该场景下需要使用Init接口单独传入tiling参数。这种方式将matmul对象的初始化和tiling的设置分离，比如，Tiling可变的场景，可通过这种方式多次对Tiling进行重新设置
asc.adv.register_matmul(pipe, workspace, mm)
mm.init(tiling)
```
