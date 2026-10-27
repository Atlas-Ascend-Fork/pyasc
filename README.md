# pyasc

## 🔥Latest News

- [2025/11] pyasc项目首次上线。

## 🚀概述

pyasc是一种用于编写高效自定义算子的编程语言，原生支持python标准规范。基于pyasc编写的算子程序，通过编译器编译和运行时调度，运行在昇腾AI处理器上。  
pyasc编程接口与Ascend C类库接口一一对应，旨在提供与Ascend C接口相同的编程能力，目前正逐步开放中。有关pyasc编程接口的支持范围和约束，请参考[Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi)。对于编程所需的抽象硬件架构和编程模型的相关知识，请参考《[Ascend C算子开发](https://www.hiascend.com/document/redirect/CannCommunityOpdevAscendC)》。本项目支持的AI处理器包括：Atlas A3训练/推理产品、Atlas A2训练/推理产品。

## 🔍目录结构
关键目录如下：
```shell
├── bin                 # 工具文件
├── docs                # 说明文档
│   ├── figures         # 文档图片
│   └── python-api      # API接口文档
├── examples            # 供用户参考的算子开发样例
├── include             # 后端头文件和td文件
│   └── ascir             ## ascir头文件和td文件
├── lib                 # 后端源文件
│   ├── Dialect           ## mlir方言定义源文件
│   ├── TableGen          ## tablegen扩展代码文件
│   └── Target            ## mlir目标代码转换源文件
├── scripts             # 相关脚本目录
├── python              # python前端代码
│   ├── asc               ## 用户可见的python包，对外发布的wheel包中以此目录为主，其他代码则按需打包
│   ├── src               ## pybind相关代码，cpp格式
│   ├── test              ## python格式的测试用例集
└── test                # 后端的测试用例集
    ├── Dialect           ## mlir方言定义模块测试用例
    ├── Target            ## mlir目标代码转换模块测试用例
    └── tools             ## 后端工具相关测试用例
```


## ⚡️快速入门

若您希望快速体验pyasc的使用过程，请访问如下文档获取简易教程。

- [构建](https://gitcode.com/cann/pyasc/blob/master/docs/quick_start.md)：介绍搭建环境、编译执行、本地验证。
- [样例执行](./examples/README.md)：介绍如何端到端执行样例代码。

## 📖学习教程

若您希望深入体验项目或参与项目贡献，请访问如下文档获取详细教程。

- [API列表](https://gitcode.com/cann/pyasc/blob/master/docs/python-api/index.md)：介绍项目提供的Ascend C Python API信息，方便快速查询。
- [调试调优](https://gitcode.com/cann/pyasc/blob/master/docs/op_debug_prof.md)：介绍常见的算子调试和调优方法。
- [pyasc模块与架构](https://gitcode.com/cann/pyasc/blob/master/docs/architecture_introduction.md)：介绍pyasc的模块与架构。
- [Ascend C Python编程接口开发指南](https://gitcode.com/cann/pyasc/blob/master/docs/developer_guide.md)：介绍如何开发Ascend C API的Python编程接口。
- [pyasc的python语法支持情况说明](https://gitcode.com/cann/pyasc/blob/master/docs/python_syntax_support.md)：介绍pyasc项目支持和不支持的python语法。
- [API文档自动生成工具使用指南](https://gitcode.com/cann/pyasc/blob/master/docs/API_docstring_generation_tool_guide.md)：介绍本项目接口文档的生成方法。
- [PyAsc算子开发指南](https://gitcode.com/cann/pyasc/blob/master/docs/pyasc_op_develop_guide.md)：介绍基于pyasc的Ascend C算子开发基本流程。

## 💻 软硬件配套说明

pyasc所需的软硬件环境依赖如下：

- 昇腾产品：[Atlas A2训练/推理产品](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html), [Atlas A3训练/推理产品](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
- CPU架构：`aarch64`/`x86_64`
- 系统：支持的Linux（进行[兼容性查询](https://www.hiascend.com/hardware/compatibility)）
- 软件依赖：
  - `python` 3.9-3.12

不同pyasc发行版可支持的硬件平台及所需的[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)版本如下表：

<table style="width: 75%; margin: 0 auto;">
  <colgroup>
    <col style="width: 25%">
    <col style="width: 22%">
    <col style="width: 22%">
  </colgroup>
  <thead>
      <tr>
          <th>pyasc社区版本</th>
          <th>支持CANN包版本</th>
          <th>支持昇腾产品</th>
      </tr>
  </thead>
  <tbody style="text-align: center">
      <tr>
          <td>v1.1.0、v1.1.1</td>
          <td>社区版<a href="https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha001">8.5.0.alpha001</a>及以上</td>
          <td><a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A2训练/推理产品</a> <br>
          <a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A3训练/推理产品</a></td>
      </tr>
      <tr>
          <td>v1.0.0</td>
          <td>社区版<a href="https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha001">8.5.0.alpha001</a>、<a href="https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha002">8.5.0.alpha002</a></td>
          <td><a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A2训练/推理产品</a> <br>
          <a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A3训练/推理产品</a></td>
      </tr>
  </tbody>
</table>


## 👥 合作贡献者

- 哈尔滨工业大学苏统华老师团队、王甜甜老师团队

## 📝相关信息

- [贡献指南](https://gitcode.com/cann/pyasc/blob/master/CONTRIBUTING.md)
- [安全声明](https://gitcode.com/cann/pyasc/blob/master/SECURITY.md)
- [许可证](https://gitcode.com/cann/pyasc/blob/master/LICENSE)