# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参阅 [cann-community](https://gitcode.com/cann/community) 了解行为准则，完成 CLA 协议签署，并熟悉源码仓库的贡献流程。除本指南所列贡献流程外，您还可以通过社区的 [基础贡献](https://gitcode.com/cann/community#基础贡献)（参与社区会议、邮件讨论、提交 Issue/PR 等）或 [进阶贡献](https://gitcode.com/cann/community#进阶贡献)（新建 SIG / 成为核心贡献者、组织会议、新建仓库及 CI 配置等）深度参与 CANN 社区建设。

---

## 📋 一、贡献特性分类

| 分类 | 说明 | 示例 |
|------|------|------|
| **L1（轻量特性）** | 简单的新增需求 / Bug 修复 / 特性优化 / 文档纠错等（< 200 行） | 新增 Ascend C Python API 接口、已支持特性场景 Bug 修复、性能优化、API 文档描述错误纠正等 |
| **L2（大特性）** | 大的功能特性 / 性能增强特性等 | 新增当前代码框架未支持的 PASS 优化大颗粒特性等 |
| **L3（架构变更）** | 核心接口变更 / 重大重构等 | 对外接口目录调整、对外核心流程接口变更、端到端编译运行流程变更等 |

---

## 🔧 二、贡献流程说明

> 💡 **提示**：按照贡献特性分类，遵循对应的贡献流程。

### L1 特性贡献流程

#### 📌 Step1：创建 Issue

根据类型，创建对应的 Issue：

| 类型 | 创建方式 | Issue 分配 |
|------|----------|------------|
| **新需求** | 新建 `Requirement\|需求建议` 类 Issue，阐明新增特性的设计方案 | 在评论框中输入 `/assign` 或 `/assign @yourself` |
| **Bug 修复** | 新建 `Bug-Report\|缺陷反馈` 类 Issue 描述 Bug | 在评论框中输入 `/assign` 或 `/assign @yourself` |
| **特性优化** | 新建 `Requirement\|需求建议` 类 Issue 说明优化点，提供设计方案 | 在评论框中输入 `/assign` 或 `/assign @yourself` |
| **文档纠错** | 新建 `Documentation\|文档反馈` 类 Issue 指出文档问题 | 在评论框中输入 `/assign` 或 `/assign @yourself` |

> 📝 **新需求 Issue 一般需包含以下内容**
> - 背景信息
> - 价值 / 作用
> - 设计方案

具体 Issue 创建和处理方式可参考 [《Issue 操作指南》](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md)。

#### 📤 Step2：代码提交与合入

- 若为 `master` 分支上的特性内容，请遵守 [master 主线分支的代码提交与合入流程](#-master-主线分支的代码提交与合入流程)，完成代码开发与合入。
- 若为 `experimental` 分支上的特性内容，请遵守 [experimental 分支的代码提交与合入流程](#-experimental-分支的代码提交与合入流程)，完成代码开发与合入。

#### ✅ Step3：标记 Issue 已完成

---

### L2 和 L3 类特性贡献流程

#### 📌 Step1：创建 Issue

同 L1 类特性的 [创建 Issue](#-step1创建-issue)。

#### 💡 Step2：方案预讨论

- Issue 责任人找 [sig 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ascendc/README.md) 的 maintainer 指定架构师，进行方案预讨论。讨论形式可以是 Issue 区讨论或单独会议。
- 预讨论完成后，架构师勾选 Issue 状态为 **技术评审中**，之后可进入下一阶段 —— sig 评审方案。

#### 📊 Step3：sig 评审方案

- **申报 sig 评审议题**：由 Issue 责任人申报评审议题，sig 会议链接和议题申报方式：https://etherpad-cann.meeting.osinfra.cn/p/sig-ascendc
- **参加 sig 例会评审方案**：按时参加 sig 例会，进行方案评审。

  - ❌ **评审未通过**
    - 若方案评审未通过，可重新设计方案，继续 Step2 → Step3 流程。
    - 若需求未接纳，则流程终止。

  - ✅ **评审通过**
    - 由 Issue 责任人填写会议纪要，重点包含以下信息：
      - 评审通过结论（如有遗留问题，请记录遗留问题内容和闭环时间）
      - sig 指定的新特性合入分支名（如有）  
        ⚠️ 请重点关注这一项，新特性一般先合入对应特性分支，待验证充分且稳定后再同步合入 master 分支。新特性分支名由 sig 指定。
      - sig 指定的新特性发布内容和 roadmap 节点（如有）
    - 基于评审结论纪要，找 [sig 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ascendc/README.md) 勾选 Issue 状态为 **已确认**，并创建对应新特性分支（如有），之后可进入下一阶段 —— 合入 experimental 分支。

#### 🔀 Step4：合入 experimental 分支

请遵守 [experimental 分支的代码提交与合入流程](#-experimental-分支的代码提交与合入流程)，完成代码开发与合入。

#### 🚀 Step5：合入 master 主线分支

请遵守 [master 主线分支的代码提交与合入流程](#-master-主线分支的代码提交与合入流程)，完成代码开发与合入。

> ⚠️ **注意事项**
> - 准备合入 `master` 主线分支的内容，必须已合入 `experimental` 分支，且经过充分验证（如对应新增的 UT/ST 测试）。
> - 准备合入 `master` 主线分支前，建议跟 [sig 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ascendc/README.md) 的 maintainer 对齐合入时间，避免您的代码被拒绝合入。（可以 PR 评论区 @maintainer_gitcode_id，对齐合入时间）
> - 相较于合入 `experimental` 分支，多一步关键流程：触发 CI 门禁并通过。

#### ✅ Step6：标记 Issue 已完成

---

## 📦 三、代码合入流程与开发交付件

### 🔀 experimental 分支的代码提交与合入流程

可参考 [《PR（Pull Request）操作指南》](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md) 进行贡献。关键流程如下：

- **Fork 仓库**
- **本地开发验证**
- **提交 Pull Request**
- **代码检视**
   - 找 [sig 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ascendc/README.md) 的 Committer 进行代码检视（可在评论区 @committer_gitcode_id 提醒 Committer 进行代码审查）
- **闭环检视意见**
   - 找参与代码检视的对应 Committer 确认意见已闭环，然后申请加分 lgtm/approve
- **合入 experimental 分支**

---

### 🚀 master 主线分支的代码提交与合入流程

可参考 [《PR（Pull Request）操作指南》](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md) 进行贡献。关键流程如下：

- **Fork 仓库**
- **本地开发验证**
- **提交 Pull Request**
- **触发 CI 门禁并通过**
   - 通过评论 `compile` 指令触发开源仓门禁，并依据 CI 检测结果进行修改。
   - 目前 CI 门禁包含以下检查项：代码编译、静态检查、UT 测试、冒烟测试。
   - 如涉及 codecheck 误报，请提交给 sig 成员 Committer 屏蔽。如未及时处理，可在评论区 @committer_gitcode_id 提醒 Committer 进行代码告警屏蔽处理。
- **代码检视**
   - 找 [sig 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ascendc/README.md) 的 Committer 进行代码检视（可在评论区 @committer_gitcode_id 提醒 Committer 进行代码审查）
- **闭环检视意见**
   - 找参与代码检视的对应 Committer 确认意见已闭环，然后申请加分 lgtm/approve
- **合入 master 主线分支**

---

### 📋 PR 上库要求

#### 代码交付件

- 需提供新特性的功能实现文件和测试用例文件。
- 如果是贡献新的 Ascend C API 的 Python 编程接口，请参考 [《Ascend C Python 编程接口开发指南》](https://gitcode.com/cann/pyasc/blob/master/docs/developer_guide.md)，完成对应代码交付件。

#### 文档交付件

- 新特性 README 文档为必选，其余文档可视情况提供。
- 如果是贡献新的 Ascend C API 的 Python 编程接口，请参考 [《Ascend C Python 编程接口开发指南》](https://gitcode.com/cann/pyasc/blob/master/docs/developer_guide.md)，完成对应文档交付件。

#### 合规检查

- 代码是否符合 [《C++ 编程规范》](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md) 和 [《Python 编程规范》](https://peps.python.org/pep-0008/)
- 代码是否编译通过
- Markdown 文档语法是否符合规范

#### PR 提交

- 通过 `git` 命令提交目标分支 PR
- 检查 PR 标题是否清晰、PR 描述是否规范（指明更改内容和原因、是否关联对应 Issue）
- 检查是否签署 CLA