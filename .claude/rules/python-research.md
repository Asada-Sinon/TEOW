---
paths: ["**/*.py"]
description: "科研 Python 代码的可复现性底线"
---

# Python 可复现性底线

**自查信号**：如果你正在写一个会产生数值结果的 `.py`，却没有显式设置随机种子，或正把超参数直接写成字面量 —— 说明本规则已被 compact 丢掉了，请重新 Read `.claude/rules/python-research.md`。（带 `paths:` 的规则压缩后不会自动重载。）

1. **随机种子必须显式设置并记录。** 覆盖用到的每个来源（`random`、`numpy`、框架 RNG、DataLoader worker）。禁止依赖默认种子，禁止不落盘的种子。

2. **配置走文件，不硬编码。** 超参数、路径、开关从配置文件或 CLI 进来，并把 resolved config 存进本次 run 目录 `experiments/<YYYYMMDD>-<slug>/`。代码里出现 `lr = 3e-4` 这种字面量就是错的。

3. **不要在实验脚本里改全局状态。** 包括 `os.environ` 赋值、`matplotlib` 全局 rcParams、`torch.set_default_dtype`、monkey-patch 第三方库、修改传入的 config 对象。要改就在显式的 setup 函数里改，并记录下来。

4. **新想法先在 `explorations/` 验证，通过才进 `src/`。** `src/` 是被复用、被信任的代码；一次性尝试、临时分析脚本、还没验证的点子都留在 `explorations/`，文件名写清它在回答什么问题。

5. **数值计算不得依赖未固定版本的库行为。** 用到的数值库要在依赖文件里固定版本；不要依赖未文档化的默认值、不同版本间会变的算子实现、字典/集合的迭代顺序，以及浮点误差范围内的相等判断。

6. **产出数字的只能是受版本控制的脚本。** 不在对话里口算指标，不用一次性 `python -c` 出正式数字 —— 那份计算过程没人能复现。
