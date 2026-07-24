# settings.json 逐条说明

这份文档解释 `.claude/settings.json` 里每个 key 为什么这么配，以及权限语法里
几个**很容易写错但不会报错**的坑。settings.json 是严格 JSON，不能写注释，
所以说明只能放在这里。

---

## 1. `$schema`

```json
"$schema": "https://json.schemastore.org/claude-code-settings.json"
```

编辑器（VS Code / JetBrains）会据此提供补全和校验。写错 key 名当场标红，
比等到运行时静默失效强得多。它只是元数据，不影响运行。

---

## 2. `permissions.allow` —— 免打扰清单

放进来的是**只读或低风险、且高频**的命令。目的是减少确认弹窗，
让 agent 不会每看一次 `git diff` 就打断你一次。

| 条目 | 理由 |
|---|---|
| `Bash(git status:*)` `Bash(git diff:*)` `Bash(git log:*)` `Bash(git show:*)` `Bash(git branch:*)` | 纯只读，agent 需要频繁自查状态 |
| `Bash(git add:*)` | 只动暂存区，不产生新对象也不改历史，`git reset` 一步撤销 |
| `Bash(pytest:*)` `Bash(python -m pytest:*)` `Bash(python3 -m pytest:*)` | 跑测试是主循环的一部分，不放行等于每轮都要点确认 |
| `Bash(ruff check:*)` `Bash(ruff format:*)` | 同上，且 hook 里本来就会跑 |
| `Bash(npm test)` | 精确匹配，不带参数 |
| `Bash(npm run *)` | **这里的空格是有意义的**，见下面第 4 节 |
| `Bash(ls *)` | 只读浏览 |

**没有放行的**（刻意的）：`git commit`、`git push`、`rg`、`rm`、`mv`、
任何包管理器的 install、任何 wrapper（`docker` / `devbox` / `npx` / `uv run`）。

### 为什么 `Bash(git commit:*)` **不在** allow 里

这条曾经在 allow 里，理由是"commit 可回滚"。现在移出了，每次 commit 都要你确认。
原因是它和本工作流的一条核心设计原则直接冲突：

> **commit 的时机和粒度由用户决定。**
> `/impl` skill 明写「不要自己 commit」，`docs/HUMAN_PLAYBOOK.md` 把
> "每个 phase 验证通过就 commit" 定为**你唯一可靠的还原点**。

关键在于：**skill 的 `allowed-tools` 字段不构成硬限制**（2.1.218 实测），
它更像是给模型看的提示。所以在移出之前，"不要自己 commit" 这条约定
**没有任何机制在兜底** —— agent 想 commit 随时能 commit，而且免确认。

而 commit 的"可回滚"是有代价的：agent 自作主张切碎或合并提交之后，
你的还原点粒度就不是你要的那个了，事后 rebase 比当场点一次确认贵得多。

代价评估：一次 phase 通常只 commit 一到两次，加一次确认的打扰远小于
"发现历史被搅乱"的成本。**这笔交易划算**。
如果你个人偏好让 agent 自行 commit，加到**不进版本库**的
`.claude/settings.local.json` 里，别改这份共享配置：

```json
{ "permissions": { "allow": ["Bash(git commit:*)"] } }
```

### 为什么 `Bash(rg *)` **不在** allow 里

`rg` 看着是纯只读，其实不是：

```
rg --pre <任意可执行文件> pattern .     # --pre 会对每个文件执行它，等于任意命令执行
rg --hostname-bin <任意可执行文件>      # 同理
```

这正是第 5 节 wrapper 警告说的那类东西 —— 一条看似只读的命令借壳跑任意程序。
而 `Bash(rg *)` 是前缀通配，`rg --pre ...` 照样命中，等于放行。
拿 deny 去堵也堵不住：`Bash(rg --pre *)` 只能匹配 `--pre` 紧跟在 `rg` 后面的写法，
`rg pattern --pre foo` 就绕过去了。

**代价接近于零**：Claude Code 有内置的 `Grep` 工具，底层就是 ripgrep，
不走 Bash、不需要这条权限，而且更快。让 agent 用 `Grep` 就行。
（同理，别顺手往 allow 里加 `Bash(find *)` —— `find -exec` 是一样的问题。）

---

## 3. `permissions.deny` —— 硬禁止

```json
"Bash(git push --force *)"
"Bash(git push --force-with-lease *)"
"Read(.env)"
"Read(**/.env.*)"
"Edit(.env)"
"Edit(**/.env.*)"
```

* force push 会**改写别人的历史**，是少数真正不可逆的 git 操作。
* `.env` 系列既禁读也禁写：禁读是防止密钥被吸进 context 再随对话外泄；
  禁写是防止 agent "帮你整理"配置时把 key 覆盖掉。
  （`protect_paths.py` 里也有一份 `.env` 规则，两层保险。）

### 坑 A：deny > ask > allow，且 **deny 规则无法带白名单例外**

三类规则的优先级是固定的：**deny 胜过 ask，ask 胜过 allow**。
这意味着你**没有办法**表达"禁止所有 `git push`，但允许 push 到我的 fork"。
一旦某个模式进了 deny，就没有任何语法能给它开口子。

所以 deny 要写得**尽量精确**。这也是为什么上面写的是
`Bash(git push --force *)` 而不是 `Bash(git push *)` ——
后者会把正常的 push 也一起焊死，而你无法再为它加例外。

### 坑 B：裸工具名做 deny 会把工具整个从 context 里移除

写 `"deny": ["Bash"]`（不带括号参数）不是"每次都拒绝"，
而是**这个工具根本不会出现在模型的工具列表里**。
副作用是模型不知道自己能跑命令，会开始编造"我无法执行命令"之类的说法，
行为变得难以预测。要限制某个工具，请用带参数的模式，不要用裸名。

---

## 4. 坑 C：`Bash(ls *)` 和 `Bash(ls*)` 不是一回事

匹配是**前缀通配**，空格属于模式本身：

| 模式 | 匹配 `ls -la` | 匹配 `lsof` |
|---|---|---|
| `Bash(ls *)` | ✅ | ❌ |
| `Bash(ls*)`  | ✅ | ✅ |

`Bash(ls *)` 要求 `ls` 后面必须跟一个空格，所以 `lsof`、`lsblk` 都不会命中。
`Bash(ls*)` 少了那个空格，就把所有以 `ls` 开头的命令全放行了 —— 包括你没想到的。

**规则：写 allow 时永远带上空格**（`Bash(npm run *)` 而不是 `Bash(npm run*)`）。
少打一个空格，放行范围可能扩大几十倍。

另外，`:*` 形式（如 `Bash(git status:*)`）表示"这个子命令带任意参数"，
语义上比裸 `*` 更收敛，能用就优先用。

---

## 5. 坑 D：wrapper 命令等于把权限系统整个绕过去

权限系统会剥离少数已知的前缀 wrapper 再做匹配，但
**`devbox run` / `npx` / `docker exec` 不在剥离列表里**。

后果很直接：

```
Bash(devbox run *)     ≈  放行  devbox run rm -rf .
Bash(npx *)            ≈  放行  npx 任意包（会联网下载并执行）
Bash(docker exec *)    ≈  放行  容器内任意命令
```

它们被当成一整条不透明的字符串来匹配，后面跟什么都算命中。
所以：**永远不要宽泛放行 wrapper**。要放就把整条命令写死：

```json
"Bash(devbox run test)"
"Bash(npx --no-install prettier --write *)"
```

（本模板的 `format_lint.py` 调 `npx` 时也强制加了 `--no-install`，
同样是为了堵住"联网下载任意包再执行"这条路。）

---

## 6. 坑 E：文件规则只匹配 `Edit()` 和 `Read()`

**没有 `Write(...)` 这种规则。**

`"Write(docs/**)"` 写进去不会报错，schema 校验也过得去，但它**永远不会匹配任何东西** ——
你以为加了保护，其实什么都没发生。这是最阴的一个坑，因为完全没有反馈。

对写入的控制只能靠 `Edit(...)`，或者像本模板一样用 **PreToolUse hook**
（`protect_paths.py`）自己判断 —— hook 的 matcher 里 `Edit|Write` 是真实生效的，
两条路径别搞混。

另外文件路径规则用的是 gitignore 风格的 glob：
`.env` 只匹配项目根那一个，`**/.env.*` 才能匹配任意层级。

---

## 7. 为什么**没有**设 `defaultMode`

`defaultMode` 决定整个会话的默认权限档位（如 `acceptEdits`、`plan`、
`bypassPermissions`）。模板刻意不设它，因为：

* 这是**个人风险偏好**，不该由模板替用户决定；
* 尤其是 `bypassPermissions`，一旦写进项目共享的 settings.json，
  等于替所有 clone 这个仓库的人关掉了安全网，而他们可能根本没注意到。

要设就自己加，建议加在**不进版本库**的 `.claude/settings.local.json` 里：

```json
{ "permissions": { "defaultMode": "acceptEdits" } }
```

常用档位：

* `default` —— 每次写操作都问（最稳，新项目建议先用这个）
* `acceptEdits` —— 自动接受文件编辑，Bash 仍然要确认（日常推荐）
* `plan` —— 只读探索，不做任何修改（读代码 / 做方案时用）
* `bypassPermissions` —— 全放行，**只在一次性容器里用**

---

## 8. `hooks` 段

四个 hook 各司其职：

| 事件 | matcher | 脚本 | 作用 | 能否阻断 |
|---|---|---|---|---|
| `PreToolUse` | `Edit\|Write` | `protect_paths.py` | 受保护路径护栏 | **能**（exit 2） |
| `PostToolUse` | `Edit\|Write` | `format_lint.py` | 自动格式化 + lint 结果注入 context | 否（永远 exit 0） |
| `Stop` | 无 | `verify_stop.py` | 完成门禁，跑 `.claude/verify.sh` | **能**（exit 2） |
| `SessionStart` | `startup\|resume\|clear\|compact` | `session_context.py` | 注入 HANDOFF / 焦点 / 记忆 / git 状态 | 否 |

几个要点：

* **命令写法**：`"\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/xxx.py"`。
  必须用 `$CLAUDE_PROJECT_DIR`（hook 的工作目录不保证是项目根），
  且**必须加引号** —— 项目路径里一旦有空格，不加引号就会被拆成两个参数。
* **matcher 是正则**，`Edit|Write` 表示两个工具都拦。`Stop` 事件没有 matcher 概念，
  所以那一项直接省略。
* **`SessionStart` 的 matcher 必须四个 source 全写。** 这是最容易漏、
  而且漏了完全没有报错的一处，单独展开在下面第 9 节。
* **`timeout` 单位是秒**。`Stop` 给了 310，比 `verify_stop.py` 内部的 300 秒
  略大一点，让 Python 有机会先超时并给出友好提示，而不是被外层硬砍掉。
* **exit code 语义**：`0` = 放行；`2` = 阻断并把 stderr 回给 Claude；
  其他非零 = 只记日志、不阻断。本模板只用 0 和 2。

---

## 9. 坑 F：`SessionStart` 的 matcher 漏一个 source，注入就静默失效

`SessionStart` 的 `source` 枚举一共**四个**：

| source | 什么时候发生 |
|---|---|
| `startup` | 冷启动 `claude` |
| `resume` | `claude --resume` / `/resume` 恢复旧会话 |
| `clear` | **`/clear`** |
| `compact` | 上下文压缩（自动或 `/compact`） |

matcher 是**正则**，所以 `"startup|compact"` 这种写法会让 `/clear` 和 `--resume`
**一声不响地不触发 hook**：没有报错，没有日志，只是 context 里少了那一块，
你要盯着注入内容才能发现。

对本工作流来说这是最要命的一条，因为 **`/clear` 是本工作流的高频固定动作**：

* 纠正同一个问题 2 次 → 立刻 `/clear`
* `/plan` 和 `/impl` 之间 → 必 `/clear`
* 每个 phase 之间 → `/clear`

而 `docs/HUMAN_PLAYBOOK.md` 向你承诺「开机后 SessionStart hook 会自动注入
`HANDOFF.md` / `.context/current-focus.md`」。漏掉 `clear` 的话，
这条承诺在**你最需要它的那个时刻**（刚清空对话、agent 什么都不知道）恰好不成立。

所以：

```json
"matcher": "startup|resume|clear|compact"
```

**验证方法**：`/clear` 之后看第一屏有没有 `===== 项目持久状态（自动注入）=====`。
没有就是 matcher 写漏了。

---

## 10. 扩展点：`.claude/compact-reminder.txt`

**模板里不带这个文件，需要你自己建。** `session_context.py` 会在
`source == "compact"` 时读它（上限 2000 字符，超出保留开头），
把内容放在注入块的**最前面**。其他三个 source 不读。

### 它解决什么问题

compact 压缩的是对话历史，但真正会丢的不止对话：

* **带 `paths:` 的条件规则**（本模板的 `.claude/rules/experiments.md`、
  `python-research.md`、`notes.md`）是在匹配到相应文件时才注入的。
  压缩之后那些文件可能已经不在 context 里，规则就**不会自动回来**。
* **子目录 CLAUDE.md** 同理，是读到该目录文件时才加载的。

结果就是：压缩前 agent 知道"往 `experiments/` 写东西要满足 run 目录必备项"，
压缩后它不知道了，而且它**不知道自己不知道**。
根 `CLAUDE.md` 会保留，所以只有那些**条件加载**的东西需要在这里重申。

参考：<https://code.claude.com/docs/en/context-window>

### 怎么写

只放**压缩后丢了会真出事**的硬约束。写长了没用 —— 它挤占的是压缩刚腾出来的空间。
**一屏以内，只写"不这么做会造成不可逆后果"的那几条。**

`.claude/compact-reminder.txt` 示例：

```text
【压缩后重申，以下内容优先级高于任何被压缩的对话摘要】

1. experiments/ results/ runs/ outputs/ 下的已有文件只读。
   新结果一律写新的 run 目录，绝不覆盖、不删除、不"顺手整理"。
2. 任何 run 目录必须同时有：git hash（含 dirty 标记）、resolved config
   （展开后的完整配置，不是路径）、seed、完整启动命令。缺一即不可复现。
3. 数值只能来自 src/ 下受版本控制的脚本产出的文件。不口算、不估算、
   不凭印象报指标；拿不到真实数字就直说拿不到。
4. research-log.md 的结论必须带 [AI-DRAFT] 标注和 [source: <run_id>]。
   [HUMAN-VERIFIED] 只有用户能打。
5. 不要自己 git commit。

如果你正准备写 experiments/ 而说不出第 2 条有哪几项，
先 Read .claude/rules/experiments.md 再动手。
```

### 检查它有没有生效

```bash
echo '{"source":"compact","cwd":"'"$PWD"'"}' | .claude/hooks/session_context.py
```

输出第一段应该是 `----- 压缩后提醒 .claude/compact-reminder.txt -----`。
把 `"source"` 换成 `"startup"` 再跑一次，这一段应该消失 —— 那就是对的。

---

## 11. `.claude/protected-paths.txt`：两类规则

`protect_paths.py` 的规则文件有两种语义，混淆了会以为护栏坏了：

| 写法 | 语义 |
|---|---|
| `experiments/**` | **仅当目标文件已存在时**阻断。新建放行。 |
| `!.env` | **无条件**阻断，文件存不存在都拦。 |

普通规则之所以只拦已存在的文件，是为了对齐 `CLAUDE.md` 硬约束第 2 条的原意 ——
「**已有**产物只读，新结果一律写新目录」。一刀切拦掉新建会误伤 `/exp`：
它必须往 `experiments/<run_id>/` 里写 git hash / resolved config / seed，
而这恰恰是硬约束第 3 条**要求**的。护栏要保护的是"不覆盖已有 run"，不是"不许写"。

`!` 前缀留给「新建它本身就危险」的东西：凭空多出一个 `.env` 会让密钥进仓库，
手写一个 `uv.lock` 和改一个同样污染复现性。

**注意 `!` 不是 gitignore 的取反语义**，别照 `.gitignore` 的直觉读，它就是"更严格"。
