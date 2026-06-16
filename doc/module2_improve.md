# ludicorp pentari

 

\ goal 请阅读以下运行记录及对应方案代码还有doc里的module1.md，分析加模块后的不足之处，并提出改进方案，并直接修改代码。这不是一次性的诊断任务。请执行多轮迭代循环：诊断 → 改进 → 实现 → 运行验证 → 继续诊断 → 继续改进。直到加模块后的版本在 ludicorp 和 pentari 上达到或超过目标 AUC。

# ludicorp pentari

\ goal 请阅读以下运行记录及对应方案代码，还有 `doc/module1.md`，分析加模块后的不足之处，并提出改进方案，直接修改代码。

这不是一次性的诊断任务。请执行多轮迭代循环：

```text
诊断 → 改进 → 实现 → 运行验证 → 继续诊断 → 继续改进
```

直到加模块后的版本在 `ludicorp` 和 `pentari` 上达到或超过目标 AUC。

---

## 目标指标

AUC 计算方式：

```text
ludicorp: AUC = mean(scores) / 20
pentari:  AUC = mean(scores) / 35
```

目标 AUC：

```text
ludicorp: >= 0.5500
pentari:  >= 0.1429
```

也就是说：

```text
ludicorp: mean(scores) >= 11.0
pentari:  mean(scores) >= 5.0
```

---

## 已有运行日志

请优先分析以下已有日志：

```bash
~/repo/aaai1/module2/EvoTest/output/ludicorp/our/gpt-oss-20b/20260616-062030
~/repo/aaai1/module2/EvoTest/output/pentari/our/gpt-oss-20b/20260616-062030
```

同时阅读：

```bash
~/repo/aaai1/module2/doc/module2_method.md
```

以及当前实现代码，尤其是：

```text
新增模块本身
新增模块和 our_agent 的接入逻辑
 
```

---

# 约束条件

## 禁止

严禁以下行为：

```text
硬编码 prompt
硬编码游戏规则
硬编码固定动作序列
使用 if game_name == "ludicorp"
使用 if game_name == "pentari"
针对某个游戏写特例逻辑
通过降低评估难度制造提升
修改 AUC 计算方式
修改 score 解析方式来伪造提升
删除新增模块后退回原版
让代码退化成没有加模块的实现
通过 git reset --hard / git checkout . / 删除 .git / 重建仓库等方式掩盖失败修改
```

## 必须满足

所有改动必须满足：

```text
改进是通用机制
对所有 Jericho 游戏理论上适用
不能明显损害其他游戏
必须保留新增模块的核心思想
必须基于日志证据做修改
必须用真实 rerun 结果验证
如果某一轮没有提升，必须继续诊断并修正，不能停止在解释阶段
```

---

## 加模块后的运行命令

环境变量：

```bash
export OPENAI_API_KEY="9975d0f6-0228-4fae-a36a-1a484aae01bf"
export OPENAI_BASE_URL="https://sd80rokkc19oii7djp10g.apigateway-cn-beijing.volceapi.com/v1"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
```

运行命令：

```bash
export OPENAI_API_KEY="9975d0f6-0228-4fae-a36a-1a484aae01bf"
export OPENAI_BASE_URL="https://sd80rokkc19oii7djp10g.apigateway-cn-beijing.volceapi.com/v1/"
export OPENAI_API_BASE="$OPENAI_BASE_URL"

/home/kaijie/miniconda3/envs/emu3/bin/python ~/repo/aaai1/module2/EvoTest/main.py \
  --game_name detective \
  --rom_path jericho-games/ \
  --agent_type our \
  --llm_model gpt-oss-20b \
  --evolution_llm_model gpt-oss-20b \
  --eval_runs 10 \
  --enable_dacs \
  --dacs_debug

```

其中 `GAME` 替换为：

```text
ludicorp
pentari
```

开发阶段可以用较小配置做 smoke test，例如 `--eval_runs 1` 或较短 step limit；但是每轮最终 validation 必须使用真实的 `--eval_runs 10` 结果，除非运行本身失败，此时必须先诊断失败原因。

---

## 多轮迭代要求

请持续循环，直到满足：

```text
ludicorp AUC >= 0.5500
pentari AUC >= 0.1429
```

每一轮必须输出：

```markdown
Iteration N

Diagnosis:
- 发现了什么问题
- 来自哪些日志证据

Hypothesis:
- 为什么这个问题会导致低分

Change:
- 修改了哪些文件
- 修改了哪些函数
- 改动逻辑是什么

Git:
- Modified files: [...]
- Commit hash: <hash>
- Commit message: "<message>"
- 如果本轮没有代码改动，说明 no commit created

Validation:
- ludicorp scores
- ludicorp mean
- ludicorp AUC
- pentari scores
- pentari mean
- pentari AUC

Decision:
- 是否达到目标
- 如果没有，下一轮继续改什么
```

如果某一轮没有提升，必须继续诊断，不要停止在解释阶段。

如果运行失败，也必须输出失败日志、失败原因、修复计划，并继续下一轮。

---

# 优先改进方向

请优先考虑以下通用机制，而不是写游戏特例。

 下面这个是凝练版，直接替换 `# 优先改进方向` 部分即可：

````markdown
# 优先改进方向

请优先从以下通用机制诊断和改进，禁止写 `ludicorp` / `pentari` 特例，禁止硬编码动作序列或游戏规则。

## 1. 确认 DACS 是否真正影响选择

优先检查 `--dacs_debug` 日志：

```text
raw_candidates / filtered_candidates
relevance / diversity / risk / final_score
DACS selected ids
final UCB selected idx
````

重点判断：

```text
DACS 是否总是保留 parent
UCB 是否仍然总选 parent
child 是否很少被尝试
DACS 是否把有潜力的 child 过滤掉
risk 是否过高导致探索不足
diversity 是否不足导致候选仍然同质化
```

优先改进：

```text
让 DACS final_score 作为 UCB 的轻量 bonus
对高 relevance + 高 diversity + 低访问次数 child 增加 novelty bonus
对连续低分 parent 降低保守偏置
确保 DACS 输出的候选池确实传入原版 UCB
```

## 2. 改进 proxy states 抽取

不要只抽 reward increase，因为 Jericho 奖励稀疏。优先从 trajectory 中抽：

```text
score increase
new observation
new location
inventory change
object discovered
valid action but no immediate score
repeated action
repeated observation
invalid feedback
no-progress loop
long stagnation segment
```

建议将 proxy states 分成：

```text
positive progress states
neutral progress states
negative failure states
```

目标是让 DACS 能识别“没立刻得分但推进了游戏”的动作。

## 3. 改进 relevance score

如果选出的 configuration 和当前失败无关，优先改 `relevance_score`。

相关性应鼓励：

```text
覆盖当前 failure pattern
继承 success memory
使用 failure memory
避免重复无效动作
包含探索新状态/检查新物品的通用策略
与 trajectory 中关键 state/action/object/location token 有 overlap
```

不要用“距离历史最优配置近”作为 relevance，否则会过度保守。

## 4. 改进 risk score

如果日志里反复低分，优先处理通用退化行为：

```text
一直 look / inventory / wait
重复同一个方向
重复 invalid action
observation-action loop
高温随机探索导致不复现成功路径
过度依赖 memory 导致不探索
```

risk 只做降权，不要粗暴删除所有高风险 child，避免探索不足。

## 5. 改进 diversity score

如果候选 prompt 不同但实际策略相似，优先改 `diversity_score`。

多样性不要只看 prompt，还要看：

```text
memory
tool-use routine
hyperparameters
failure-handling strategy
explore/exploit tendency
```

目标是候选池同时保留：

```text
探索型配置
成功记忆利用型配置
反循环配置
风险规避配置
新状态发现配置
```

## 6. 改进 child configuration 质量检查

如果 Evolver 生成的 child 本身质量差，优先加入通用 validation / repair：

```text
child 为空或解析失败 → fallback parent + safe mutation
缺少 memory/tool/prompt 字段 → 回填 parent 字段
prompt 过短 → 保留 parent prompt 并追加 mutation
缺少 failure avoidance → 加入通用反循环规则
缺少 success memory 使用规则 → 加入通用 memory retrieval rule
```

只做结构修复，不写游戏内容。

## 7. 调整探索-利用平衡

如果探索不足：

```text
提高 dacs_beta_diversity
降低 dacs_gamma_risk
增加 child novelty bonus
限制 parent 连续被选次数
```

如果探索过度：

```text
提高 dacs_gamma_risk
提高 success-memory relevance
提高历史高分配置 prior
降低 diversity 权重
对低分 child 加 cooldown
```

所有调整必须通过参数或通用机制实现。

## 8. 增强可诊断日志

`--dacs_debug` 至少输出：

```text
parent_idx
raw candidate ids
selected candidate ids
final UCB selected idx
score table
top proxy states summary
fallback reason
why candidate was filtered
```

日志要能回答：

```text
DACS 看到了什么 proxy states
为什么某个候选得分高/低
最终 UCB 为什么选这个配置
```

## 9. 每轮改动必须直接服务 AUC

不要只做代码清理。每轮改动必须对应以下至少一个目标：

```text
提高有效探索
减少重复无效动作
减少 action/observation loop
提升 success memory 复用
提升 child configuration 多样性
降低 bad child 破坏
让 UCB 从更有意义的候选中选择
```

如果 AUC 没提升，必须继续判断：

```text
是 DACS 没选到好配置
还是 Evolver 没生成好配置
还是 proxy states 抽取不准
还是 relevance/risk/diversity 权重不合适
还是 UCB 覆盖了 DACS 效果
```

```
```


---

## Git 提交要求

每发生一次代码改动，必须立即提交到 git。

具体要求：

1. 在修改代码前，先检查当前 git 状态：

```bash
git status
```

2. 每一轮 Iteration 中，只要修改了任何代码、配置、文档或脚本，都必须在该轮 validation 之前或之后立即提交。

3. 每次提交必须包含：

* 本轮修改涉及的文件
* 修改目的
* 对应的 Iteration 编号
* 简短说明该修改解决了什么问题

提交格式示例：

```bash
git add <modified_files>
git commit -m "Iteration N: improve generic loop escape and failure suppression"
```

4. 禁止把多轮修改堆积到最后一次性提交。

5. 如果某一轮有多处独立改动，应拆成多个 git commit，而不是混在一起。

6. 如果某一轮没有代码改动，只做了日志分析或验证，则不需要提交，但必须在该轮输出中明确说明：

```text
Git:
- No code changes in this iteration; no commit created.
```

7. 每一轮输出必须额外包含 Git 部分：

```text
Git:
- Modified files: [...]
- Commit hash: <git commit hash>
- Commit message: "<message>"
```

8. 每次 commit 后，必须记录当前 commit hash：

```bash
git rev-parse --short HEAD
```

9. 最终总结中必须列出本次任务产生的所有 commit：

```text
Git Commits:
- <hash1> Iteration 1: ...
- <hash2> Iteration 2: ...
- <hash3> Iteration 3: ...
```

10. 不允许通过 `git reset --hard`、`git checkout .`、删除 `.git`、重建仓库等方式掩盖失败修改。若需要回滚某次失败修改，必须使用新的 commit 明确 revert 或修正。

---

## 最终总结要求

当两个目标都达到后，输出最终总结：

```markdown
Final Summary

Target:
- ludicorp target AUC: >= 0.5500
- pentari target AUC: >= 0.1429

Final Results:
- ludicorp scores:
- ludicorp mean:
- ludicorp AUC:
- pentari scores:
- pentari mean:
- pentari AUC:

Main Problems Found:
- ...

Main Changes:
- ...

Why It Works:
- ...

Remaining Limitations:
- ...

Git Commits:
- <hash1> Iteration 1: ...
- <hash2> Iteration 2: ...
- <hash3> Iteration 3: ...
```

