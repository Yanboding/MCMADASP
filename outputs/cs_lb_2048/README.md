# 六组训练系数的样本外下界评估（2048 条共用样本路径）

目的：检验训练得到的 penalty coefficients 是否能收紧 information relaxation lower bound。
对 `cs_base`、`cs_l1_0.001`、`cs_l2_0.001`、`cs_n256`、`cs_l1_0.01`、`cs_l2_0.01` 六个结果目录中的系数，各自在**同一批** 2048 条测试样本路径上计算 information relaxation cost，
每条路径同时给出 penalty ratio 0（零 penalty）和 1（训练系数）两个下界。共 6 × 2048 = 12288 条命令（前四组先生成，后两组随后追加，路径相同）。

## 固定配置

- 环境：case-study（EJOR）环境，discount factor 0.99，与六组训练命令的 `env_args` 完全相同（已校验一致）。
- 初始状态：占用率 0.5 的固定状态（每天 68 个 regular 预约，overtime 0，waitlist 取到达率四舍五入），即训练时的 shared initial state；2048 条路径全部从该状态出发，无 warm-up。
- 路径长度：`{"type": "geometric", "discount_factor_proposal": 0.99}`，即 Geom(1 − 0.99)，支撑 {1, 2, …}；不做 importance sampling。记录中的 `period_weights` 为 `[1, 0.99, 0.99, …]`（目标分布下的 survival weights），`terminal` 为 `absorbed`。
- Penalty：`absorption_linear_penalty`，系数按 uid 从各目录锁定（见 manifest 的 `training_uid` / `coefficients_file`）。命令只算下界（`policy_specs` 为空，记录 `policy_id` 为 `information_relaxation_only`），`penalty_ratios = [0, 1]`。
- 样本路径随机种子：训练种子 42 加 offset 1001（env / arrival / stop-time 三个种子均为 1043），与训练路径（种子 42–71）无交集。六组实验使用相同种子，路径逐条相同（生成脚本已逐条校验 `init_state`、`sample_path`、`period_weights`、`terminal`）。
- 路径长度统计（2048 条）：均值 100.4，中位数 71，最短 1，最长 719。

## 六组系数

| 目录 | 训练 uid | N | 正则 | 系数上界 | 训练样本目标值 | 命令行号 |
|---|---|---|---|---|---|---|
| cs_base | a059bcb03751e9ffccf4120612e3bc31 | 512 | 无 | 1000 | 263979.06 | 1–2048 |
| cs_l1_0.001 | ca79aeebf6e5674ad47990c94e03124e | 512 | L1, λ=0.001 | 1000 | 263929.80 | 2049–4096 |
| cs_l2_0.001 | 54fdbe3d4f4d14d02f464847aa241912 | 512 | L2, λ=0.001 | 1000 | 228114.95 | 4097–6144 |
| cs_n256 | 15aeed3b86d80ed7d2ac49d57b92131b | 256 | 无 | 1000 | 396433.01 | 6145–8192 |
| cs_l1_0.01 | ae63a04776c918164186257ace40a782 | 512 | L1, λ=0.01 | 1000 | 259859.36 | 8193–10240 |
| cs_l2_0.01 | a49e7a2dfe393dcd1e072970c56522ec | 512 | L2, λ=0.01 | 1000 | 227037.51 | 10241–12288 |

训练目标值是训练 proposal（mixture geometric）上的全 LP 松弛值，只作参考，不能与样本外下界直接比较。

## 文件

- 仓库根目录 `table.dat`：仅含这 12288 条评估命令（第 1–12288 行），每条命令一条路径，按实验分段排列（见上表行号）。原来的 41 条 `cs_*` 训练命令已移到 `table_train_cs.dat`（编号 1–41，顺序不变）。
- `generate_commands.py`：生成脚本。从 `table_train_cs.dat` 读取六组种子 42 的训练命令（`env_args`、`agent_args`），从 `experiments/results/<name>/` 中按 uid 锁定系数记录并复制到 `coefficients/<name>/<uid>.jsonl`，再调用 `generate_test_paths_and_init_state`（与 `generate_params.py lowerbound` 相同的生成函数）生成记录并追加到 `table.dat`。`cs_*` 四个实验名没有注册在 `EXPERIMENT_SPECS` 中，所以不能直接用 `generate_params.py lowerbound cs_base ...`；脚本等价于对每组执行 `lowerbound --paths 2048 --groups 2048 --eval-proposal '{"type": "geometric", "discount_factor_proposal": 0.99}' --seed-offset 1001 --penalty-ratios 1 --penalty-dir outputs/cs_lb_2048/coefficients/<name>`，只是变体直接取自训练命令。写入函数是追加模式，重复运行会因命令文本相同而跳过；要从头重建先删除 `table.dat`：

  ```bash
  rm -f table.dat
  python outputs/cs_lb_2048/generate_commands.py
  ```

- `manifest.json`：种子、proposal、初始状态、路径长度统计、每组系数的来源文件与行号范围。
- `coefficients/<name>/<uid>.jsonl`：锁定的训练记录副本（`.gitignore` 忽略 `*jsonl`）。

## 运行

在仓库根目录：

```bash
mkdir -p STATUSES
bash single_case.sh table.dat 1
```

集群上按 job array 提交第 1–12288 行（`N_cases=12288`）。每条命令只建一个 Gurobi env（无 subproblem），不需要多核。本地只做过一次冒烟检查（`cs_base` 段中一条 15 期路径，两个下界各约 0.8 秒，结果文件已删除），未做任何正式评估。

## 结果与汇总

结果写入 `experiments/results/<name>/<job_id>.jsonl`，与训练记录同目录；汇总代码会跳过没有 `policy_id` 的训练记录。每条记录包含 `zero_information_relaxation_cost`、`penalized_information_relaxation_cost`、`information_relaxation_cost_by_penalty_ratio`（`[[0, cost], [1, cost]]`）和 `coefficients_source`。

按目录汇总（zero / penalized 下界均值、配对差及置信区间）：

```python
from experiments.new_result_aggregration import report_information_relaxation_lower_bounds
for name in ['cs_base', 'cs_l1_0.001', 'cs_l2_0.001', 'cs_n256', 'cs_l1_0.01', 'cs_l2_0.01']:
    report_information_relaxation_lower_bounds(name)
```

`report_penalty_shrinkage(name)` 给出同样的 t=0 / t=1 对比以及 penalized 下界低于零 penalty 下界的路径比例。六组路径相同、`uid` 不同（uid 含系数），跨组比较时按路径顺序（`manifest.json` 行号）配对即可。
