# Base case 的样本路径随机种子敏感性实验

共 30 个配置，sample-path seed 为 42–71。以生成时 `table.dat` 中的 `cs_base` 为完整模板，仅调整样本路径随机种子，并相应更新 UID；实验名统一为 `cs_base`。

固定配置：case-study 环境、discount factor=0.99、absorption_linear_penalty、系数绝对上界 1000、512 个场景、mixture probability=0.1、无 regularization、初始占用率 0.5。所有运行使用完全相同的 shared initial state，init_state_seed 固定为 12345。

按照现有 `_pin_sampling_seeds` 的定义，每个 seed 同时赋给 `arrival_random_seed`、`stop_time_random_seed` 和 `env_random_seed`，改变到达过程与路径长度的随机流。初始状态已明确给定，不会随这三个 seed 重新抽样。

30 组实验名统一为 `cs_base`，结果集中在同一个文件夹。不同 seed 对应不同 UID，因此 checkpoint 文件相互区分。seed 42 与原 Base 完全相同，合并为同一条命令，并使用同一个 checkpoint。

## 命令位置

- 仓库根目录 `table.dat`：共 41 条命令。原有 12 条保留；seed 42 对应第 1 条，seed 43–71 对应第 13–41 条。30 组 seed 配置全部保留，重复的 seed 42 命令已合并。
- 本目录 `commands.dat`：仅包含这 30 条命令，编号 1–30。
- `manifest.json`：记录 seed、两个命令文件中的编号、实验名、UID 和结果目录。

在仓库根目录运行一个指定 seed 的示例（seed 42）：

```bash
mkdir -p STATUSES
bash single_case.sh table.dat 1
```

这一步仅生成并验证命令，尚未启动训练。

## 结果对应

训练结果写入 `experiments/results/cs_base/<job_id>.jsonl`。其中 `coefficients` 为待比较的系数向量，`training_time_seconds` 为本次训练调用耗时，`tight_penalized_lower_bound` 为训练样本上的目标值。使用 manifest 将结果 UID 对应回 seed。

同一 seed 的重复运行应按 UID 去重；seed 42 就是原 Base，不作为另一组独立样本计数。若比较不同系数的目标表现，应使用共同的独立评估样本，不能直接将各自训练样本上的目标差异全部解释为系数差异。
