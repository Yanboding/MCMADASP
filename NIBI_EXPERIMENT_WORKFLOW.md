# MCMADASP 实验运行流程（项目默认规则）

由用户于 2026-10-18 确认。今后本项目的服务器实验默认遵循本流程；用户明确的新指令优先。本文是 Codex、Claude Code 和其他 agent 的共同操作依据。

## 1. 固定位置与登录

- 本地代码：`/Users/yanboding/Desktop/PhdTopic/MCMADASP`。
- 本地论文：`/Users/yanboding/Desktop/Dynamic-Advance-Scheduling`。
- SSH：`ssh nibi.sharcnet.ca`，用户名 `dingyanb`。
- 服务器代码：`/home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP`。
- MFA 默认选择 **Duo Push to Pixel 8 root**，由用户在手机上批准。按设备名称选择，不假定菜单编号永远不变。
- 优先复用仍有效的 SSH 连接；连接失效时正常完成 MFA，不绕过认证，不在文件或聊天中记录密码、验证码或私钥。
- 只有实际收到服务器命令输出，才能报告登录、提交或下载成功。

## 2. 代码用 Git 同步；结果不进入 Git

1. 检查本地和服务器的分支、commit 与工作区修改，再同步此次实验所需的代码。默认仓库为 `https://github.com/Yanboding/MCMADASP.git`；最近核验分支为 `bh_penalty`，运行前重新确认。
2. 保留服务器已有修改；不得为同步而直接 `reset --hard`、`git clean`、强推或覆盖提交脚本。若发生真实冲突，先检查差异并解决。
3. 只将所需代码、轻量配置和说明纳入 Git。**实验结果、原始 sample paths、日志、大型输出和 farm 状态不得上传到 Git。**
4. `table.dat` 当前被 Git 忽略。Git 同步不会更新它：必须从对应代码/配置在服务器生成，或把本地已核验的命令表单独传到服务器。
5. 同步后记录服务器的完整 commit SHA。若保留未提交的运行脚本修改，同时记录其 diff 或校验值，不能只用 commit 表示整个运行配置。

## 3. 提交新 farm 前的顺序

1. 检查 `squeue -u dingyanb` 和现有 farm 状态。旧 worker 或 `-auto` 重提交任务仍可能使用该目录时，不清理或覆盖命令表；不擅自取消旧任务。
2. 在清理前，把需要保留的结果、命令表、`farm.log`、`OUTPUT/`、`STATUSES/` 和相关配置下载或另行保存。实验结果的位置以代码和本次命令配置为准，不假定全在 `OUTPUT/`。
3. 在服务器项目根目录中删除以下旧 farm 文件或目录：
   - `MISC/`
   - `OUTPUT/`
   - `STATUSES/`
   - `TMP`
4. 清理前核实工作目录和各路径类型。**`TMP` 若是符号链接，只删除链接本身，不递归删除链接指向的目录；操作符号链接时不加尾部 `/`。** `TMP` 若是普通目录，只删除本 farm 的该目录。其他清理目标若也是符号链接，同样不跟随链接。
5. 不扩大清理范围：不删除代码、实验结果目录或 `table.dat`，不使用未审查的通配符或 `git clean` 替代上述四项清理。
6. 清理是启动新 farm 前的一次性步骤。运行中的 farm 和 `-auto` 的续跑必须保留状态，不能重复清理。

## 4. 所有正式命令写入 table.dat

- Meta Farm 的命令入口固定为服务器项目根目录的 `table.dat`，每个 case 一行，格式为：

  ```text
  <unique-case-id> python run.py <this experiment's arguments>
  ```

- 实验要求的每条任务命令都必须出现在该文件中；不能仅放在另一个文件名中就提交。Git 同步、环境检查、清理和提交属于管理步骤，不写入实验命令表。
- 核查命令数量、唯一 case ID、参数/路径引用、随机种子和结果标识，确保并发 case 不覆盖彼此的结果。参数结构和 CLI 以当前代码为准。
- 提交前保存命令表的副本或校验值及其 case-to-setting/repetition 映射。运行期间不覆盖 `table.dat`。
- `number of jobs` 是 Meta Farm worker 数量，不是命令行数。worker 会依次取用 `table.dat` 的命令，Python 内部还可以并行求解子问题。

## 5. Gurobi token 是强制限制

**本账户同时运行任务的 Gurobi token 总占用必须严格小于 5000。达到或超过该限制可能导致账户被封禁。**

提交前必须核实：

- 单个 job 的 token 占用上限，包括主问题环境、子问题环境和其他同时存活的 Gurobi 环境。
- 本账户其他任务的占用，以及已排队、续跑或计划同时启动的任务所需额度。不能只看到自己的某个 job 未运行，就忽略其之后启动的占用。
- CPU 数不是 token 数的充分依据。当前 PO 代码除子问题环境外还创建主问题环境；实际 license checkout 方式要核实，不能默认二者共享 token。

对同配置 worker，提交数须满足：

```text
number_of_jobs * tokens_per_job + other_concurrent_tokens < 5000
```

整数 token 计数下，理论最大值为：

```text
floor((4999 - other_concurrent_tokens) / tokens_per_job)
```

没有可用额度时不提交。若没有其他占用：

| 每个 job 的实际 token 占用 | 最大 worker 数 | 总 token |
|---:|---:|---:|
| 32 | 156 | 4992 |
| 33 | 151 | 4983 |

这些是上限，不是默认必须申请的数量。使用可核验的保守占用上限，并给不确定或其他并发使用留出余量；无法确定安全预算时先解决预算问题，不启动大批量任务。不得绕过或在不同批次中重复使用同一份 token 额度。提交时的一次使用量快照不足以保证后续安全；要协调所有 farm 的并发预算，或使用可执行的共享并发限制，避免其他任务随后启动而超限。

## 6. 环境与正式提交

2026-10-18 已核验的环境如下，仅为已知配置；每次实验还要检查当前脚本与依赖：

| 项目 | 已核验值 |
|---|---|
| 基础环境 | `StdEnv/2023` |
| 科学计算模块 | `scipy-stack/2026a` |
| Python virtualenv | `~/env_gurobi`，Python `3.11.5` |
| Gurobi | `gurobi/12.0.0`，许可证初始化成功 |
| Meta Farm | `meta-farm/1.0.3` |
| 当时 worker 配置 | 1 task × 16 CPUs，64 GB，48 小时，`def-sarhangi` |

- 环境核查只单独加载所需模块、激活 virtualenv 并进行轻量检查。**不要为初始化环境而运行 `setup.sh`：它会立即执行实验命令。**
- 正式实验在计算节点通过 Meta Farm 运行，不在登录节点执行 PO 或长时间求解。
- 核查 `job_script.sh` 的 CPU、内存、时限、工作目录和环境，以及 `resubmit_script.sh` 的自动续跑配置。资源数随实验确定，不能把上述历史值当作永久设置。
- 用户指定的当前资源配置：**PO training 两个脚本均设 `--ntasks-per-node=32`、`--mem=128G`；policy evaluation 两个脚本均设 `--ntasks-per-node=16`、`--mem=64G`**。不要保留旧的 `--cpus-per-task=16`，否则会与 task 数相乘。使用 `experiments/configure_farm.py` 切换配置，并传入已核验的其他任务 token 占用。
- 提交前加载 `meta-farm/1.0.3`（或已核验的兼容版本），使用用户指定形式：

  ```bash
  submit.run <number_of_jobs> -auto
  ```

- `<number_of_jobs>` 必须替换为已核验 token 预算允许的正整数。参数 `-auto` 需要 `resubmit_script.sh`，且续跑不得扩大已批准的并发/token 预算。
- 不在 worker 启动时并发修改共享 virtualenv 或安装依赖。注意现有脚本可能写共享 `gurobi.env`，或在缺少 `tqdm` 时执行 `pip install`；扩大并发前检查这些行为。
- 保存提交命令、Slurm/farm job ID、代码版本、环境版本、资源数、token 预算、命令表校验值和实验配置。

## 7. 检查完成状态并下载结果

1. 结合 `squeue`、Slurm accounting、`farm.log`、`OUTPUT/`、`STATUSES/` 和实验自己的输出检查进度。
2. 不把队列为空、Slurm job 结束或退出码为 0 单独当作全部实验成功：按 `table.dat` 的 case ID 核对缺失/重复/失败记录，并检查所需结果字段、求解状态和最优性界是否保存。
3. Meta Farm 当前通过 `STATUSES/status.*` 记录 case ID 和退出码，并通过 `source` 调用 `single_case.sh`。不要未经核对就在该脚本中添加会退出整个 worker 的 `exit`。
4. 结果用 `rsync`、`scp` 或等价下载方式拉回本地独立的实验目录，保留 setting、重复编号、seed 和 case ID。核对文件数量、大小，必要时核对校验值；不要用下载命令删除本地已有结果。
5. 结果不提交 Git、不 push。对当前 repeated-SAA 实验，confidence interval 与可选降方差后处理在服务器计算完成、结果下载后在本地执行。
6. 仅当用户要求定期监控时才建立监控安排；不要承诺连接失效后仍可自动绕过 MFA 或无人值守执行所有操作。

## 8. 范围与完成报告

- 这些是用户已确认的默认规则，正常执行已授权实验时不需反复询问同样的流程选择。
- 若当前请求仅为生成命令、检查环境或写文档，不能因此启动实验、清理 farm 或更改已有作业。
- 报告实际完成的步骤、Git commit、命令数、worker 数/token 预算、job ID、服务器结果位置、本地下载位置及失败/缺失项。没有执行的步骤明确标为未执行。
