**Brown–Haugh martingale penalty：当前工作树代码审查**

审查对象是当前工作树，包括已暂存和未提交改动；基准 HEAD 为 `f97d1a68644de62fc746c39629e31e00fc3f34eb`。范围覆盖 generating function、路径与 proposal、训练 master/子问题、Benders 停止条件、策略决策、IR 求值、评估数据和缓存。没有修改生产代码；本目录只包含审查材料。未运行整个仓库的全部实验。

**结论：时间权重层部分支持 Brown–Haugh，现有内置生成函数不等同于论文的状态价值函数形式；而且实际期望项存在错误，目前不能为任意训练系数宣称 martingale 或弱对偶保证。** `AbsorptionLinearPenaltyFunction` 比 legacy 更接近论文的吸收时间权重结构，但类名和一致性测试不能代替条件期望验证。

论文核对来源：Brown, D. B. & Haugh, M. B. (2017), *Information Relaxation Bounds for Infinite Horizon Markov Decision Processes*, Operations Research 65(5), 1355–1379，[DOI](https://doi.org/10.1287/opre.2017.1631)、[作者公开全文](https://martin-haugh.github.io/files/Research/InfoRelaxation_OR.pdf)。已检查正文和公式所在页面的渲染图。

**定义与代码的对应。** 论文 p.1358 的式 (2) 使用状态函数产生折扣 martingale 差分；p.1359–1360 的式 (6)–(8) 改写为吸收时间形式，且吸收态函数值为零；p.1364 的式 (15) 给出 change-of-measure 权重，并区分覆盖条件与 subsolution 条件。以下用 `b` 表示实际调度动作、`a` 表示待训练系数，避免与代码的 `theta_vars` 混淆。

原始折扣形式的单步项为

\[
\pi_t=\gamma^{t+1}\{\mathbb E[v(f(s_t,b_t,\Delta))\mid\mathcal F_t]-v(s_{t+1})\}.
\]

对当前代码所采用的、仅改变路径长度分布的形式，设 `L` 为 arrival 数量，`L+1` 为决策期数；用零起始索引，`W_t = gamma^t / P_q(L >= t)`。在仍有下一次到达时，代码累计

\[
\pi_t^{code}=\gamma W_t\bar g_a(s_t,b_t)-W_{t+1}g_a(s_t,b_t,\Delta_{t+1}).
\]

| 检查项 | 当前行为 | 判断 |
|---|---|---|
| 正负号 | 成本加“期望项减实现项” | 与最小成本 IR 的方向一致 |
| 已存活路径权重 | `gamma * W[t]`、`W[t+1]` | 通常的正参数、正确采样/加权情形下，满足所需的逐期抵消关系 |
| 真正吸收的最后一期 | 保留 `gamma * W[t] * E[g]`，实现项为 0 | 吸收边界的结构正确 |
| 人工截断的最后一期 | 两项都删除 | 是有限前缀的构造；不能等同于论文保留价值函数修正的 subsolution 截断方案 |
| 生成函数 | `g_a(s,b,delta) = sum(delta) * a.T @ phi0(s,b)` | 一般不是同一个 `v_a(s_next)` |
| 条件期望 | 使用未截断 Poisson 均值 | 不等于真实采样分布下的期望，破坏 martingale 条件 |
| 一致性 | absorption 的训练、hindsight、评估共用 form | 架构上有利；不证明这个共用公式本身满足理论 |
| 默认模式 | 默认仍是 `linear_penalty` / `LegacyForm` | 使用时必须确认具体 spec；legacy 不应称作论文式 (6) 的实现 |

权重和最后一期的实现见 [penalty_forms.py:66](/Users/yanboding/Desktop/PhdTopic/MCMADASP/generating_function/penalty_forms.py:66)，统一路径构建见 [approximate_q_agent.py:188](/Users/yanboding/Desktop/PhdTopic/MCMADASP/decision_maker/approximate_q_agent.py:188)。

**两个核心数学反例。** 以下均通过真实 toy 环境验证，不依赖 Monte Carlo 误差。

第一，toy 的到达总量服从 Poisson(3) 条件于不超过 9。枚举 `get_system_dynamic()`，实际均值为 `2.991889546526636`，代码期望项使用 `3.0`。选取当前预约第一坐标为 1、其他坐标为 0、零动作，以及仅该特征系数为 1，则在目标吸收概率下：

\[
\mathbb E[\pi_t\mid\mathcal F_t]
=0.99(3-2.991889546526636)
=0.008029348938630393\ne0.
\]

第二，构造两个合法状态，只让今天的既有预约 `u[0]` 分别为 1 和 0，动作均为零。状态转移会丢弃今天的预约，因此对于**每一个**可能的 arrival，两者下一状态均相同。取上述系数，arrival 总数为 1 时，代码的 `g` 却分别为 1 和 0。即使修正均值，在非吸收的一步上，两者 penalty 仍分别为 `1.9619706510613695` 和 `0`。因为下一状态的整个条件分布相同，这不可能由原状态空间上的同一个 `v(s_next)` 产生，也不是加一个可抵消的确定项就能解决。

这不表示状态–动作–噪声生成函数永远不合法：在期望精确、采样权重一致且满足可积性条件时，它也能产生一般 martingale 差分。但其合法性要另行论证，不能直接继承论文关于状态价值函数、理想 penalty 或 subsolution 改进的全部结论。

**已验证的实现问题，按优先级排列。** P1 表示应先处理的正确性问题；P2 表示特定配置或调用路径中的实际错误。这里包括原有问题及本次重构相关问题，未把所有问题归因于本次改动。

1. **[P1] 条件期望与到达分布不一致。** [arrival_generator.py:37](/Users/yanboding/Desktop/PhdTopic/MCMADASP/environment/arrival_generator.py:37) 把名义均值当作截断分布均值，[penalty_function.py:32](/Users/yanboding/Desktop/PhdTopic/MCMADASP/generating_function/penalty_function.py:32) 将其带入所有 penalty 期望项。上面的枚举已证明非零条件均值，因此一般的 dual feasibility 不成立。应从 `truncate_poisson_pmf` 计算 `sum(n * pmf[n])`，再按类型概率分配；如其他代码还需要名义参数，应把两种均值明确区分。修复后重新训练并评估，旧系数与旧结果不可直接视为同一实验。

2. **[P1] mixture 缺失分层时改用等权，造成带偏估计。** [datasets.py:438](/Users/yanboding/Desktop/PhdTopic/MCMADASP/param_generation/datasets.py:438) 吞掉分层权重异常，但仍使用完整 mixture 的 survival 分母。`gamma=.99, q=.5, lambda_0=.1, N=4` 实际只生成短路径层，记录的 `path_weight/path_stratum` 都为 `None`。在这一实际分布下，单位逐期成本估计量的期望约为 `5.117609146652267`，目标却为 `100`；常数生成项的 penalty 均值约为 `0.9488239085334773`。应拒绝缺层分配，或重新构造能覆盖所有非零质量分层的分配与权重。

3. **[P2] 评估缓存忽略 terminal 和 period_weights。** [run.py:242](/Users/yanboding/Desktop/PhdTopic/MCMADASP/run.py:242) 的签名缺少这两项，而 [datasets.py:476](/Users/yanboding/Desktop/PhdTopic/MCMADASP/param_generation/datasets.py:476) 的记录 UID 也未包含它们。缓存命中还早于路径验证。真实单期复现：absorbed 的 penalty 为 `2.97`；用相同 state/arrivals 改为 truncated 后返回旧值 `2.97`，新缓存目录计算的正确截断结果为 `0.0`。应在缓存和 checkpoint 标识中纳入终止/权重语义，并拒绝或版本隔离旧缓存。

4. **[P2] q=0 的实际有限支撑被标成 ABSORBED。** [base.py:102](/Users/yanboding/Desktop/PhdTopic/MCMADASP/importance_sampling/proposals/base.py:102) 及 [geometric.py:62](/Users/yanboding/Desktop/PhdTopic/MCMADASP/importance_sampling/proposals/geometric.py:62) 没有处理实际支撑终点。`GeometricLengthProposal(0)` 和 `TruncatedGeometricLengthProposal(0,5)` 都必然生成 L=1；`gamma=.9, E[g]=g=1` 时产生确定的正 penalty `0.81`。应拒绝不支持继续采样的无限期配置，或按真实有限支撑标记截断，而非保留最后一个期望项。

5. **[P2] 固定初始化点的 cut gap 被用于判断全局最优。** [benders_decomposition_solver.py:536](/Users/yanboding/Desktop/PhdTopic/MCMADASP/metaheuristic_algorithm/benders_decomposition_solver.py:536) 固定首轮动作，但 [line 641](/Users/yanboding/Desktop/PhdTopic/MCMADASP/metaheuristic_algorithm/benders_decomposition_solver.py:641) 仍按零 gap 停止。现有 `test_benders_cut_purging.build_solver()` 在 `init_solution=zeros(4)` 时一轮返回 `2.25`；不固定初值时五轮达到 `2.5`。固定点处的 cuts 紧只能证明该点求值准确。必须先求解不固定动作的 master，才能给出全局停止证书。当前训练入口把 init_solution 设为 None，因此该缺陷主要影响通用 solver 的显式初值调用。

6. **[P2] min-norm 二次求解混用目标值，且使 X 不可读取。** [benders_decomposition_solver.py:789](/Users/yanboding/Desktop/PhdTopic/MCMADASP/metaheuristic_algorithm/benders_decomposition_solver.py:789) 保留首次目标，却返回允许相对 slack 的二次求解中的 epigraph 值；[line 1023](/Users/yanboding/Desktop/PhdTopic/MCMADASP/metaheuristic_algorithm/benders_decomposition_solver.py:1023) 恢复模型并 update 后使解属性失效。单场景 `Q(a)=a, 0<=a<=2e6, tol=1e-6`：报告 evaluated_value=`2000000`、gap=`0`，实际评估动作=`1999999.998`，实际差距约 `.002`；随后读取 `action_vars.X` 报错。应保存被评估的动作快照和对应原始目标，把二次求解损失纳入证书，并让训练从快照取系数。

7. **[P2] 同一 agent 再训练会忽略新 regularization，却报告新配置。** [approximate_q_agent.py:536](/Users/yanboding/Desktop/PhdTopic/MCMADASP/decision_maker/approximate_q_agent.py:536) 只在首次构建目标；[line 610](/Users/yanboding/Desktop/PhdTopic/MCMADASP/decision_maker/approximate_q_agent.py:610) 总是使用最新参数描述结果。实测 L2 lambda=.001 后改 L1 lambda=1e6、scale=none，目标仍是 QuadExpr，系数范数约 `107.37508551`，SAA 都为 `9862.72597304158`；metadata 却称巨大 L1，scale 仍不是全 1。新建 agent 的巨大 L1 测试能使系数归零。应更新/重建或显式拒绝不兼容重训练；coefficient_bound/init_state 同样受缓存限制。

8. **[P2] 更新系数后策略仍使用旧目标。** [approximate_q_agent.py:159](/Users/yanboding/Desktop/PhdTopic/MCMADASP/decision_maker/approximate_q_agent.py:159) 缓存 decision model，训练完成后 `set_coefficients` 没有使它失效；hindsight workers 也在构建时捕获系数。实测 waitlist 系数由 0 改为 -1000：缓存 agent 返回目标 `0`、安排 3 人；新建 agent 返回 `-2910`、安排 0 人。应在系数/penalty_ratio/生成函数改变时重建或更新所有相关模型。

9. **[P2] 一次 fixed-action 求值会锁住之后的自由求解。** [approximate_q_agent.py:164](/Users/yanboding/Desktop/PhdTopic/MCMADASP/decision_maker/approximate_q_agent.py:164) 固定变量上下界，却未在 `action=None` 时恢复。toy 中先固定零动作，再自由调用，目标仍为 `6000`、安排 0 人；新 agent 的自由目标为 `0`、安排 3 人。应使用临时固定并恢复界限的上下文。

10. **[P2] 公共 solve 修改动作后仍返回修改前目标。** [approximate_q_agent.py:813](/Users/yanboding/Desktop/PhdTopic/MCMADASP/decision_maker/approximate_q_agent.py:813) 无条件执行 regular-first repair，甚至改写调用者传入的、满足当前优化模型约束的 fixed action。零系数复现：输入 overtime `[3,0,...]`，返回 `[0,0,...]`，报告目标 `4300`，但返回动作成本为 `4000`。该输入不在只枚举 regular-first overtime 的 `env.valid_actions` 输出中，因此此反例证明的是接口/求值不一致，而不是主张它属于预期 MDP 的全部合法动作。若 repair 是预期策略行为，应明确分别返回求解目标和执行动作求值；显式固定动作应遵守接口约定或被明确拒绝。现有 hindsight 一致性测试绕过了公共 wrapper。

11. **[P2] extensive-form 入口调用不存在的方法。** [run.py:984](/Users/yanboding/Desktop/PhdTopic/MCMADASP/run.py:984) 在 `PENALTY_TRAIN_SOLVER=extensive` 时调用 `agent.extensive_form_train`，当前类和父类均无此方法，实际 `hasattr(ApproxQAgent, 'extensive_form_train') == False`。应恢复对应实现或移除/明确拒绝该选项。

12. **[P2] 显式 mixture lambda_0=1 被错误当作未提供 proposal。** [proposals/__init__.py:50](/Users/yanboding/Desktop/PhdTopic/MCMADASP/importance_sampling/proposals/__init__.py:50) 返回 None，随后 [datasets.py:408](/Users/yanboding/Desktop/PhdTopic/MCMADASP/param_generation/datasets.py:408) 拒绝 absorption 评估。已通过实际数据生成入口复现：提供 `.99/.95/1.0` mixture spec 仍报需要 `--eval-proposal`。应返回能携带终止/权重信息的具体默认 proposal，或在评估端正确处理其语义。

**还需要补齐的理论和统计边界。** 以下与已复现的实现错误分开，不表示所有配置一定失败。

- 人工截断目前删除最后一期 penalty，并省略后续成本。修正期望后，可按“有限前缀的零均值 penalty + 非负尾部成本”论证下界；这不是论文 subsolution 截断定理的直接实现。若要支持可负成本、任意 reformulation 或论文中的 bound-improvement 结论，需另行满足相应条件。
- 当前 waitlist 可能无全局界，不能仅凭系数有限就声称生成函数有界。对现有 bounded arrivals 和线性特征，可以尝试用随时间至多线性增长及折扣可和性证明一阶可积性；应把这个证明写清楚。
- 普通 geometric proposal 若 `0<q<=gamma^2`，即使单位逐期成本的 survival-weighted 估计量也可能具有无限二阶矩。有效均值与可用的高斯/t 置信区间是不同问题。mixture 保留目标尾部能缓解这一问题。
- mixture 使用分层的 scrambled Sobol。当前 `StratifiedRunningStats` 的组内 `s^2/n` 不是经验证的 RQMC 误差估计；而 [stratified_running_stat.py:62](/Users/yanboding/Desktop/PhdTopic/MCMADASP/utils/stratified_running_stat.py:62) 对仅有一个样本的层直接忽略方差，不能解释为该层没有不确定性。建议独立 scrambles/replications 后估计误差。
- 训练返回的是选择系数后的 in-sample SAA 值。它不是每次都保证低于真实最优值的确定性下界。应固定训练好的系数后使用独立评估样本，并区分 Benders 优化 gap、统计误差和截断误差。数据生成中训练/评估种子分离是正确方向。

**若要严格增加论文状态价值函数版本，建议保持现有两种模式，再新增明确命名的实现。** 定义 `v_a(s)=a.T @ psi(s)`；`features(s,b,delta)` 返回 `psi(env.get_next_state(s,b,delta))`，`expected_features` 计算同一转移分布下的精确期望，吸收态取 0。对 affine 状态特征可以继续保持 LP 子问题和现有 Benders cut 结构。选择更复杂 psi 时，需要重新核实子问题可解性及全局最优求值条件。一般状态–动作–噪声版本可保留，但应给出独立的 martingale/可积性论证和准确命名。

**验证记录。** 两组审查运行去重后覆盖 45 个 pytest 测试，均通过：

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  test/test_benders_regularization.py test/test_agent_regularization.py \
  test/test_benders_termination.py test/test_penalty_forms.py test/test_sample_path.py
# 25 passed

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  test/test_benders_cut_purging.py
# 3 passed

python -m pytest -q test/test_sample_path.py test/test_penalty_forms.py \
  test/test_importance_sampling_proposals.py test/test_stratified_path_weights.py \
  -k 'not monte_carlo_zero_mean_along_myopic_rollouts'
# 28 passed, 1 deselected
python -m pytest -q test/test_penalty_forms.py -k monte_carlo_zero_mean_along_myopic_rollouts
# 1 passed, 4 deselected

PYTHONDONTWRITEBYTECODE=1 python -m test.test_penalty_builder
# 12 个集成/回归检查完成；All penalty-builder tests passed.
```

完整 builder 日志见 [penalty_builder.log](/Users/yanboding/Desktop/PhdTopic/MCMADASP/outputs/code_review/penalty_builder.log)。上述通过不覆盖本报告反例：现有 Monte Carlo 零均值测试仅使用较宽的随机置信区间；另一些测试只比较共享实现的两条路径。应补充精确枚举的条件均值检查、相同下一状态的状态函数一致性检查、人工截断/真吸收分支、异常采样层分配，以及两个 Benders 停止反例。

无需 Gurobi 的核心复现脚本：[reproduce_brown_penalty.py](/Users/yanboding/Desktop/PhdTopic/MCMADASP/outputs/code_review/reproduce_brown_penalty.py)，本次结果：[core_reproduction.json](/Users/yanboding/Desktop/PhdTopic/MCMADASP/outputs/code_review/core_reproduction.json)。策略缓存/动作复现：[reproduce_policy_cache.py](/Users/yanboding/Desktop/PhdTopic/MCMADASP/outputs/code_review/reproduce_policy_cache.py)、[reproduce_terminal_cache.py](/Users/yanboding/Desktop/PhdTopic/MCMADASP/outputs/code_review/reproduce_terminal_cache.py)。审查源文件哈希见 [source_hashes.json](/Users/yanboding/Desktop/PhdTopic/MCMADASP/outputs/code_review/source_hashes.json)。

建议优先修复 R1、R2、R3、R4，并验证固定非预见策略的精确零均值；接着修复 R5–R7 的训练证书与缓存，再决定是否新增严格的状态价值函数版本。只有这些条件落实后，才能据此重新评估旧实验中的 lower-bound 解释。
