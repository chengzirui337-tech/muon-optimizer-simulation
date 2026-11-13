# Muon optimizer simulation

Muon 优化器课程调研的可复现实验代码，使用 NumPy 实现精确极因子和有限步矩阵多项式更新。包括三个实验：矩阵二次目标的理论上界核验、近似正交化误差比较、3152 参数模型的合成下一符号预测。

## 运行

需要 Python 3.12。克隆或下载本仓库后执行：

```bash
python -m pip install -r code/requirements.txt
python code/simulate.py
python code/make_tables.py
```

脚本以自身所在目录确定输出位置，运行目录不限。所有输入数据均由程序生成，无须下载语料或模型。`results/`保存实测原始记录；运行后`figures/`生成四张矢量PDF及PNG预览。仅从已有记录重绘可运行`python code/plot_saved.py`。

## 实验设置

- 二次目标：24×12矩阵，种子7；T=100、400、1600、6400分别重新初始化，按理论公式选取各自的固定步长。
- 正交化误差：32×16矩阵，条件数10000，种子8；比较工程五次多项式与经典三次Newton–Schulz迭代。
- 合成预测：16符号、两个符号上下文，目标符号为`(a+2b) mod 16`，目标分布含0.1的均匀平滑质量。模型为32→64(tanh)→16(softmax)，共3152参数，不是自然语言或大模型预训练。

预测任务将256个不同上下文按种子2026划分为192/32/32个训练/验证/测试样本。每次800步、批量64；初始化种子0用于验证选参，每种方法七个候选。随后以11、22、33三个种子独立评估，各方法共享同一种子下的初始化和样本顺序。测试集不参与选参。

AdamW隐藏矩阵学习率候选为`[.001,.003,.01,.03,.1,.3,1]`；Muon两组为`[.005/3,.005,.015,.045,.135,.405,1.215]`。最终均选择了网格上边界，故结果不代表全局最优超参数下的排序。

隐藏矩阵分别使用AdamW、Muon-NS5或Muon-SVD，输出头和偏置在各组均使用峰值学习率0.003的AdamW，其动量为(0.9,0.99)、epsilon=1e-8。Muon使用heavy-ball动量0.95，尺寸缩放0.2√max(m,n)；矩阵衰减0.01，偏置不衰减。预热40步后余弦下降至峰值的10%。采用CPU、float64和单线程。

## 保存的结果

| 方法 | 训练交叉熵 | 测试交叉熵 |
|---|---:|---:|
| AdamW | 0.5735 ± 0.0004 | 1.716 ± 0.048 |
| Muon-NS5 | 0.5782 ± 0.0015 | 1.939 ± 0.262 |
| Muon-SVD | 0.5788 ± 0.0013 | 1.864 ± 0.085 |

数值为三个评估种子的均值±样本标准差（ddof=1），交叉熵单位为nats。绘图阴影也是一个样本标准差，不是置信区间。目标分布熵为0.565009 nats。本设置中Muon没有优于AdamW，不据此推断工业大模型训练的普遍优劣。

`quadratic.csv`记录理论上界核验；`orthogonalization.csv`记录多项式误差；`tuning.csv`包含全部21次候选试验；`predictor_runs.csv`和`predictor_history.csv`包含三种子结果与逐步曲线；`summary.json`包含环境和统计汇总。短时CPU计时仅记录复现成本，不作为GPU吞吐率基准。

## 文件与核验

`simulate.py`包含极因子对偶等式、零矩阵处理、手写梯度有限差分、数据划分不重叠和二次目标下降上界的核验。`plot_saved.py`只负责从CSV绘图，`make_tables.py`从汇总JSON生成LaTeX表格。

## 原始来源

- Keller Jordan, [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/).
- Bernstein and Newhouse, [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325).
- Li and Hong, [A Note on the Convergence of Muon](https://arxiv.org/abs/2502.02900v2).
- Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982).
- 官方参考实现：[KellerJordan/Muon](https://github.com/KellerJordan/Muon)、[MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight)。本仓库是独立教学仿真，不是上述官方训练工程。
