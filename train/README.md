# Qwen3-VL + TOMATO 稀疏注意力训练

在 **TOMATO** 上训练 **稀疏注意力** 模块（**冻结** Qwen3-VL 全部 backbone 权重）：

- **指定一层**（`--train_layer_id`，默认 0）、**每个 attention head** 各有一组可训练参数，默认形状 **`(16384,)`**（16K，`--rel_pos_buckets`）
  - 第 `d` 维 = 距离 \(d\) 上的可训练系数 \(f(d)\)；student pre-softmax = **`f(d) × Q_pre·K_pre / √d + mask`**（RoPE **之前**的 Q/K），teacher = **`RoPE(Q)·RoPE(K) / √d + mask`**
  - 初始化：近距离偏大 `exp(-d/τ)` + 随距离衰减的正弦震荡（每 head 相位不同）；也可用 `baseline_relpos_scores.pt`（从全量 attention dump 按距离打包）
  - `d=0`：同位置；`d=1`：相距 1；…（不是每层共用一个参数）
- 对 query 位置 `p`，使用所有合法历史位置 `0..p` 的相对位置分数（不再做 top-k 截断）
- 默认损失（`run_train.sh`）：仅 **蒸馏 MSE**（pre-softmax 分数；与全量 teacher 同为因果下三角 mask `k<=q` ∩ `attention_mask`），无语言建模 CE

数据与 prompt 与 `lmms_eval` 的 `tomato` 任务、`examples/models/qwen3vl.sh` 评测配置对齐。

## 依赖

在已安装 `lmms-eval` 的环境中（需 `transformers>=4.57`、`qwen-vl-utils`）：

```bash
pip install -r train/requirements-train.txt accelerate
```

## 数据

- 数据集：`lmms-lab/TOMATO`（与 `tomato.yaml` 一致）
- 视频缓存目录：`$HF_HOME/TOMATO/`（与评测相同）
- 首次使用需能访问 Hugging Face（脚本里已配置代理）

```bash
export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412

python -c "from datasets import load_dataset; load_dataset('lmms-lab/TOMATO', split='test', token=True)"
```

TOMATO 仅有 `test` split；训练脚本默认用其中 90% 做训练、10% 做验证。

## 训练

默认用 **Accelerate 单卡启动**（`train/accelerate_single_gpu.yaml`，`mixed_precision: bf16`）：

```bash
bash train/run_train.sh
```

不用 Accelerate CLI 时：

```bash
USE_ACCELERATE=0 bash train/run_train.sh
```

**说明（单卡）**：非 sparse 层默认 `flash_attention_2`；仅 `train_layer_id` 那一层走自定义 sparse forward + teacher。想更快可减小 `--num_frames` / `--max_pixels`。

调试（小样本）：

```bash
bash train/run_train.sh --limit 8 --num_train_epochs 1 --save_steps 50
```

权重输出：`train/outputs/qwen3vl-sparse-attn/final/`
  - `sparse_rel_pos_bias.pt`（含 `_meta.train_layer_id` 与 `layer_{id}.head_{h}`）
  - `sparse_rel_pos_layer{id}.pt`（同内容别名，便于辨认层号）

## 训练后再评测

使用训练好的纯位置 attention（`sparse_rel_pos_bias.pt`），与 `examples/models/qwen3vl.sh` 对齐的 TOMATO 设置：

```bash
# 默认权重：/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt
bash examples/models/qwen3vl_sparse_attn.sh

# 或指定路径
SPARSE_REL_POS=/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt \
BASE_MODEL=/path/to/Qwen3-VL-4B-Instruct \
bash examples/models/qwen3vl_sparse_attn.sh
```

模型名：`qwen3_vl_sparse`（默认 `attn_implementation=flash_attention_2`；仅 `sparse_layer_id` 层 decode 用 rel-pos）。基座仍为原始 `pretrained`。

## 主要参数（与评测对齐）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model_path` | 本地 4B 路径 | 同 `pretrained=` |
| `--max_pixels` | 12845056（`run_train.sh`） | 同 `model_args` 的 `max_pixels` |
| `--rel_pos_buckets` | 16384 | 每 head 距离系数长度 |
| `--learning_rate` | 1（`run_train.sh`） | 仅 rel-pos 可训练；建议扫多组对比 |
| `--num_frames` | 16（`run_train.sh`） | 同 `max_num_frames`；`min_pixels` 自动为 3136 |
| `--train_ratio` | 0.9 | test 集划分训练/验证 |
| `--rel_pos_init_path` | 无 | 可选：从 `.pt` 加载 `layer_{i}.head_{h}` 初始化 rel-pos（训练内不做统计） |

## Todo

- [ ] **代码检查**：系统检查训练/推理路径是否一致（数据预处理、attention patch、生效参数、保存与加载逻辑）。
- [ ] **学习率对比实验**：固定其余超参，扫几组 `learning_rate`，观察 pre-softmax 分数 MSE 收敛与评测指标。
- [ ] **学习率复核**：当前学习率峰值为 `1`，结合是否 `resume_from_checkpoint` 与 warmup 配置，核对实际 step-level 学习率曲线。
- [ ] **推理精度与注意力分析**：评估训练后推理精度，并统计“纯位置打分”下注意力分数分布与占比（当前非 top-k 截断）。
- [ ] **过拟合风险评估**：对比 train/eval loss 走势与最终指标，判断是否过拟合，并给出正则化或早停建议。
- [ ] **长度配置验证**：当前相对位置参数长度为 `4K`，常见输入序列约 `512`；后续验证有效桶利用率、长距离桶是否长期闲置，以及是否需要缩短参数长度或改桶策略。
- [ ] **样本级输入核对**：检查第一个训练样本的完整输入内容（system/user/video/assistant）、实际采样帧数，以及 `labels != -100` 的 token 区间（即真正参与 loss 计算的位置）。

---

## Version 2

1. **分数**：用 softmax **之前**的注意力分数；注意**当前序列长度**（有效距离范围）；看**训练**和**校准（评测采集）**是否差不多；不同长度下曲线规律是否不同。

2. **初始化、学习率、loss**（待调/待记）。

3. **评测**
   - 可视化曲线；
   - 计算排序指标；
   - 按分数选 top-k，看占**真实注意力分数**多少；
   - 多个数据集上看推理精度（**prefill 保持全量注意力**）。

4. **三种曲线 / 相关性**（具体定义待补）。

---

## Version 3

1. 多模态评测里不少数据集是多张图，常见流程是 **一次 prefill、一次 decode**；若文本很长，decode 会很慢。先在多模态场景做初步验证，通过后再做下一步。

2. 若稀疏放在 **prefill 阶段**，训练或校准时不能只盯最后一个 token，**其余位置也要一并考虑**。

3. 当前权重与评测指标是否不应只看 **decode 的第一个 token**？应看 **prefill 的最后一个**，还是 **prefill 的多个位置**？（待统一）

（或者多种组合方式？）

---

## Version 4

1. Student 分数改为 **`f(d) × Q_pre·K_pre / √d + mask`**（RoPE 前的 Q/K；Teacher 仍为 RoPE 后标准 QK）。

2. 对比不同 **初始化**（默认 prior / baseline dump）与 **学习率** 设置。

3. 多模态评测：**prefill 用稀疏 pre-softmax**，**decode 用全量 RoPE QK**。

4. 先在 layer 0 上把训练方式定下来（公式、init、lr、prefill/decode 分工与评测采集），再按同样流程训练其他层。
