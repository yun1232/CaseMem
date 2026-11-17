# CaseMem

CaseMem 医学视觉问答实现，使用 VeRL 运行 GRPO。项目包含图像级数据排重、错误驱动记忆库、视觉区域检索、训练定位奖励、定点裁剪推理和实验统计。

## 代码入口

| 路径 | 功能 |
| --- | --- |
| `casemem/build.py` | 找出原图答错的训练案例，经教师提议、Grounding DINO 精炼、原模型重答验证后入库 |
| `casemem/memory.py` | 聚合文本记忆簇，保存局部裁剪、归一化框及整图和局部特征 |
| `casemem/retrieval.py` | 文本召回、整图重排、高斯空间先验、原型聚合、归一化匹配与双阈值激活 |
| `casemem/verl_reward.py` | CaseMem 的答案奖励与定位奖励 |
| `casemem/pipeline.py` | GRPO Parquet、推理流程与 Fig. 2 的 `C/N` 统计 |
| `config/casemem.json` | 论文讨论后的配置及可修改的运行默认值 |

## 安装

建议使用 Python 3.11 或 3.12 和 CUDA 环境。先按所用 CUDA 版本安装有 GPU 支持的 PyTorch 与 vLLM。默认 Qwen3-VL 入口使用 VeRL v0.7.1，建议使用该版本的 vLLM CUDA 安装说明。对应源码包含 Qwen3-VL 的多模态训练适配。

```bash
# 在 CaseMem 根目录运行。VeRL 仅用于 GRPO 训练。
git clone --branch v0.7.1 --depth 1 https://github.com/verl-project/verl.git ../verl-v0.7.1
python -m pip install -e ../verl-v0.7.1 -e '.[data,models]'
python -m casemem --help
```

如果只进行建库或推理，可跳过 VeRL 的安装，安装 `.[data,models]` 和与当前 GPU 环境兼容的 vLLM 即可。模型权重、数据和 API 密钥由运行环境提供。

## 1. 数据准备

输入为 JSONL，每行一个问题。图像路径相对 JSONL 文件或为绝对路径。`answer_type` 可为 `open`、`closed` 或 `choice`。可在 `answers` 提供多个可接受的答案。选择题把选项已经拼入 `question`，并以选项字母或实际答案文本作为参考答案。

```json
{"id":"vqarad-001","dataset":"VQA-RAD","split":"train","image":"images/001.png","question":"Is there a pleural effusion?","answers":["no"],"answer_type":"closed"}
```

已有本地数据可直接生成此格式。也可用 `import-hf --parquet <local.parquet>` 导入本地数据分片。以 Hugging Face 上的 VQA-RAD 和 PathVQA 为例：

```bash
python -m casemem import-hf --dataset-id flaviagiammarino/vqa-rad --dataset-name VQA-RAD --hf-split train --split train --output-dir data/vqarad
python -m casemem import-hf --dataset-id flaviagiammarino/vqa-rad --dataset-name VQA-RAD --hf-split test --split test --output-dir data/vqarad
python -m casemem import-hf --dataset-id flaviagiammarino/path-vqa --dataset-name PathVQA --hf-split test --split test --output-dir data/pathvqa
python -m casemem prepare-splits --train data/vqarad/train.jsonl --test data/vqarad/test.jsonl data/pathvqa/test.jsonl --output-dir work/vqarad
```

`prepare-splits` 按解码后的图像像素排除跨 split 完全相同的图像，并产生 `split_audit.json`。有已定的验证集时传入 `--val`。省略时，从排除测试图像后的训练集按图像分组划出验证集。同一图像的所有问题保留在同一个 split。此检查覆盖解码后像素相同的图像，近重复图像需要按实验数据协议另行检查。

## 2. 构建固定记忆库

建库需要 `DASHSCOPE_API_KEY`。如账户使用其他地域端点，可设置 `DASHSCOPE_BASE_URL`。建库时，答错样本的图像和问题会发送给教师 API。

```bash
python -m casemem --config config/casemem.json build-memory \
  --train work/vqarad/train.jsonl --output work/vqarad/memory.zip
python -m casemem inspect-bank --bank work/vqarad/memory.zip
```

初始回答和裁剪后的重答由同一个 Qwen3-VL-8B 完成。Qwen3.5-Flash 只从当前图像和问题提议候选区域，Grounding DINO 精炼候选框，只有重答成功的候选才入库。每批结束后写出原子快照。中断后用相同的输入和配置追加 `--resume`，已处理问题会跳过。

## 3. GRPO

```bash
python -m casemem --config config/casemem.json prepare-grpo \
  --train work/vqarad/train.jsonl --val work/vqarad/validation.jsonl \
  --bank work/vqarad/memory.zip --output-dir work/vqarad/grpo
bash scripts/train_qwen3vl.sh --train work/vqarad/grpo/train.parquet \
  --val work/vqarad/grpo/validation.parquet --output runs/casemem --gpus 4
bash scripts/train_qwen3vl.sh --train work/vqarad/grpo/train.parquet \
  --val work/vqarad/grpo/validation.parquet --output runs/answer-only --gpus 4 --answer-only
```

训练样本只包含原图和问题。每题选出的伪框只写入 `extra_info`，该题的所有 rollout 共享同一个框。训练阶段排除记忆库中与当前查询像素相同的实例。奖励为 `R_task + lambda_loc * IoU`。IoU 仅在成功激活且预测框有效时计入，缺框时仍计算答题奖励。答案奖励对照使用同一 Parquet 与输出模板，并将 `lambda_loc` 设为 0。

默认训练为 200 个更新步、每批 16 题、每题 8 条 rollout。GPU 数和推理显存比例可以按机器调整。使用 `--dry-run` 可预览下发给 VeRL 的完整配置。Hugging Face 格式的模型保存于输出目录中的 `global_step_200/actor/huggingface`（步数随配置变化）。

## 4. 推理、消融和跨域检索

直接回答与冻结骨干的裁剪增强。

```bash
python -m casemem evaluate --data data/vqarad/test.jsonl --output-dir results/direct
python -m casemem evaluate --data data/vqarad/test.jsonl --bank work/vqarad/memory.zip --output-dir results/base
```

完整版和关闭测试裁剪的消融。使用定位格式训练的 checkpoint 时传入 `--predict-box`。

```bash
checkpoint=runs/casemem/global_step_200/actor/huggingface
python -m casemem evaluate --data data/vqarad/test.jsonl --bank work/vqarad/memory.zip \
  --model "$checkpoint" --predict-box --output-dir results/full
python -m casemem evaluate --data data/vqarad/test.jsonl --model "$checkpoint" \
  --disable-crop --predict-box --output-dir results/full-no-crop
```

Fig. 2 在两个测试集上使用同一个 VQA-RAD 记忆库。对各测试集分别用 `--strategy spatial`、`--strategy cosine` 和 `--strategy full` 运行。

```bash
python -m casemem evaluate --data data/pathvqa/test.jsonl --bank work/vqarad/memory.zip \
  --strategy full --output-dir results/pathvqa-ood-full
```

每次评估输出 `predictions.jsonl` 和 `metrics.json`。其中 `success` 表示通过所用策略的检索激活阈值，`fail` 表示未激活。各组的 `N` 为题数，`C` 为答对题数。`amortized_online_seconds_per_query` 是在线批耗时除以样本数，覆盖检索、裁剪和生成，不计初始化、图像文件读取及离线建库。可用 `--warmup 1` 跳过首批预热开销。

## 配置口径与验证

- 纸面超参数集中在 `config/casemem.json`。教师候选数、Grounding DINO 阈值、语义簇描述上限、余弦消融阈值、训练学习率、答案评分和输出 token 上限属于实现默认值。对照已有论文结果时，按原始配置和评分协议调整。默认开放题用规范化精确匹配，也可配置 token F1 与正确阈值。评估指标与训练数据按同一评分设置生成。
- 图像局部定位在 512 网格计算，最后将框映射到原图像素。`<bbox>` 使用原图的 0--1000 归一化坐标。候选 patch 特征与加权聚合后的原型分别归一化后再点积。
- CPU 测试使用可控的模型替身，覆盖错例验证、检索、归一化、伪框、裁剪开关和 Parquet。已检查定位奖励版和仅答案奖励版的训练配置可由 VeRL 的 Hydra 配置解析。此仓库快照未包含 GPU 训练产物或论文结果复现日志。

```bash
python -m pip install -e '.[data]'
python -m unittest discover -s tests -v
```
