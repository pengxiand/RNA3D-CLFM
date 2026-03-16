# RNA3D-CLFM 下游执行计划（中文内部）

本文件聚焦你说的“下游怎么做”，即在 two-tower 方向上快速形成可提交实验链路。

## 1. 下游任务定义

下游主任务：
1. active/decoy 区分（分类）
2. pocket 内排序（ranking）
3. 可选 affinity score 回归
4. residue-level site 预测

统一要求：
- 同一套 encoder 主干支持以上任务
- 不再依赖固定 2.5D 表示

## 2. 推荐下游模型组合（第一版）

- RNA tower: GVP-GNN
- Ligand tower: D-MPNN
- Shared space: projection z_rna, z_lig
- Heads:
  - rank head
  - decoy separation head
  - optional docking/affinity regression head
  - site head

## 3. 下游训练阶段

Stage A（表征预热）
- 数据: simulated + decoy
- Loss: contrastive + decoy

Stage B（亲和与排序）
- 数据: docking + binary + decoy
- Loss: contrastive + ranking + decoy (+ optional affinity regression)

Stage C（位点联合）
- 数据: pocket node labels (+ optional Stage B mixed mini-batch)
- Loss: site + weak pair consistency

## 4. Loss 配置建议

总损失:
L = wc * L_contrastive + wr * L_rank + wd * L_decoy + ws * L_site + wa * L_affinity

建议初始权重:
- wc = 1.0
- wr = 1.0
- wd = 1.0
- ws = 0.6
- wa = 0.3

说明:
- 先保证 contrastive/rank/decoy 三件套稳定，再逐步拉高 site/affinity 权重。

## 5. Decoy 课程学习

按阶段增加难度和数量：
1. Stage A: 20 negatives/positive
2. Stage B: 50 negatives/positive
3. Stage C: 100 negatives/positive（压力测试）

负样本来源建议：
- 同 pocket（局部难负）
- 同 family 近邻 pocket（几何相似）
- 全局池（远负样本）

## 6. 最小可复现命令

1. 先准备 manifest
- bash scripts/bootstrap_data.sh

2. 跑当前基础 scaffold
- python scripts/train_unified_multitask.py --config configs/unified_multitask.yaml

3. 跑 two-tower 目标配置（下游 profile）
- python scripts/train_unified_multitask.py --config configs/two_tower_gvp_dmpnn.yaml

说明:
- 第 3 步是你要的“下游主配置”，用于统一实验协议和 trainer 对接。

## 7. 必做下游实验表

A. 模型消融
1. RNA: GVP vs EGNN
2. Ligand: D-MPNN vs AttentiveFP
3. Loss: contrastive only vs contrastive+rank vs full

B. split 协议
1. ligand scaffold split
2. RNA family split
3. scaffold+family hard split

C. 指标
1. 分类: ROC-AUC, PR-AUC, MCC
2. 排序: EF1%, NDCG, Spearman
3. 位点: node AUROC, AUPRC
4. 稳定性: 3 seeds mean/std

## 8. 近期落地优先级

1. 优先把 two_tower_gvp_dmpnn 配置跑通（日志和 checkpoint 完整）。
2. 增加 decoy curriculum 和 hard split 实验。
3. 再上 EGNN 与 AttentiveFP 消融。

达到这三步，基本就具备一版投稿级核心证据链。
