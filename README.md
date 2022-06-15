# Augmenting Knowledge Transfer across Graphs

This is the pytorch implementation of _**TransNet**_.

We introduce a novel notion named trinity signal that can naturally formulate various graph signals at different granularity (e.g., node attributes, edges, and subgraphs). With that, we further propose a domain unification module together with a trinity-signal mixup scheme to jointly minimize the domain discrepancy and augment the knowledge transfer across graphs. Comprehensive empirical results corroborate our theoretical findings and show that _**TransNet**_ outperforms all existing approaches on seven benchmark datasets by a significant margin.

### Datasets
Here, we only provide datasets M2 and A1. Please download other datasets from the original papers listed in our paper.

### Demo case: Task M2 -> A1 & A1 -> M2
```
 python ./src/transnet.py --datasets="M2+A1 --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```
