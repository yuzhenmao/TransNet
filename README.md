# TransNet

This is the pytorch implementation of [_**Augmenting Knowledge Transfer across Graphs**_](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10027706) (ICDM 2022).

We introduce a novel notion named trinity signal that can naturally formulate various graph signals at different granularity (e.g., node attributes, edges, and subgraphs). With that, we further propose a domain unification module together with a trinity-signal mixup scheme to jointly minimize the domain discrepancy and augment the knowledge transfer across graphs. Comprehensive empirical results show that _**TransNet**_ outperforms all existing approaches on seven benchmark datasets by a significant margin.

![image (5)](https://user-images.githubusercontent.com/57878927/190012138-2c54f3b8-9179-4159-9d35-45b8653bbed8.png)


### Datasets
Here, we only provide datasets M2 and A1. Please download other datasets from the original papers listed in our paper.

### Implementation Details
_**TransNet**_ is firstly pre-trained on the source dataset for 2000 epochs; then it is fine-tuned on the target dataset for 800 epochs using limited labeled data in each class. We use Adam optimizer with learning rate 3e-3. α in the beta-distribution of trinity-signal mixup is set to 1.0 and the output dimension of MLP in domain unification module is set to 100 by default. Precision is used as the evaluation metric.

### Demo case: Task (M2 -> A1) & (A1 -> M2)
```
 python ./src/transnet.py --datasets='M2+A1' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

# Reference

Please cite the following paper if you found this library useful in your research:

### Augmenting Knowledge Transfer across Graphs
[Yuzhen Mao](https://scholar.google.com/citations?user=9wKn1A0AAAAJ&hl=en), [Jianhui Sun](https://jsycsjh.github.io/), [Dawei Zhou](https://sites.google.com/view/dawei-zhou/home)\
*IEEE International Conference on Data Mining (ICDM)*, 2022

```
@inproceedings{mao2022augmenting,
  title={Augmenting Knowledge Transfer across Graphs},
  author={Mao, Yuzhen and Sun, Jianhui and Zhou, Dawei},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)},
  pages={1101--1106},
  year={2022},
  organization={IEEE}
}
```
