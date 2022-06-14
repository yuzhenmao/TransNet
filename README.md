# TransNet

For dataset, please download from the orginal papers listed in our paper.

```
 python ./src/transnet.py --datasets="C+Wdblp" --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.05  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01  --only=True
```
