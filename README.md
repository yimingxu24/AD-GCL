# Revisiting Graph Contrastive Learning on Anomaly Detection from a Structural Imbalance Perspective

<img src="https://github.com/yimingxu24/AD-GCL/blob/main/framework.svg" width="100%">

## Code structure
| *folder*  |                         description                          |
| :-------: | :----------------------------------------------------------: |
|   Data    |      Datasets.       |
|   AD-GCL    | AD-GCL implementation code is provided. |

## Datasets

| **Dataset** |  # Nodes  |  # Edges  |  # Attribute  |  # Anomalies  |
| :---------: | :--------: | :-----: | :--------: | :--------: |
|  **Cora**   |  2,708  |  5,429  |  1,433  |  5.5%  |
|  **Citeseer**   |  3,327  |  4,732  |  3,703  |  4.5%  |
|  **Pubmed**   |  19,717  |  44,338  |  500  |  3.0%  |
|  **Bitcoinotc**   |  5,881  |  35,592  |  128  |  5.1%  |
|  **BITotc**   |  4,863  |  28,473  |  128  |  6.2%  |
| **BITalpha** |  3,219  |  19,364  |  128  |  9.3%  |


## Usage
```python
# Cora
python ./AD-GCL/run.py --dataset 'cora' --lr 5e-3 --num_epoch 200 --threshold 7 --gpu_id 0

# Citeseer
python ./AD-GCL/run.py --dataset 'citeseer' --lr 3e-3 --num_epoch 200 --threshold 6 --gpu_id 0

# Pubmed
python ./AD-GCL/run.py --dataset 'pubmed' --lr 4e-3 --num_epoch 100 --threshold 8 --gpu_id 0

# Bitcoinotc
python ./AD-GCL/run.py --dataset 'bitcoinotc' --lr 4e-4 --num_epoch 100 --threshold 8 --gpu_id 0

# BITotc
python ./AD-GCL/run.py --dataset 'bitotc' --lr 5e-4 --num_epoch 100 --threshold 7 --gpu_id 0

# BITalpha
python ./AD-GCL/run.py --dataset 'bitalpha' --lr 5e-3 --num_epoch 100 --threshold 8 --gpu_id 0
```


## Dependencies

- Python 3.8.13
- PyTorch 1.12.1
- dgl 0.4.3.post1
- Scipy 1.9.1
- Tqdm 4.64.1


## Reference
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{xu2025revisiting,
  title={Revisiting Graph Contrastive Learning on Anomaly Detection: A Structural Imbalance Perspective},
  author={Xu, Yiming and Peng, Zhen and Shi, Bin and Hua, Xu and Dong, Bo and Wang, Song and Chen, Chen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={12972--12980},
  year={2025}
}
```
