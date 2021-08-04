View-Consistent Heterogeneous Networks

# Experiments
- For different data sets and different label rates, we set different numbers of pseudo-labels and filtering strengths.
- The parameter setting and the command to run the code are as follows:

## Three heads
- For Cora with verification:
  - 0.5%: `python train.py --dataset cora --percent 0.005 --t1 200 --t2 300 --k 15`
  - 1.0%: `python train.py --dataset cora --percent 0.01 --t1 180 --t2 280 --k 13`
  - 3.0%: `python train.py --dataset cora --percent 0.03 --t1 140 --t2 240 --k 9`

- For Cora without verification:
    - 0.5%: `python train.py --dataset cora --percent 0.005 --fastmode True --t1 200 --t2 300 --k 15`
    - 1.0%: `python train.py --dataset cora --percent 0.01 --fastmode True --t1 180 --t2 280 --k 13`
    - 2.0%: `python train.py --dataset cora --percent 0.02 --fastmode True --t1 160 --t2 260 --k 11`
    - 3.0%: `python train.py --dataset cora --percent 0.03 --fastmode True --t1 140 --t2 240 --k 9`
    - 4.0%: `python train.py --dataset cora --percent 0.04 --fastmode True --t1 120 --t2 220 --k 7`
    - 5.0%: `python train.py --dataset cora --percent 0.05 --fastmode True --t1 100 --t2 200 --k 5`

- For Citeseer with verification:
    - 0.5%: `python train.py --dataset citeseer --percent 0.005 --t1 200 --t2 300 --k 15`
    - 1.0%: `python train.py --dataset citeseer --percent 0.01 --t1 180 --t2 280 --k 13`

- For Citeseer without verification:
    - 0.5%: `python train.py --dataset citeseer --percent 0.005 --fastmode True --t1 200 --t2 300 --k 15`
    - 1.0%: `python train.py --dataset citeseer --percent 0.01 --fastmode True --t1 180 --t2 280 --k 13`
## Eight heads
- For PubMed with verification:
    - 0.03%: `python train.py --dataset pubmed --percent 0.0003 --t1 1000 --t2 2000 --k 15`
    - 0.05%: `python train.py --dataset pubmed --percent 0.0005 --t1 1000 --t2 2000 --k 13`
    - 0.10%: `python train.py --dataset pubmed --percent 0.001 --t1 800 --t2 1800 --k 11`

- For PubMed without verification:
    - 0.03%: `python train.py --dataset pubmed --percent 0.0003 --fastmode True --t1 1000 --t2 2000 --k 15`
    - 0.05%: `python train.py --dataset pubmed --percent 0.0005 --fastmode True --t1 800 --t2 1800 --k 13`
    - 0.10%: `python train.py --dataset pubmed --percent 0.001 --fastmode True --t1 800 --t2 1800 --k 11`

# Citation
We appreciate it if you cite the following paper:
```
@Article{LiaoTcy2022,
  author =  {Zhuolin Liao and Xiaolin Zhang and Wei Su and Kun Zhan},
  title =   {View-Consistent Heterogeneous Network on Graphs with Few Labeled Nodes},
  journal = {IEEE Transactions on Cybernetics},
  year =    {2022},
  volume =  {},
  number =  {},
  pages =   {}
 }

```

# Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)
