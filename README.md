# View-Consistent Heterogeneous Networks
Performing transductive learning on graphs with very few labeled data, i.e., two or three samples for each category, is challenging due to lack of supervision. In existing work, self-supervised learning via a single view model is widely adopted to address the problem. However, recent observation shows multi-view representations of an object share the same semantic information in high-level feature space. For each sample, we generate heterogeneous representations and use view-consistency loss to make their representations consistent with each other. Multiview representation also inspires to supervise the pseudo labels generation by the aid of mutual supervision between views. In this paper, we thus propose a View-Consistent Heterogeneous Network (VCHN) to learn better representations by aligning view-agnostic semantics. Specifically, VCHN is constructed by constraining the predictions between two views so that the view pairs could supervise each other. To the best use of cross-view information, we further propose a novel training strategy to generate more reliable pseudo labels, which thus enhances predictions of the VCHN. Extensive experimental results on three benchmark datasets demonstrate that our method achieves superior performance over state-of-the-art methods under very low label rates.

# Experiments
- For different data sets and different label rates, we set different numbers of pseudo-labels and filtering strengths.
- Results in `./Result` are not the final since some parameters are changed after this version, e.g., No. head.
- The parameter setting and the command to run the code are as follows:

## Three heads
- For Cora with Validation:
  - 0.5%: `python train.py --dataset cora --percent 0.005 --t1 200 --t2 300 --k 15`
  - 1.0%: `python train.py --dataset cora --percent 0.01 --t1 180 --t2 280 --k 13`
  - 3.0%: `python train.py --dataset cora --percent 0.03 --t1 140 --t2 240 --k 9`

- For Cora without Validation:
    - 0.5%: `python train.py --dataset cora --percent 0.005 --fastmode True --t1 200 --t2 300 --k 15`
    - 1.0%: `python train.py --dataset cora --percent 0.01 --fastmode True --t1 180 --t2 280 --k 13`
    - 2.0%: `python train.py --dataset cora --percent 0.02 --fastmode True --t1 160 --t2 260 --k 11`
    - 3.0%: `python train.py --dataset cora --percent 0.03 --fastmode True --t1 140 --t2 240 --k 9`
    - 4.0%: `python train.py --dataset cora --percent 0.04 --fastmode True --t1 120 --t2 220 --k 7`
    - 5.0%: `python train.py --dataset cora --percent 0.05 --fastmode True --t1 100 --t2 200 --k 5`

- For Citeseer with Validation:
    - 0.5%: `python train.py --dataset citeseer --percent 0.005 --t1 200 --t2 300 --k 15`
    - 1.0%: `python train.py --dataset citeseer --percent 0.01 --t1 180 --t2 280 --k 13`

- For Citeseer without Validation:
    - 0.5%: `python train.py --dataset citeseer --percent 0.005 --fastmode True --t1 200 --t2 300 --k 15`
    - 1.0%: `python train.py --dataset citeseer --percent 0.01 --fastmode True --t1 180 --t2 280 --k 13`
## Parts of PubMed Experiments using Eight heads 
- You need to modify the head in `model.py` for PubMed.
- For PubMed with Validation:
    - 0.03%: `python train.py --dataset pubmed --percent 0.0003 --t1 1000 --t2 2000 --k 15`
    - 0.05%: `python train.py --dataset pubmed --percent 0.0005 --t1 1000 --t2 2000 --k 13`
    - 0.10%: `python train.py --dataset pubmed --percent 0.001 --t1 800 --t2 1800 --k 11`

- For PubMed without Validation:
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
