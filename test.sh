#Cora_val
name=Cora_val_0.005.log
python train.py --dataset cora --public 0 --percent 0.005 --t1 200 --t2 300 --k 15 | tee ./$name

name=Cora_val_0.01.log
python train.py --dataset cora --public 0 --percent 0.01 --t1 180 --t2 280 --k 13 | tee ./$name

name=Cora_val_0.03.log
python train.py --dataset cora --public 0 --percent 0.03 --t1 140 --t2 240 --k 9 | tee ./$name

#Citeseer_val
name=Citeseer_val_0.005.log
python train.py --dataset citeseer --public 0 --percent 0.005 --t1 200 --t2 300 --k 15 | tee ./$name

name=Citeseer_val_0.01.log
python train.py --dataset citeseer --public 0 --percent 0.01 --t1 180 --t2 280 --k 13 | tee ./$name

#Pubmed_val
name=Pubmed_val_0.0003.log
python train.py --dataset pubmed --public 0 --percent 0.0003 --t1 1000 --t2 2000 --k 15 | tee ./$name

name=Pubmed_val_0.0005.log
python train.py --dataset pubmed --public 0 --percent 0.0005 --t1 1000 --t2 2000 --k 13 | tee ./$name

name=Pubmed_val_0.001.log
python train.py --dataset pubmed --public 0 --percent 0.001 --t1 800 --t2 1800 --k 11 | tee ./$name

#Cora_no_val
name=Cora_no_val_0.005.log
python train.py --dataset cora --fastmode True --public 0 --percent 0.005 --t1 200 --t2 300 --k 15 | tee ./$name

name=Cora_no_val_0.01.log
python train.py --dataset cora --fastmode True --public 0 --percent 0.01 --t1 180 --t2 280 --k 13 | tee ./$name

name=Cora_no_val_0.02.log
python train.py --dataset cora --fastmode True --public 0 --percent 0.02 --t1 160 --t2 260 --k 11 | tee ./$name

name=Cora_no_val_0.03.log
python train.py --dataset cora --fastmode True --public 0 --percent 0.03 --t1 140 --t2 240 --k 9 | tee ./$name

name=Cora_no_val_0.04.log
python train.py --dataset cora --fastmode True --public 0 --percent 0.04 --t1 120 --t2 220 --k 7 | tee ./$name

name=Cora_no_val_0.05.log
python train.py --dataset cora --fastmode True --public 0 --percent 0.05 --t1 100 --t2 200 --k 5 | tee ./$name

#Citeseer_no_val
name=Citeseer_no_val_0.005.log
python train.py --dataset citeseer --fastmode True --public 0 --percent 0.005 --t1 200 --t2 300 --k 15 | tee ./$name

name=Citeseer_no_val_0.01.log
python train.py --dataset citeseer --fastmode True --public 0 --percent 0.01 --t1 180 --t2 280 --k 13 | tee ./$name

#Pubmed_no_val
name=Pubmed_no_val_0.0003.log
python train.py --dataset pubmed --fastmode True --public 0 --percent 0.0003 --t1 1000 --t2 2000 --k 15 | tee ./$name

name=Pubmed_no_val_0.0005.log
python train.py --dataset pubmed --fastmode True --public 0 --percent 0.0005 --t1 800 --t2 1800 --k 13 | tee ./$name

name=Pubmed_no_val_0.001.log
python train.py --dataset pubmed --fastmode True --public 0 --percent 0.001 --t1 800 --t2 1800 --k 11 | tee ./$name

