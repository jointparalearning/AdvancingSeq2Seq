# AdvancingSeq2Seq
Repository for EMNLP 2020 Submission "Advancing Seq2Seq Semantic Parsing with Joint Paraphrase Learning" 

## Model Performances for the Overnight Dataset

Performances of our implementation of the baseline Seq2Seq and our proposed models (ParaGen, ParaDetect, ParaGen+Detect) on all domains of the Overnight Dataset. 
Also, we compare our results to the baseline/ state-of-the-art results made by <a href="https://www.aclweb.org/anthology/N19-2003">Damonte et al.</a>
All performances are measured by exact match accuracy, not denotation accuracy.

| Model                                | Basketball | Blocks | Calendar | Publications | Recipes | Restaurants | Housing | SocialNetwork |
| ------------------------------------ |:----------:|:------:|:--------:|:------------:|:-------:|:-----------:|:-------:|--------------:|
| Baseline Seq2Seq (Our Implementation)| 82.8 %     | 39.3 % | **59.5 %**   | 60.2 %       | 75.0 %  | 53.3 %      | 47.1 %  | 67.6 %        | 
| ParaGen                              | 82.09 %    | 40.9 % | 54.8 %   | 59.6 %       | **75.5 %**  | **53.9 %**      | **49.2 %**  | **68.3 %**    | 
| ParaDetect                           | **83.8 %** | **42.4 %** | 54.2 %   | 60.9 %       | 74.5 %  | 51.5 %      | 44.4 %  | **68.3 %**    | 
| ParaGen + Detect                     | 82.6 %     | 38.6 % | 56.5 %   | **63.4 %**       | 70.4 %  | 52.4 %      | 45.5 %  | 67.1 %    | 
| Baseline Seq2Seq (Damonte et al.)    | 69.6 %     | 25.1 % | 43.5 %   | 32.9 %       | 58.3 %  | 37.3 %      | 29.6 %  | 51.2 %        | 
| Transfer Learning (Damonte et al.)   | 71.1%      | 25.1 % | 48.8 %   | 40.4 %       | 63.4 %  | 39.2 %      | 38.1 %  | 54.5 %        | 

## How to Replicate Results

<!---### Prerequisites

Note: Currently, this codebase only works on machines with cuda gpu ver 10. 

You need to install latest version of PyTorch, numpy, NLTK. -->

### Results for Semantic Parsing on emrQA 

1. Download pre-processed data at https://www.dropbox.com/sh/5o5ss42iwe2fpxb/AADT3DZYIvlLOBOjXy62HN0ja?dl=0

2. Copy the downloaded folder 'data' into SP_emrQA. 

3. To replicate results for splittng scheme I (Table 1), run the 5 commands and average the numbers obtained from test.py over 5 splits for each model:

**3-1: Baseline**

```
python train.py -a 0 -sh 0 -spl 0 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl0/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1; python test.py -sh 0 -spl 0 -load_dir copy_real_bi/shuffle0/spl0/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0 -sh 0 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl1/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 ;python test.py -sh 0 -spl 1 -load_dir copy_real_bi/shuffle0/spl1/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0 -sh 0 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl2/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 ; python test.py -sh 0 -spl 2 -load_dir copy_real_bi/shuffle0/spl2/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 2 -bi 1 -m 0 
```
```
python train.py -a 0 -sh 0 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl3/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 ; python test.py -sh 0 -spl 3 -load_dir copy_real_bi/shuffle0/spl3/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 2 -bi 1 -m 0 
```

**3-2: ParaGen**
```
python train.py -a 0.01 -sh 0 -spl 0 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl0/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1; python test.py -sh 0  -spl 0 -load_dir copy_real_bi/shuffle0/spl0/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0
```
```
python train.py -a 0.01 -sh 0 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl1/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -s_e_e 1 ; python test.py -sh 0  -spl 1 -load_dir copy_real_bi/shuffle0/spl1/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 0 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 0 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl2/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -s_e_e 1 ; python test.py -sh 0  -spl 2 -load_dir copy_real_bi/shuffle0/spl2/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 0 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 0 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl3/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -s_e_e 1 ; python test.py -sh 0  -spl 3 -load_dir copy_real_bi/shuffle0/spl3/ac0_qp1/1e-3_alha0.01_lr_1.5  -e 18 -bin 0 -con 1 -c 0 -bi 1 -m 0 
```

**3-3: ParaDetect**
```
python train.py -a 0.01 -sh 0 -spl 0 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl0/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 0  -spl 0 -load_dir copy_real_bi/shuffle0/spl0/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 0 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl1/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 0  -spl 1 -load_dir copy_real_bi/shuffle0/spl1/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0
```
```
python train.py -a 0.01 -sh 0 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 0  -spl 2 -load_dir copy_real_bi/shuffle0/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0
```
```
python train.py -a 0.01 -sh 0 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 0  -spl 3 -load_dir copy_real_bi/shuffle0/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

**3-4: ParaGen + ParaDetect**

```
python train.py -a 0.0075 -sh 0 -spl 0 -m 0 save_dir copy_real_bi/shuffle0/spl0/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 0  -spl 0 -load_dir copy_real_bi/shuffle0/spl0/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0
```
```
python train.py -a 0.0075 -sh 0 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 0  -spl 1 -load_dir copy_real_bi/shuffle0/spl1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.0075 -sh 0 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl2/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 0  -spl 2 -load_dir copy_real_bi/shuffle0/spl2/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.0075 -sh 0 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 0  -spl 3 -load_dir copy_real_bi/shuffle0/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

4. To replicate results for splitting scheme II (Table 1), run and average the numbers obtained from test.py over 5 splits for each model:

**4-1: Baseline**
```
python train.py -a 0 -sh 2 -spl 0 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl0/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1; python test.py -sh 2 -spl 0 -load_dir copy_real_bi/shuffle2/spl0/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0 -sh 2 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl1/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1; python test.py -sh 2 -spl 1 -load_dir copy_real_bi/shuffle2/spl1/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0 -sh 2 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl2/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1; python test.py -sh 2 -spl 2 -load_dir copy_real_bi/shuffle2/spl2/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1; python test.py -sh 2 -spl 3 -load_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

**4-2: ParaGen**
```
python train_trainingsample.py -a 0.01 -sh 2 -spl 0 -m 0  -kl 0 -save_dir re-sampled/shuffle2/spl0/paragen_lr1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1
```
```
python train_trainingsample.py -a 0.01 -sh 2 -spl 1 -m 0  -kl 0 -save_dir re-sampled/shuffle2/spl1/paragen_lr1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1
```
```
python train_trainingsample.py -a 0.01 -sh 2 -spl 2 -m 0  -kl 0 -save_dir re-sampled/shuffle2/spl2/paragen_lr1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1
```
```
python train_trainingsample.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir re-sampled/shuffle2/spl3/paragen_lr1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1
```

**4-3: ParaDetect**

```
python train_trainingsample.py -a 0.01 -sh 2 -spl 0 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl0/paradetect -ac 0 -qp 1 -bin 0 -con 1 -c 2 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 
```
```
python train_trainingsample.py -a 0.01 -sh 2 -spl 1 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl1/paradetect -ac 0 -qp 1 -bin 0 -con 1 -c 2 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 
```
```
python train_trainingsample.py -a 0.01 -sh 2 -spl 2 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl2/paradetect -ac 0 -qp 1 -bin 0 -con 1 -c 2 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 
```
```
python train_trainingsample.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl3/paradetect -ac 0 -qp 1 -bin 0 -con 1 -c 2 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 
```

**4-4: ParaGen + ParaDetect**

```
python train_trainingsample.py -a 0.0075 -sh 2 -spl 0 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl0/para_gen_and_detect  -ac 0 -qp 1 -bin 0 -con 1 -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 
```
```
python train_trainingsample.py -a 0.0075 -sh 2 -spl 1 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl1/para_gen_and_detect  -ac 0 -qp 1 -bin 0 -con 1 -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 
```
```
python train_trainingsample.py -a 0.0075 -sh 2 -spl 2 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl2/para_gen_and_detect  -ac 0 -qp 1 -bin 0 -con 1 -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 ```
```
```
python train_trainingsample.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir re-sampled/shuffle2/spl3/para_gen_and_detect  -ac 0 -qp 1 -bin 0 -con 1 -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 
```

### Results for Semantic Parsing on Overnight

1. Download pre-processed data at https://www.dropbox.com/sh/y9fvys5ogc4yvnt/AAAuZf8A7PDAdvb_9Ia36wMPa?dl=0 and unzip 'data.zip'.

2. Copy the uncompressed folder 'data' into the folder bowser_recipes_para. 

3. To replicate results (Table 2), run the following commands for each model by replacing 

    3.1 $domain$ with one of {basketball, recipes, restaurants, publications, socialnetwork, housing, calendar, blocks}
    
    3.2 $k$ with one of {5,7}
    
    3.3 $(half, start)$ with one of {(8,30), (10,32), (12,34), (14,36)}
    
    3.4 $lr$ from one of {0.001, 0.003, 0.005}


**3-1: Baseline (Simple Seq2seq with Copy)**

```
python bowser_train_copy.py -a 0 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr $lr$ -d 0 -c 0 -l 0 -k $k$ -e 50 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start $start$ -half_end $end$ -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$
```

**3-2: ParaGen**

```
python bowser_train_copy.py -a 0.01 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr $lr$  -d 0 -c 0 -l 0 -k $k$ -e 50 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start $start$ -half_end $end$ -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$ -qp 1 -multi_para 1
```

**3-3: ParaDetect**

```
python bowser_train_copy.py -a 0.01 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr $lr$  -d 0 -c 0 -l 0 -k $k$ -e 50 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start $start$ -half_end $end$ -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$ -qp 1 -multi_para 1 -cos_only 1 -cos_alph 1 -cos_obj 1
```

**3-4: ParaGen + ParaDetect**

```
python bowser_train_copy.py -a 0.075 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr $lr$  -d 0 -c 1 -l 0 -k $k$ -e 50 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start $start$ -half_end $end$ -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$ -qp 1 -multi_para 1  -cos_alph 0.75 -cos_obj 1 
```



