# AdvancingSeq2Seq
Repository for ACL 2020 Submission "Advancing Seq2Seq Models with Joint Paraphrase Learning" 

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

Note: Currently, this codebase only works on machines with cuda gpu.

You need to install requirements for this project before . -->

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
```
python train.py -a 0 -sh 0 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl-1/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 ; python test.py -sh 0 -spl -1 -load_dir copy_real_bi/shuffle0/spl-1/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 2 -bi 1 -m 0 
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
```
python train.py -a 0.01 -sh 0 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl-1/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -s_e_e 1 ; python test.py -sh 0  -spl -1 -load_dir copy_real_bi/shuffle0/spl-1/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 0 -bi 1 -m 0 
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
```
python train.py -a 0.01 -sh 0 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl-1/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 0  -spl -1 -load_dir copy_real_bi/shuffle0/spl-1/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
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
```
python train.py -a 0.0075 -sh 0 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle0/spl-1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 ; python test.py -sh 0  -spl -1 -load_dir copy_real_bi/shuffle0/spl-1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
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
```
python train.py -a 0 -sh 2 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl-1/1e-3_alpha0_lr0.5 -bin 0 -con 1  -c 0 -bi 1 -H decay5e-4.py -s_e_e 1; python test.py -sh 2 -spl -1 -load_dir copy_real_bi/shuffle2/spl-1/1e-3_alpha0_lr0.5 -e 20  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

**4-2: ParaGen**
```
python train.py -a 0.01 -sh 2 -spl 0 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl0/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1; python test.py -sh 2 -spl 0 -load_dir copy_real_bi/shuffle2/spl0/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl1/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1; python test.py -sh 2 -spl 1 -load_dir copy_real_bi/shuffle2/spl1/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl2/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1; python test.py -sh 2 -spl 2 -load_dir copy_real_bi/shuffle2/spl2/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1; python test.py -sh 2 -spl 3 -load_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl-1/ac0_qp1/1e-3_alha0.01_lr_1.5 -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py -s_e_e 1; python test.py -sh 2 -spl -1 -load_dir copy_real_bi/shuffle2/spl4/ac0_qp1/1e-3_alha0.01_lr_1.5 -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

**4-3: ParaDetect**

```
python train.py -a 0.01 -sh 2 -spl 0 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl0/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 2  -spl 0 -load_dir copy_real_bi/shuffle2/spl0/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl1/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 2  -spl 1 -load_dir copy_real_bi/shuffle2/spl1/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 2  -spl 2 -load_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.01 -sh 2 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl-1/qp_cos_rev/cos_only_0.98_1.5_clip6 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6; python test.py -sh 2  -spl -1 -load_dir copy_real_bi/shuffle2/spl-1/qp_cos_rev/cos_only_0.98_1.5_clip6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

**4-4: ParaGen + ParaDetect**

```
python train.py -a 0.0075 -sh 2 -spl 0 -m 0 save_dir copy_real_bi/shuffle2/spl0/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 2  -spl 0 -load_dir copy_real_bi/shuffle2/spl0/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.0075 -sh 2 -spl 1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 2  -spl 1 -load_dir copy_real_bi/shuffle2/spl1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.0075 -sh 2 -spl 2 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 2  -spl 1 -load_dir copy_real_bi/shuffle2/spl1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```
```
python train.py -a 0.0075 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5; python test.py -sh 2  -spl 2 -load_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0  
```
```
python train.py -a 0.0075 -sh 2 -spl -1 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl-1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -ac 0 -qp 1 -bin 0 -con 1  -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 ; python test.py -sh 2  -spl -1 -load_dir copy_real_bi/shuffle2/spl-1/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

<!---### and without  --> 
5. To replicate results for models with word2vec for a single split (splitting scheme II) (Table 2), run the following:

**5-1: Baseline (Simple Seq2seq with Copy)**
```
python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir word2vec/shuffle2/spl3/baseline -bin 0 -con 1 -c 2 -bi 1 -H decay5e-4.py -s_e_e 1 -word_vec 1 -word_vec_medical 1; python test.py -sh 2  -spl 3 -load_dir  word2vec/shuffle2/spl3/baseline  -e 8 -bin 0 -con 1 -c 2 -bi 1 -m 0 -word_vec 1 -word_vec_medical 1
```
**5-2: ParaGen**
```
python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir word2vec/shuffle2/spl3/geb -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py  -s_e_e 1 -word_vec 1 -word_vec_medical 1; python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir word2vec/shuffle2/spl3/geb -ac 0 -qp 1 -bin 0 -con 1  -c 1 -bi 1 -H decay1.5e-3.py  -s_e_e 1 -word_vec 1 -word_vec_medical 1
```
**5-3: ParaDetect**
```
python train.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir word2vec/shuffle2/spl3/cos -ac 0 -qp 1 -bin 0 -con 1 -c 3 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 -word_vec 1 -word_vec_medical 1; python test.py -sh 2  -spl 3 -load_dir word2vec/shuffle2/spl3/cos -e 14 -bin 0 -con 1 -c 3 -bi 1 -m 0 
```
**5-4: ParaGen + ParaDetect**
```
python train.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir word2vec/shuffle2/spl3/both -ac 0 -qp 1 -bin 0 -con 1 -c 2 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 -word_vec 1 -word_vec_medical 1; python test.py -sh 2  -spl 3 -load_dir  word2vec/shuffle2/spl3/both -e 13 -bin 0 -con 1 -c 3 -bi 1 -m 0 
```



6. To replicate results for baseline with BERT sentence embeddings, run:
```
python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir bertsent/shuffle2/spl3/baseline_bertsent -bin 0 -con 1 -c 1 -bi 1 -H decay5e-4.py -s_e_e 1 -bert_sent 1; python test.py -sh 2 -spl 3 -load_dir bertsent/shuffle2/spl3/baseline_bertsent -e 25  -bin 0 -con 1 -c 1 -bi 1 -m 0 
```

7. To replicate results for the shrinking experiment for a single split (Figure 5), run the following: 
**7-1: Baseline (Simple Seq2seq with Copy)**
```
python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.8 -bin 0 -con 1 -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 -down_sample_perc 0.8; python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.7 -bin 0 -con 1 -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 -down_sample_perc 0.7 ; python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.6 -bin 0 -con 1 -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 -down_sample_perc 0.6; python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.5 -bin 0 -con 1 -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 -down_sample_perc 0.5; python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.4 -bin 0 -con 1 -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 -down_sample_perc 0.4; python train.py -a 0 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.3 -bin 0 -con 1 -c 0 -bi 1 -H decay5e-4.py -s_e_e 1 -down_sample_perc 0.3
```
```
python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.8 -e 20 -bin 0 -con 1 -c 0 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.7 -e 20 -bin 0 -con 1 -c 0 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.6 -e 20 -bin 0 -con 1 -c 0 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.5 -e 20 -bin 0 -con 1 -c 2 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.4 -e 20 -bin 0 -con 1 -c 2 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/1e-3_alpha0_s1_lr0.5_downsample0.3 -e 20 -bin 0 -con 1 -c 2 -bi 1 -m 0
```
**7-2: ParaGen**
```
python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.8 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -down_sample_perc 0.8 -s_e_e 1 ;python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.7 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -down_sample_perc 0.7 -s_e_e 1 ; python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.6 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -down_sample_perc 0.6 -s_e_e 1 ; python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.5 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -down_sample_perc 0.5 -s_e_e 1; python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.4 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -down_sample_perc 0.4 -s_e_e 1; python train.py -a 0.01 -sh 2 -spl 3 -m 0  -kl 0 -save_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.3 -ac 0 -qp 1 -bin 0 -con 1  -c 3 -bi 1 -H decay1.5e-3.py -down_sample_perc 0.3 -s_e_e 1
```
```
python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.8  -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.7  -e 18 -bin 0 -con 1 -c 1 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.6  -e 18 -bin 0 -con 1 -c 2 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.5  -e 18 -bin 0 -con 1 -c 2 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.4  -e 18 -bin 0 -con 1 -c 3 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/ac0_qp1/1e-3_alha0.01_s1_lr_1.5_downsampled0.3  -e 18 -bin 0 -con 1 -c 3 -bi 1 -m 0
```
**7-3: ParaDetect**
```
python train.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.8 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 -down_sample_perc 0.8 ; python train.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.7 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 -down_sample_perc 0.7 ; python train.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.6 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 -down_sample_perc 0.6; python train.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.5 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 -down_sample_perc 0.5; python train.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.4 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 -down_sample_perc 0.4; python train.py -a 0.01 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl2/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.3 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.98_1.5e-3.py -cos_obj 1 -cos_alph 1 -cos_only 1 -s_e_e 1 -clip 6 -down_sample_perc 0.3
```
```
python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.8 -e 14 -bin 0 -con 1 -c 3 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.7 -e 14 -bin 0 -con 1 -c 3 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.6 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.5 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0 ; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.4 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_only_0.98_1.5_clip6_downsample0.3 -e 14 -bin 0 -con 1 -c 1 -bi 1 -m 0
```
**7-4: ParaGen + ParaDetect**
```
python train.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.8 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 -down_sample_perc 0.8  ; python train.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.7 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 -down_sample_perc 0.7 ; python train.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.6 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 -down_sample_perc 0.6; python train.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.5 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 -down_sample_perc 0.5; python train.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.4 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 -down_sample_perc 0.4; python train.py -a 0.0075 -sh 2 -spl 3 -m 0 -kl 0 -save_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.3 -ac 0 -qp 1 -bin 0 -con 1 -c 0 -bi 1 -H decay_0.965_1.75e-3.py -s_e_e 1 -cos_obj 1 -cos_alph 0.75 -clip 5 -down_sample_perc 0.3
```
```
python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.8 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.7 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.6 -e 17 -bin 0 -con 1 -c 0 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.5 -e 17 -bin 0 -con 1 -c 1 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.4 -e 17 -bin 0 -con 1 -c 3 -bi 1 -m 0; python test.py -sh 2  -spl 3 -load_dir copy_real_bi/shuffle2/spl3/qp_cos_rev/cos_alph0.75_decay0.965_1.75_alpha0.0075_downsample0.3 -e 17 -bin 0 -con 1 -c 3 -bi 1 -m 0
```

### Results for Semantic Parsing on Overnight

1. Download pre-processed data at https://www.dropbox.com/sh/y9fvys5ogc4yvnt/AAAuZf8A7PDAdvb_9Ia36wMPa?dl=0 and unzip 'data.zip'.

2. Copy the uncompressed folder 'data' into the folder bowser_recipes_para. 

3. To replicate results (Table 2), run the following commands for each model by replacing $domain$ with one of {basketball, recipes, restaurants, publications, socialnetwork, housing, calendar, blocks}:

**3-1: Baseline (Simple Seq2seq with Copy)**

```
python bowser_train_copy.py -a 0 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr 0.005 -d 0 -c 0 -l 0 -k 5 -e 200 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start 14 -half_end 36 -seed 0 -save_dir $domain$/baseline_tes_k3  -domain $domain$
```

**3-2: ParaGen**

```
python bowser_train_copy.py -a 0.01 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr 0.003 -d 0 -c 0 -l 0 -k 7 -e 200 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start 12 -half_end 44 -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$ -qp 1 -multi_para 1
```

**3-3: ParaDetect**

```
python bowser_train_copy.py -a 0.01 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr 0.003 -d 0 -c 3 -l 0 -k 7 -e 200 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start 8 -half_end 40 -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$ -qp 1 -multi_para 1 -cos_only 1 -cos_alph 1 -cos_obj 1
```

**3-4: ParaGen + ParaDetect**

```
python bowser_train_copy.py -a 0.075 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr 0.003 -d 0 -c 3 -l 0 -k 7 -e 200 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start 12 -half_end 44 -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$ -qp 1 -multi_para 1  -cos_alph 0.75 -cos_obj 1 
```

### Results for Czech -> English Machine Translation on CzENG 1.6 + ParaNMT50M   

1. Download pre-processed data at https://www.dropbox.com/sh/mbc9pkgv5trx8xh/AAAHvIDGjkmkf9nJGh4VtdgJa?dl=0

2. Copy the uncompressed folder 'data' into the folder CzEng.

3. To replicate results (Table 2), run the following commands for each model:
**3-1-1: Baseline (Simple Seq2seq without Copy)**
```
python CzEng_train_seq2seq.py -a 0 -save_dir 0.001_no_word2vec -lr 0.001 -k 1000 -c 2 -v 1 -batch_size 64 -hid_size 256  -cross_ent 0 -pad 0 -prob_already 0  -half_start 1000 -half_end 1000 -half_factor 0.8 -num_epoch 100 -train_size 1 -s_e_e 1 -s_from 1
```
```
python test_file.py -c 2 -load_dir  0.001_no_word2vec -epoch 50 -a 0
```

**3-1-2: Baseline (Simple Seq2seq without Copy) with Word2Vec**
```
python CzEng_train_seq2seq.py -a 0 -save_dir 0.001_word2vec -lr 0.001 -k 1000 -c 0 -v 1 -batch_size 64 -hid_size 256  -cross_ent 0  -pad 0 -prob_already 0  -half_start 1000 -half_end 1000 -half_factor 0.8 -num_epoch 100 -train_size 1 -word_vec 1 -s_e_e 1 -s_from 0
```
```
python test_file.py -c 3 -load_dir 0.001_word2vec -epoch 50 -a 0
```

**3-2-1: ParaGen**
```
python CzEng_train_seq2seq.py -a 0 -save_dir test -lr 0.01 -k 5 -c 2 -v 0 -batch_size 64 -hid_size 256  -cross_ent 0 -clip 0.8 -pad 1 -prob_already 0  -half_start 5 -half_end 55 -half_factor 0.8 -num_epoch 100 -small_train 1 -word_vec 1
```
```
python test_file.py -load_dir  gen_0.01_no_word2vec -epoch 60 -c 1 -a 0.01
```

**3-2-2: ParaGen with Word2Vec**
```
python CzEng_train_seq2seq.py -a 0.01 -save_dir gen_0.01_no_word2vec -lr 0.001  1000 -c 3 -v 0 -batch_size 32 -hid_size 256  -cross_ent 0 -pad 0 -prob_already 0  -half_start 1000 -half_end 1000 -half_factor 0.8 -num_epoch 100 -train_size 1 -s_e_e 1 -s_from 1
```
```
python test_file.py -load_dir  gen_0.01_no_word2vec -epoch 60 -c 1 -a 0.01 
```

**3-3-1: ParaDetect**
```
python bowser_train_copy.py -a 0.01 -which_attn_g general -which_attn_c general -bahd_g 1 -bahd_c 1 -lr 0.003 -d 0 -c 0 -l 0 -k 7 -e 200 -bi 1  -v 0  -pad 0 -jia 1  -train_f $domain$/train_v1.json  -test_f $domain$/test_v1.json -half_start 12 -half_end 44 -seed 100 -save_dir $domain$/baseline_tes_k3  -domain $domain$ -qp 1 -multi_para 1
```
```
python test_file.py -c 2 -load_dir 0.001_cos_0.01_nopad_noword2vec  -epoch 100 -a 0.01  -cos_alph 1 -cos_only 1
```

**3-3-2: ParaDetect with Word2Vec**
```
python CzEng_train_seq2seq.py -a 0.01 -save_dir 0.001_cos_0.01 -lr 0.001 -k 1000 -c 0 -v 1 -batch_size 64 -hid_size 256  -cross_ent 0 -pad 0 -prob_already 0  -half_start 1000 -half_end 1000 -half_factor 0.8 -num_epoch 100 -train_size 1 -word_vec 1 -cos_alph 1 -cos_only 1 
```
```
python test_file.py -c 2 -load_dir 0.001_cos_0.01 -epoch 100 -a 0.01 -cos_alph 1 -cos_only 1
```


**3-4-1: ParaGen + ParaDetect**
```
python CzEng_train_seq2seq.py -a 0.0075 -save_dir both_noword2vec_alph_0.0075 -lr 0.001 -k 1000 -c 0 -v 0 -batch_size 32 -hid_size 256  -cross_ent 0 -pad 0 -prob_already 0  -half_start 1000 -half_end 1000 -half_factor 0.8 -num_epoch 100 -train_size 1  -s_e_e 1 -s_from 40 -cos_alph 0.75 -load_dir both_noword2vec_alph_0.0075 
```
```
python  test_file.py -load_dir both_noword2vec_alph_0.0075 -a 0.01   -epoch 65 -c 3
```
**3-4-2: ParaGen + ParaDetect with Word2Vec**
```
python CzEng_train_seq2seq.py -a 0.0075 -save_dir both_withword2vec_alph_0.0075 -lr 0.001 -k 1000 -c 3 -v 0 -batch_size 32 -hid_size 256  -cross_ent 0 -pad 0 -prob_already 0  -half_start 1000 -half_end 1000 -half_factor 0.8 -num_epoch 100 -train_size 1 -word_vec 1 -s_e_e 1 -s_from 40 -cos_alph 0.75
```
```
python test_file.py -load_dir  both_withword2vec_alph_0.0075 -a 0.01   -epoch 65 -c 0
```



