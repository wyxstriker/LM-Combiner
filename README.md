# LM-Combiner
Implementation of COLING 2024 paper ["LM-Combiner: A Contextual Rewriting Model for Chinese Grammatical Error Correction"](https://aclanthology.org/2024.lrec-main.934.pdf).
<div align="center">
    <image src='pic/main.png' width="80%">
</div>
All the code and [model](https://huggingface.co/DecoderImmortal/LM-Combiner) are released. Thank you for your patience!

# Requirements

The part of the model is implemented using the huggingface framework and the required environment is as follows:
- Python
- torch
- transformers
- datasets
- tqdm

For the evaluation, we refer to the relevant environment configurations of [ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT).

# Training Stage
## Preprocessing
### Baseline Model 
- Firstly, we train a baseline model (Chinese-Bart-large) for LM-Combiner on the FCGEC dataset using the Seq2Seq format.
```bash
sh ./script/run_bart_baseline.sh
```
### Candidate Datasets
1. Candidate Sentence Generation
- We use the baseline model to generate candidate sentences for the training and test sets
- On tasks where the model fits better (spelling correction, etc.), we recommend using the K-fold cross-inference from the paper to generate candidate sentences separately.
```bash
python ./src/predict_bl_tsv.py
```
2. Golden Labels Merging
- We use the ChERRANT tool to fully decouple the error correction task and the rewriting task by merging the correct labels.
```bash
python ./scorer_wapper/golden_label_merging.py
```
## LM-combiner (gpt2)
- Subsequently, we train LM-Combiner on the constructed candidate dataset
- In particular, we supplement the gpt2 vocab (mainly **double quotes**) to better fit the FCGEC dataset, see ```./pt_model/gpt2-base/vocab.txt``` for details.
```bash
sh ./script/run_lm_combiner.py
```

# Evaluation
- We use the official ChERRANT script to evaluate the model on the FCGEC-dev.
```shell
sh ./script/compute_score.sh
```
|method|Prec|Rec|F0.5|
|-|-|-|-|
| bart_baseline|28.88|**38.95**|40.46|
|+lm_combiner|**52.15**|37.41|**48.34**|
# Citation

If you find this work is useful for your research, please cite our paper:

```
@inproceedings{wang-etal-2024-lm-combiner,
    title = "{LM}-Combiner: A Contextual Rewriting Model for {C}hinese Grammatical Error Correction",
    author = "Wang, Yixuan  and
      Wang, Baoxin  and
      Liu, Yijun  and
      Wu, Dayong  and
      Che, Wanxiang",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.934",
    pages = "10675--10685",
}
```
