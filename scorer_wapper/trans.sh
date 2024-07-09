

HYP_PARA_FILE=/share/home/chewanxiang/yxwang/cbart_cot/data/m2/train.txt
HYP_M2_FILE=/share/home/chewanxiang/yxwang/cbart_cot/data/m2/train.json

python scorer_wapper/parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char  # char-level evaluation
