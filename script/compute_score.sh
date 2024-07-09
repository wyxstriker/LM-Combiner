

# HYP_M2_FILE=../temp_predict_bart.m2
HYP_M2_FILE=../temp_predict_gpt.m2

REF_M2_FILE=../data/fcgec/dev.m2.char

cd scorer_wapper

python compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE