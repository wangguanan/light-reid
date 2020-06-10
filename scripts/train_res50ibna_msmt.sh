{
TRAINDATA1=msmt
TESTDATA=msmt
STEPS=1500
PIDNUM=4101

'''train res50 backbone on msmt, use_rea false, colorjitor True, combineall True'''
python3 main.py \
--mode train --cnnbackbone res50ibna --use_rea False --use_colorjitor True --combine_all True \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA --steps $STEPS --pid_num $PIDNUM \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/

TESTDATA=market
python3 main.py \
--mode test --cnnbackbone res50ibna --use_rea False --use_colorjitor True  --combine_all True \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA --pid_num $PIDNUM \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/model_120.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/

TESTDATA=duke
python3 main.py \
--mode test --cnnbackbone res50ibna --use_rea False --use_colorjitor True  --combine_all True \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA --pid_num $PIDNUM \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/model_120.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/

TESTDATA=wildtrack
python3 main.py \
--mode test --test_mode all --cnnbackbone res50ibna --use_rea False --use_colorjitor True  --combine_all True \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA --pid_num $PIDNUM \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/model_120.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA-all/

TESTDATA=wildtrack
python3 main.py \
--mode test --cnnbackbone res50ibna --use_rea False --use_colorjitor True  --combine_all True \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA --pid_num $PIDNUM \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/model_120.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/
}

wait
exit