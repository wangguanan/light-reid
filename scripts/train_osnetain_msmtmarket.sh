{
BACKBONE=osnetain
LR=0.0015
EPOCHS=50

TRAINDATA1=msmt
TRAINDATA2=market
TRAINDATA3=none
TRAINDATA4=none
TESTDATA=msmt
STEPS=1500
PIDNUM=5604

'''train res50 backbone on msmt, use_rea false, colorjitor True, combineall True'''
python3 main.py \
--mode train --cnnbackbone $BACKBONE --use_rea False --use_colorjitor True --combine_all True \
--base_learning_rate $LR --total_train_epochs $EPOCHS \
--train_dataset $TRAINDATA1 $TRAINDATA2  --test_dataset $TESTDATA --steps $STEPS --pid_num $PIDNUM \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4/

TESTDATA=market
python3 main.py \
--mode test --cnnbackbone $BACKBONE --test_dataset $TESTDATA \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//model_$EPOCHS.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//test-on-$TESTDATA/

TESTDATA=duke
python3 main.py \
--mode test --cnnbackbone $BACKBONE --test_dataset $TESTDATA \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//model_$EPOCHS.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//test-on-$TESTDATA/

TESTDATA=wildtrack
python3 main.py \
--mode test --cnnbackbone $BACKBONE --test_dataset $TESTDATA \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//model_$EPOCHS.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//test-on-$TESTDATA/

TESTDATA=wildtrack
python3 main.py \
--mode test test_mode all --cnnbackbone $BACKBONE --test_dataset $TESTDATA \
--resume_test_model ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//model_$EPOCHS.pkl \
--output_path ./results/res50ibna_combineall_reafasle_colortrue/$TRAINDATA1+$TRAINDATA2+$TRAINDATA3+$TRAINDATA4//test-on-$TESTDATA-all/
}

wait
exit