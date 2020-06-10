{
'''parameters of training osnetain'''
LR=0.0015
EPOCHS=100

TRAINDATA1=market
TESTDATA=market
STEPS=200
PIDNUM=1503

'''train res50 backbone on msmt, use_rea false, colorjitor True, combineall False'''
python3 main.py \
--mode train --cnnbackbone osnetain --use_rea False --use_colorjitor True --combine_all False \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA \
--steps $STEPS --pid_num $PIDNUM --base_learning_rate $LR --total_train_epochs $EPOCHS \
--output_path ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/

TESTDATA=duke
python3 main.py \
--mode test --test_dataset $TESTDATA \
--cnnbackbone osnetain --resume_test_model ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/model_$EPOCHS.pkl \
--output_path ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/

TESTDATA=msmt
python3 main.py \
--mode test --test_dataset $TESTDATA \
--cnnbackbone osnetain --resume_test_model ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/model_$EPOCHS.pkl \
--output_path ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/
}&

{
'''parameters of training osnetain'''
LR=0.0015
EPOCHS=100

TRAINDATA1=duke
TESTDATA=duke
STEPS=200
PIDNUM=1812

'''train res50 backbone on msmt, use_rea false, colorjitor True, combineall False'''
python3 main.py \
--mode train --cnnbackbone osnetain --use_rea False --use_colorjitor True --combine_all False \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA \
--steps $STEPS --pid_num $PIDNUM --base_learning_rate $LR --total_train_epochs $EPOCHS \
--output_path ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/

TESTDATA=market
python3 main.py \
--mode test --test_dataset $TESTDATA \
--cnnbackbone osnetain --resume_test_model ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/model_$EPOCHS.pkl \
--output_path ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/

TESTDATA=msmt
python3 main.py \
--mode test --test_dataset $TESTDATA \
--cnnbackbone osnetain --resume_test_model ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/model_$EPOCHS.pkl \
--output_path ./results/res50ibna_cbfalse_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/
}&

{
'''parameters of training osnetain'''
LR=0.0015
EPOCHS=50

TRAINDATA1=msmt
TESTDATA=msmt
STEPS=1000
PIDNUM=4101

'''train res50 backbone on msmt, use_rea false, colorjitor True, combineall False'''
python3 main.py \
--mode train --cnnbackbone osnetain --use_rea False --use_colorjitor True --combine_all True \
--train_dataset $TRAINDATA1 --test_dataset $TESTDATA \
--steps $STEPS --pid_num $PIDNUM --base_learning_rate $LR --total_train_epochs $EPOCHS \
--output_path ./results/res50ibna_cbTrue_reafasle_colortrue/$TRAINDATA1/

TESTDATA=duke
python3 main.py \
--mode test --test_dataset $TESTDATA \
--cnnbackbone osnetain --resume_test_model ./results/res50ibna_cbTrue_reafasle_colortrue/$TRAINDATA1/model_$EPOCHS.pkl \
--output_path ./results/res50ibna_cbTrue_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/

TESTDATA=market
python3 main.py \
--mode test --test_dataset $TESTDATA \
--cnnbackbone osnetain --resume_test_model ./results/res50ibna_cbTrue_reafasle_colortrue/$TRAINDATA1/model_$EPOCHS.pkl \
--output_path ./results/res50ibna_cbTrue_reafasle_colortrue/$TRAINDATA1/test-on-$TESTDATA/
}&

wait
exit