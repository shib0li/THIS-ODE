#!/bin/bash


DOMAIN=$1
METHOD=$2
MAX_EPOCHS=$3
RANK=$4
DEVICE=$5
FOLD=$6

BATCH_SIZE=$7
TEST_INTERVAL=$8

echo "======== DEPLOY SUMMARY ========"
echo "domain:         $DOMAIN"
echo "method:         $METHOD"
echo "max epochs:     $MAX_EPOCHS"
echo "rank:           $RANK"
echo "device:         $DEVICE"
echo "fold:           $FOLD"
echo "batch size:     $BATCH_SIZE"
echo "test interval:  $TEST_INTERVAL"
echo "--------------------------------"


if [[ "$METHOD" == "CPTF_linear" ]];
then

python run_CPTF.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='linear'

elif [[ "$METHOD" == "CPTF_rnn" ]];
then

python run_CPTF.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='rnn'
       
elif [[ "$METHOD" == "CPTF_time" ]];
then

python run_CPTF.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='time'

elif [[ "$METHOD" == "GPTF_linear" ]];
then

python run_GPTF.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='linear'

elif [[ "$METHOD" == "GPTF_rnn" ]];
then

python run_GPTF.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='rnn'

elif [[ "$METHOD" == "GPTF_time" ]];
then

python run_GPTF.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='time'
       
elif [[ "$METHOD" == "Tucker" ]];
then

python run_Tucker.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK 

elif [[ "$METHOD" == "Neural_linear" ]];
then

python run_Neural.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='linear'
       
elif [[ "$METHOD" == "Neural_rnn" ]];
then

python run_Neural.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='rnn'
       
elif [[ "$METHOD" == "Neural_time" ]];
then

python run_Neural.py \
       -domain=$DOMAIN \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK \
       -trans='time'

elif [[ "$METHOD" == "NODE" ]];
then

python run_NODE.py \
       -domain=$DOMAIN \
       -est='pt' \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK

elif [[ "$METHOD" == "NODE_noise" ]];
then

python run_NODE.py \
       -domain=$DOMAIN \
       -est='noise' \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK
       
elif [[ "$METHOD" == "NODE_auto" ]];
then

python run_NODE.py \
       -domain=$DOMAIN \
       -est='auto' \
       -max_epochs=$MAX_EPOCHS \
       -test_interval=$TEST_INTERVAL \
       -batch_size=$BATCH_SIZE \
       -device=$DEVICE \
       -fold=$FOLD \
       -verbose=False \
       -R=$RANK
     
else
    echo "Error: no such method found.."
    exit 1
fi
