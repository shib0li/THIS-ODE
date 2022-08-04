#!/bin/bash

# DOMAIN=$1
# METHOD=$2
# MAX_EPOCHS=$3

# TEST_INTERVAL=0
# BATCH_SIZE=100
# DEVICE=$4

# DEPLOY=$5
# FOLD_IDX=$6

DOMAIN=$1
METHOD=$2
MAX_EPOCHS=${3:-500}
RANK=${4:-3}
DEPLOY=${5:-SEQ}
DEVICE=${6:-cpu}
FOLD=${7:-0}

BATCH_SIZE=${8:-100}
TEST_INTERVAL=${9:-0}



echo "DEPLOYMENT: $DEPLOY"

if [[ "$DEPLOY" == "SEQ" ]];
then

#     for Fidx in {0..4..1}
#     do
#        bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE $Fidx $BATCH_SIZE $TEST_INTERVAL
#     done

    for Fidx in {0..2..1}
    do
       bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE $Fidx $BATCH_SIZE $TEST_INTERVAL
    done
    
elif [[ "$DEPLOY" == "PAR" ]];
then

    trap "kill 0" EXIT

    bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE 0 $BATCH_SIZE $TEST_INTERVAL &
    P1=$!
    bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE 1 $BATCH_SIZE $TEST_INTERVAL &
    P2=$!
    bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE 2 $BATCH_SIZE $TEST_INTERVAL &
    P3=$!
#     bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE 3 $BATCH_SIZE $TEST_INTERVAL &
#     P4=$!
#     bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE 4 $BATCH_SIZE $TEST_INTERVAL &
#     P5=$!

#     wait $P1 $P2 $P3 $P4 $P5

    wait $P1 $P2 $P3
    
elif [[ "$DEPLOY" == "FOLD" ]];
then

    bash run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEPLOY $DEVICE $FOLD $BATCH_SIZE $TEST_INTERVAL 
    
else
    echo "Error: DEPLOYMENT.."
    exit 1
fi
