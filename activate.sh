#!/bin/bash

IMG=$1
ACCESS=$2
PORT=$3

PRFX_JUPYTER="jupyter_env_"
PRFX_CMD="cmd_env"

if [ "$IMG" = "amaml" ]; then
    echo "Activate the environment for Ajoint MAML..."
elif [ "$IMG" = "dpml" ]; then
    echo "Activate the environment for Differential Private Machine Learning..."
elif [ "$IMG" = "tutorial" ]; then
    echo "Activate the enviroment for CS6350 PyTorch Tutorial..."
elif [ "$IMG" = "event" ]; then
    echo "Activate the enviroment for ODE Tensor Event..."
else
    echo "ERROR: No name Docker image has found !"
    exit 1
fi

echo "  - docker image: $IMG"
echo "  - access mode: $ACCESS"
echo "  - port remap: $PORT"

if [ "$ACCESS" = "jupyter" ]; then
  docker run -it --rm \
          --gpus=all \
          --name="$PRFX_JUPYTER$IMG" \
          -p $PORT:8888 \
          -v ${PWD}:/workspace \
          -w /workspace \
          $IMG \
          jupyter notebook
elif [ "$ACCESS" = "backend" ]; then
  docker exec -it "$PRFX_JUPYTER$IMG" bash
elif [ "$ACCESS" = "cmd" ]; then
  docker run -it --rm \
          --gpus=all \
          --name="$PRFX_CMD$IMG$PORT" \
          -p $PORT:22 \
          -v ${PWD}:/workspace \
          -w /workspace \
          $IMG \
          /bin/bash
else
  echo "Invalid access!"
fi
