#!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J efficientTrain
 #BSUB -n 1
 #BSUB -W 10:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 source /zhome/bd/4/181258/02514/venv_1/bin/activate
 echo "Running script..."
 python3 efficientnet_train.py

