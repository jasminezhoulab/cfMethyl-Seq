#!/bin/bash
mkdir -p output

export PYTHONPATH=../src/:$PYTHONPATH

echo "======================"
echo "Perform cancer detection using example data"
echo "======================"
python ../src/cancer_detection.py ./example.input_data.config
echo ""

echo "======================"
echo "Perform TOO prediction using example data"
echo "======================"
python ../src/TOO.py ./example.input_data.config
echo ""

