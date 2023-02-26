#!/bin/sh
python3 make_dataset.py
python train_knn_model.py
python example_selection.py
