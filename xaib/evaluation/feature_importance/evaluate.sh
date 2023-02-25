#!/bin/sh
python make_dataset.py
python train_model.py
python feature_importance.py
