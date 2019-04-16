#!/usr/bin/env bash
python generate_comp_datasets.py
python kl_samples.py
python classification.py