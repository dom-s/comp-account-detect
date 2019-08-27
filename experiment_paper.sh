#!/usr/bin/env bash
gunzip -k -f data/kl-samples_0.05.json.gz
gunzip -k -f data/kl-samples_0.1.json.gz
gunzip -k -f data/kl-samples_0.25.json.gz
gunzip -k -f data/kl-samples_0.5.json.gz
gunzip -k -f data/kl-samples_RND.json.gz
python classification.py