# Identifying Compromised Accounts on Social Media Using Statistical Text Analysis

Paper: (https://arxiv.org/abs/1804.07247)

Python: 2.7

### Run Code

To run a short demo of the source code please execute the following command in your unix shell:

`bash demo.sh`

Since we are not allowed to publish the entire dataset with raw tweets, the results will not be 
the same as in our paper. If you would like to reproduce the results in our paper run:

`bash experiment_paper.sh`

### Using Code for Your Reserach Project

You can use this code out-of-the-box by providing a custom dataset. For an example, 
please take a look at `data/tweets.tsv.gz`. Each line has the following format:

`<user_id>\t<timestamp>\t<tweet_content>\n`

You can modify hyperparameters of our model in the config file `config.yaml`.

### Citation
Please cite the following paper if you make use of the source code:
>@article{seyler2018identifying,
  title={Identifying Compromised Accounts on Social Media Using Statistical Text Analysis},
  author={Seyler, Dominic and Li, Lunan and Zhai, ChengXiang},
  journal={arXiv preprint arXiv:1804.07247},
  year={2018}
}

*(c) Dominic Seyler*
