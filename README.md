# Difference-Masking

![[diffmask_process.png]]

This is code for the paper Difference-Masking (under submission to ACL'23). For any questions about the experiments, please reach out to {awilf,sakter}@cs.cmu.edu.

## Domain-Adaptive Setting (NLP)
We reproduce the experiments from [Don't Stop Pretraining](https://github.com/allenai/dont-stop-pretraining) using the setting from [AANG](https://arxiv.org/pdf/2205.14082.pdf).  Our modifications on the repositories for these papers will be posted soon.

## Task-Adaptive Setting (Multimodal Video)
### Resources
All of these experiments were run using MERLOT-Reserve on Google Cloud TPUs. If you do not have access to Cloud TPU's, this is a [good resource](https://sites.research.google/trc/about/). Here is the [github for MERLOT-Reserve ](https://github.com/rowanz/merlot_reserve)that our code is based on.

### TVQA
To run TVQA in the finetuning setting, please first get the data from the TVQA dataset. You'll need to fill out a form to access the raw data. Then, code will be posted soon.

### Social-IQ
At this time, we are not permitted to share the raw Social-IQ data, though we plan to process and release features that can be used to reproduce this code.
