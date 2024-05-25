# EaasPrivacy

This is the code for the paper "Split-and-Denoise: Protect large language model inference with local differential privacy" in ICML 2024.

### Split-and-Denoise (SnD)
Train and test the denoise model by:
- cd path_of_eaas-privacy
- running ./run.sh

Refer to util.parameters for the full list of parameters.

### Baselines
- Token Embedding Privatization: python -m baseline.test_with_noise --base_model bert-base-uncased --task glue_mrpc
- Text2Text privatization: python -m baseline.text2text_privatization --base_model bert-base-uncased --task glue_mrpc
- PART: python -m baseline.part --base_model bert-base-uncased --task glue_mrpc