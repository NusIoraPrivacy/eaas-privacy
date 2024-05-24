# EaasPrivacy

Run the following main scripts sequentially:
- train_denoise_model.py: generate training data, and use them to train a denoise model
- run_denoise.py: test the performance of denoise model

Below are the supporting scriptsL
- layers.py & model.py: architecture of denoise and classification model
- denoise.py: functions to generate training data for denoise model, and to train/test the denoise model
- bert_cls.py: test the accuracy of classification model without noise
- test_with_noise.py: test the accuracy of classification model with noise (no denoise)
- parameters.py: define the parameters
- utils.py: other supporting functions

Train and test the model by running:
./run.sh
