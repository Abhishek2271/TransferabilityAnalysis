# TransferabilityAnalysis

The source code in the repository is a Python API that is able to fulfil a complete workflow of:
1. Traning full-precision and quantized neural networks.
2. Creating adversarial examples on these networks.
3. Transferaing adversarial examples from the network where samples are created (source) to another network (target) network.

The API is based on Tensropack [(Y. Wu et al., 2016)](https://github.com/tensorpack) for training. Tensorpack is a part of TensorFlow 1.13 [(Abadi et al., 2016)](https://www.tensorflow.org) API.  
For quantization DoReFa-Net method [(Zhou et al., 2018)](https://arxiv.org/abs/1606.06160) is used
For adversarial attack generation ART [(Nicolae et al., 2019)](https://arxiv.org/abs/1807.01069 ) is used. 

The API is fairly simple to use. 
A use case is already implemented with a YAML parser. To run this please follow the following steps

## For training:
In the config.yml file:
1. Update the "task" --> "type" to "training".  
Then in the "training-options" section:  
2. Specify the dataset that the model is to be trained on in the "dataset" field
3. Specify the model to be trained in the "model" field
4. Specify the quantization bitwidth in the "precision" field.
5. To initate the training run "run_exp.py" file.

## For creating adversarial examples:
In the config.yml file:
1. Update the "task" --> "type" to "create"  
Then in the "attack-options" section:  
2. Set the "attack-mode" to "create".  
In the "create-attack" section:  
3. Load a saved model on which the attack is to be created in the "load-model" field. This can be a Tensorflow checkpoint.
4. Specify the base-model, that is the type of the model to load in the "base-model" field.
5. Specify the attack algorithm to use in the "algorithm" field.
6. Specify the dataset to created the adversarial examples in the "dataset" field. Data points will be sampled from this dataset.
7. Specify the quantization bitwidth of the loaded model in the "precision" field.
8. To initate the attack creation run "run_exp.py" file.

## For adversarial transfer:
In the config.yml file:
1. Update the "task" --> "type" to "transfer"  
Then in the "attack-options" section:  
2. Set the "attack-mode" to "transfer".  
In the "transfer-attack" section:  
Specify source properties in the "Source" section:  
3. Add the location of the .npz file containing adversarial examples in the "source-images" field.
4. Specify the dataset used to create the adversarial examples in the "dataset" field.
Specify target properties in the "Target" section:  
7. Load a saved model on which the attack is to be transferred (target model) in the "target-model" field. This can be a Tensorflow checkpoint.
8. Specify the base-model, that is the type of the target mode "base-model" field.
9. Specify the quantization bitwidth of the loaded model in the "precision" field.
10. To initate the attack creation run "run_exp.py" file.


# References
## Quantization is based on DoReFa-Net method as proposed in the paper: https://arxiv.org/abs/1606.06160  
Cited as:  
Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., & Zou, Y. (2018). DoReFa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv:1606.06160 [cs].

#### For quantization, the library available from the authors is used. This is available at:
https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net

The networks used are from the examples provided on the Tensorpack repository:   
Tensorpack cited as:  
Wu, Y. et al. (2016). Tensorpack. https://github.com/tensorpack.

#### Tensorpack models are available at:  
https://github.com/tensorpack/tensorpack/tree/master/examples

## Adversarial Examples are created using Adversarial Robustness Toolbox (ART) v. 1.5.1. Official paper: https://arxiv.org/abs/1807.01069  
Cited as:  
Nicolae, M.-I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M., Zantedeschi, V., Baracaldo, N., Chen, B., Ludwig, H., Molloy, I. M., & Edwards, B. (2019). Adversarial robustness toolbox v1.0.0. arXiv:1807.01069

ART is one of the popular APIs for adversaral examples generation and supports a large number of attacks. It is open-source with large number of very well explained examples. Please check their repository at:
https://github.com/Trusted-AI/adversarial-robustness-toolbox
