# TransferabilityAnalysis

The source code in the repository is a Python API that is able to fulfil a complete workflow of:
1. Traning full-precision and quantized neural networks.
2. Creating adversarial examples on these networks.
3. Transferaing adversarial examples from the network where samples are created (source) to another network (target) network.

The API is fairly simple to use. 
A use case is already implemented with a YAML parser. To run this please follow the following steps

## For training:




# References
Quantization is based on DoReFa-Net method as proposed in the paper: https://arxiv.org/abs/1606.06160  
Cited as:  
Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., & Zou, Y. (2018). DoReFa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv:1606.06160 [cs].

#### For quantization, the library available from the authors is used. This is available at:
https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net

The networks used are from the examples provided on the Tensorpack repository:   
Tensorpack cited as:  
Wu, Y. et al. (2016). Tensorpack. https://github.com/tensorpack.

#### Tensorpack models are available at:  
https://github.com/tensorpack/tensorpack/tree/master/examples

#### Adversarial Examples are created using Adversarial Robustness Toolbox (ART) v. 1.5.1. Official paper: https://arxiv.org/abs/1807.01069  
Cited as:  
Nicolae, M.-I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M., Zantedeschi, V., Baracaldo, N., Chen, B., Ludwig, H., Molloy, I. M., & Edwards, B. (2019). Adversarial robustness toolbox v1.0.0. arXiv:1807.01069

ART is one of the popular APIs for adversaral examples generation and supports a large number of attacks. It is open-source with large number of very well explained examples. Please check their repository at:
https://github.com/Trusted-AI/adversarial-robustness-toolbox
