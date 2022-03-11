#  -*- coding: utf-8 -*-
#  File: __init__.py
from .lenet import lenet_fp, lenet_q, lenet_fp_b, lenet_fp_c, lenet_q_b, lenet_q_c, lenet_tf #, model_abc, model_abc_q
from .mnist import model_a, model_a_q, model_b, model_b_q, model_c, model_c_q
from .cifar import cifar_convnet, cifar_convnet_q, cifar_convnet_b, cifar_convnet_qb, cifar_convnet_c, cifar_convnet_qc, cifar10_resnet, cifar10_resnet_q
from .supported_models import SupportedModels
from .get_model import get_model, get_mod_typ, get_model_cifar, get_mod_typ_cifar