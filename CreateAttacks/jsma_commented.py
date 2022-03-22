# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Jacobian-based Saliency Map attack `SaliencyMapMethod`. This is a white-box attack.
| Paper link: https://arxiv.org/abs/1511.07528

There is no change in code here. Just some comments that could be helpful later if I forget why some steps were done!


THIS CLASS IS A COPY OF ATTACK IMPLEMENTATION IN ART:
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/990236dfca56618f25ae1aa7e14fae2eef0892ac/art/attacks/evasion/saliency_map.py

THIS IS ONLY HERE SO THAT IT IS COMMENTED WHEREEVER APPROPRIATE AND IS NOT USED IN THE PROJECT IN ANY WAY. IT IS HERE JUST FOR REFERENCE. 

CREDITS TO ART DEVELOPERS:
WHEN ART IS INSTALLED THIS FILE IS INSTALLED IN THE FOLLOWING LOCATION.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import check_and_transform_label_format, compute_success

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class SaliencyMapMethod(EvasionAttack):
    """
    Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).
    | Paper link: https://arxiv.org/abs/1511.07528
    """

    attack_params = EvasionAttack.attack_params + ["theta", "gamma", "batch_size", "verbose"]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        theta: float = 0.1,
        gamma: float = 1.0,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Create a SaliencyMapMethod instance.
        :param classifier: A trained classifier.
        :param theta: Amount of Perturbation introduced to each modified feature per step (can be positive or negative).
        :param gamma: Maximum fraction of features being perturbed (between 0 and 1).
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.theta = theta
        self.gamma = gamma
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        # Initialize variables
        # Here dims is used to compute total number of features in an image.
        dims = list(x.shape[1:])        # for an image batch of (1000,32,32,3) this will give (32,32,3) 
        self._nb_features = np.product(dims) #for (32,32,3) nb_features has 32x32x3 = 3072 features.
        #nb_features is used later to compute the total amount of features used when crafting adv samples.
        #and thus it is clear that the code considers multiple channels as individual features
        x_adv = np.reshape(x.astype(ART_NUMPY_DTYPE), (-1, self._nb_features))
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # Determine target classes for attack
        if y is None:
            # Randomly choose target from the incorrect classes for each sample
            from art.utils import random_targets

            targets = np.argmax(random_targets(preds, self.estimator.nb_classes), axis=1)
        else:
            if self.estimator.nb_classes == 2 and y.shape[1] == 1:
                raise ValueError(
                    "This attack has not yet been tested for binary classification with a single output classifier."
                )

            targets = np.argmax(y, axis=1)

        # Compute perturbation with implicit batching
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="JSMA", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            # Initialize the search space; optimize to remove features that can't be changed
            if self.estimator.clip_values is not None:
                search_space = np.zeros(batch.shape)
                clip_min, clip_max = self.estimator.clip_values
                if self.theta > 0:
				#for all images within the batch that are less than max pixel values, include them in the search space
                    search_space[batch < clip_max] = 1
                else:
                    search_space[batch > clip_min] = 1
            else:
                search_space = np.ones(batch.shape)

            # Get current predictions
            current_pred = preds[batch_index_1:batch_index_2]
            target = targets[batch_index_1:batch_index_2]
            active_indices = np.where(current_pred != target)[0]
            all_feat = np.zeros_like(batch)
            
            # Active indices contain:
            #   1. Images that yet did not cause misclasfication to target class
            #   2. Images that have not been modified till the clipping values (0 and 1 mostly)
            # Also to note here is that active indices represent indies of a batch where each index represents an # image and NOT features within an image
            while active_indices.size != 0:
                # Compute saliency map
                feat_ind = self._saliency_map(
                    np.reshape(batch, [batch.shape[0]] + dims)[active_indices],
                    target[active_indices],
                    search_space[active_indices],
                )

                # Update used features
                all_feat[active_indices, feat_ind[:, 0]] = 1
                all_feat[active_indices, feat_ind[:, 1]] = 1

                # Apply attack with clipping
                if self.estimator.clip_values is not None:
                    # Prepare update depending of theta
                    if self.theta > 0:
                        clip_func, clip_value = np.minimum, clip_max
                    else:
                        clip_func, clip_value = np.maximum, clip_min

                    # Update adversarial examples
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] = clip_func(
                        clip_value,
                        tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] + self.theta,
                    )
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] = clip_func(
                        clip_value,
                        tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] + self.theta,
                    )
                    batch[active_indices] = tmp_batch

                    # Remove indices from search space if max/min values were reached
                    search_space[batch == clip_value] = 0

                # Apply attack without clipping
                else:
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] += self.theta
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] += self.theta
                    batch[active_indices] = tmp_batch

                # Recompute model prediction
                current_pred = np.argmax(
                    self.estimator.predict(np.reshape(batch, [batch.shape[0]] + dims)),
                    axis=1,
                )

                # Update active_indices
                active_indices = np.where(
                    (current_pred != target)
                    * (np.sum(all_feat, axis=1) / self._nb_features <= self.gamma)
                    * (np.sum(search_space, axis=1) > 0)
                )[0]

            x_adv[batch_index_1:batch_index_2] = batch

        x_adv = np.reshape(x_adv, x.shape)

        logger.info(
            "Success rate of JSMA attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, batch_size=self.batch_size),
        )

        return x_adv

    def _saliency_map(self, x: np.ndarray, target: Union[np.ndarray, int], search_space: np.ndarray) -> np.ndarray:
        """
        Compute the saliency map of `x`. Return the top 2 coefficients in `search_space` that maximize / minimize
        the saliency map.
        :param x: A batch of input samples.
        :param target: Target class for `x`.
        :param search_space: The set of valid pairs of feature indices to search.
        :return: The top 2 coefficients in `search_space` that maximize / minimize the saliency map.
        """
        grads = self.estimator.class_gradient(x, label=target)
        grads = np.reshape(grads, (-1, self._nb_features))

        # Remove gradients for already used features
        used_features = 1 - search_space
        coeff = 2 * int(self.theta > 0) - 1
        grads[used_features == 1] = -np.inf * coeff

        if self.theta > 0:
            ind = np.argpartition(grads, -2, axis=1)[:, -2:]
        else:
            ind = np.argpartition(-grads, -2, axis=1)[:, -2:]

        return ind

    def _check_params(self) -> None:
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError("The total perturbation percentage `gamma` must be between 0 and 1.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
