import numpy as np
import json
from sklearn.linear_model import BayesianRidge
from scipy import stats
from .utils.data_utils import get_feature_min_values, get_feature_max_values
from .hyperparams.hyperparam_optimization import HyperparamOptimizer


class Imputer:

    def __init__(
        self,
        train_data: np.array,
        hyperparam_opt: HyperparamOptimizer = None,
        init_multiple_imputation: bool = False,
        debug: bool = False,
    ):
        if init_multiple_imputation:
            self.fcs_models = self._init_fcs_models(train_data, hyperparam_opt)
            self.min_values = get_feature_min_values(train_data)
            self.max_values = get_feature_max_values(train_data)
        self.feature_means = np.apply_along_axis(np.mean, 0, train_data)
        self.debug = debug

    def _init_fcs_models(
        self, train_data: np.array, hyperparam_opt: HyperparamOptimizer
    ) -> list[BayesianRidge]:
        """Initializes feature-specific imputation models for FCS multiple imputation.

        :param np.array train_data: train dataset without target
        :param HyperparamOptimizer hyperparam_opt: object containing optimized hyperparams
        :return list[BayesianRidge]: list of models
        """
        models = []
        hyperparam_dicts = hyperparam_opt.get_best_hyperparams_for_imputation_models(
            True
        )
        for feat_ind in range(train_data.shape[1]):
            hyperparam_dict = hyperparam_dicts[feat_ind]
            mask = np.ones(train_data.shape[1], dtype=bool)
            mask[feat_ind] = False
            X = train_data[:, mask]
            y = train_data[:, feat_ind]
            model = BayesianRidge(
                max_iter=hyperparam_dict["max_iter"],
                tol=hyperparam_dict["tol"],
                alpha_1=hyperparam_dict["alpha_1"],
                alpha_2=hyperparam_dict["alpha_2"],
                lambda_1=hyperparam_dict["lambda_1"],
                lambda_2=hyperparam_dict["lambda_2"],
                fit_intercept=hyperparam_dict["fit_intercept"],
            ).fit(X, y)
            models.append(model)
        return models

    def mean_imputation(
        self,
        input: np.array,
        indices_with_missing_values: np.array,
    ) -> np.array:
        """Returns input with missing value imputed with dataset mean.

        :param np.array input: 1D input array
        :param np.array indices_with_missing_values: indices of input where value is missing
        :return np.array: imputed version of input
        """
        input_copy = input.copy()
        input_copy[indices_with_missing_values] = self.feature_means[
            indices_with_missing_values
        ]
        return input_copy

    def subgroup_mean_imputation(
        self,
        input: np.array,
        indices_with_missing_values: np.array,
    ) -> np.array:
        raise NotImplementedError

    def regression_imputation(
        self,
        input: np.array,
        indices_with_missing_values: np.array,
        target_feature: str,
    ) -> np.array:
        raise NotImplementedError

    def _fcs_multiple_impute_input(
        self,
        input: np.array,
        indices_with_missing_values: np.array,
    ) -> np.array:
        """Applied multiple imputation based on pre-defined model for each variable.
        Corresponds to method 6 of Hoogland et al. (2020) with the exception that the number
        of iterations is limited.

        :param np.array input: input array initialized to some imputed values
        :param np.array indices_with_missing_values: indices that contained missing values before imputation
        :return np.array: multiply imputed input
        """
        # Adapted from sklearn IterativeImputer: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/impute/_iterative.py
        for _ in range(2):  # two iters?
            # Update each initially missing variable based on its FC model
            for feat_i in indices_with_missing_values:
                estimator = self.fcs_models[feat_i]
                estimator_input = np.delete(input, feat_i)
                mu, sigma = estimator.predict([estimator_input], return_std=True)
                if self.debug:
                    print(f"mu: {mu}, sigma: {sigma} for feat {feat_i}")
                # two types of problems: (1) non-positive sigmas
                # (2) mus outside legal range of min_value and max_value
                # (results in inf sample)
                min_value = self.min_values[feat_i]
                max_value = self.max_values[feat_i]
                if sigma < 0:
                    input[feat_i] = mu
                elif mu < min_value:
                    # input[feat_i] = min_value
                    input[feat_i] = mu
                elif mu > max_value:
                    # input[feat_i] = max_value
                    input[feat_i] = mu
                else:
                    # Random draws from distribution
                    a = (min_value - mu) / sigma
                    b = (max_value - mu) / sigma

                    truncated_normal = stats.truncnorm(a=a, b=b, loc=mu, scale=sigma)
                    sample = truncated_normal.rvs()
                    input[feat_i] = sample
        return input

    def multiple_imputation(
        self,
        input: np.array,
        indices_with_missing_values: np.array,
        n: int,
    ) -> np.array:
        """For given input, generate n imputed versions with multiple imputation.

        :param np.array input: 1D input array
        :param np.array indices_with_missing_values: indices of input where value is missing
        :param int n: number of imputed inputs to create
        :return np.array: array with n rows, each an imputed version of input
        """
        multiply_imputed_inputs = []
        for _ in range(n):
            # Initialize with feature means
            imputed_input = self.mean_imputation(input, indices_with_missing_values)
            multiply_imputed_inputs.append(
                self._fcs_multiple_impute_input(
                    imputed_input,
                    indices_with_missing_values,
                )
            )
        return np.array(multiply_imputed_inputs)
