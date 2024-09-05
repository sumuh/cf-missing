import numpy as np
import json
import os

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import BayesianRidge


class HyperparamOptimizer:

    def __init__(self):
        self.imputation_models_hyperparams = {}
        self.imputation_model_hyperparams_file_path = ""

    def get_best_hyperparams_for_imputation_models(
        self,
        get_from_file: bool,
    ) -> dict[int, dict[str, any]]:
        """Returns best hyperparameters BayesianRidge imputation model for each feature.
        Assumes run_hyperparam_optimization_for_imputation_models has been run.

        :param bool get_from_file: read hyperparams from file
        :return dict[int, dict[str, any]]: result dict where key is feature index and value
        is dict of hyperparameters
        """
        if get_from_file:
            current_file_path = os.path.dirname(os.path.realpath(__file__))
            target_file_path = f"{current_file_path}/imputation_model_hyperparams.json"
            with open(target_file_path, "r") as f:
                hyperparams_dict = json.load(f)
                return {int(k): v for k, v in hyperparams_dict.items()}
        else:
            if len(self.imputation_models_hyperparams.keys()) == 0:
                raise RuntimeError("Imputation model hyperparams not initialized!")
            return self.imputation_models_hyperparams

    def run_hyperparam_optimization_for_imputation_models(
        self, train_data_without_target: np.array
    ):
        """Chooses best hyperparameters for BayesianRidge imputation model for each feature
        and stores them in self.imputation_models_hyperparams

        :param np.array train_data_without_target: dataset
        """
        param_grid = {
            "max_iter": [10, 100, 300, 500, 1000],
            "tol": [
                1e1,
                1,
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
                1e-8,
                1e-9,
                1e-10,
            ],
            "alpha_1": [
                1e1,
                1,
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
                1e-8,
                1e-9,
                1e-10,
            ],
            "alpha_2": [
                1e1,
                1,
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
                1e-8,
                1e-9,
                1e-10,
            ],
            "lambda_1": [
                1e1,
                1,
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
                1e-8,
                1e-9,
                1e-10,
            ],
            "lambda_2": [
                1e1,
                1,
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
                1e-8,
                1e-9,
                1e-10,
            ],
            "fit_intercept": [True, False],
        }
        for feat_ind in range(train_data_without_target.shape[1]):
            mask = np.ones(train_data_without_target.shape[1], dtype=bool)
            mask[feat_ind] = False
            X = train_data_without_target[:, mask]
            y = train_data_without_target[:, feat_ind]
            model = BayesianRidge()
            # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grid, cv=5, n_jobs=-1
            )
            grid_search.fit(X, y)
            best_params_for_feat = grid_search.best_params_
            self.imputation_models_hyperparams[feat_ind] = best_params_for_feat
