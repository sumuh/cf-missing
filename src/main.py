import pandas as pd
import numpy as np
import os
from datetime import datetime
from .classifier import Classifier
from .counterfactual_generator import CounterfactualGenerator
from .imputer import Imputer
from .evaluation import CounterfactualEvaluator, LoocvEvaluator
from .data_utils import (
    load_data,
    explore_data,
    get_cat_indices,
    get_num_indices,
    get_indices_with_missing_values,
    get_target_index,
    get_test_input_1,
    get_test_input_2,
    get_diabetes_dataset_config,
    get_wine_dataset_config,
    transform_target_to_binary_class,
)
from .constants import *


def test_single_instance():
    data = load_data()
    print(data)
    # explore_data(data)
    target_index = get_target_index(data, "Outcome")
    cat_indices = get_cat_indices(data, target_index)
    num_indices = get_num_indices(data, target_index)
    predictor_indices = np.sort(np.concatenate([cat_indices, num_indices]))
    X_train = data.iloc[:, predictor_indices].to_numpy()
    y_train = data.iloc[:, target_index].to_numpy().ravel()
    data_np = data.to_numpy()

    classifier = Classifier(num_indices)
    classifier.train(X_train, y_train)

    # input = pd.DataFrame([get_test_input_1()])
    input = pd.DataFrame([get_test_input_2()])
    # Change single value to missing
    input.at[0, "Insulin"] = pd.NA
    input = input.to_numpy().ravel()

    n = 1
    indices_with_missing_values = get_indices_with_missing_values(input)
    if len(indices_with_missing_values) > 0:
        imputer = Imputer()
        # input = imputer.mean_imputation(data_np, input, indices_with_missing_values, n)
        input = imputer.multiple_imputation(
            data_np, input, indices_with_missing_values, target_index, n
        )
    else:
        input = np.array([input])

    print(f"Input: {input}")

    for row in range(input.shape[0]):
        prediction, probability = classifier.predict_with_proba(input[row, :])
        print(f"Prediction was {prediction}, probability of 1 was {probability}\n")

    cf_generator = CounterfactualGenerator(classifier, data, target_index)

    if prediction == 1:
        counterfactuals = cf_generator.generate_explanations(
            input, indices_with_missing_values, 3
        )

        if counterfactuals.ndim == 1:
            counterfactuals = np.array([counterfactuals])

        evaluator = Evaluator(X_train)
        evaluator.evaluate_explanation(input[0], counterfactuals, classifier.predict, 0)


def main():
    SAVE_TO_FILE = True
    DEBUG = False

    if SAVE_TO_FILE:
        results_dir = (
            f"{os.path.dirname(os.path.realpath(__file__))}/../evaluation_results"
        )
        current_time = datetime.now()
        formatted_time = current_time.strftime("%d-%m-%Y-%H-%M-%S")
        results_filename = f"{results_dir}/results-{formatted_time}.txt"

    data_config = get_wine_dataset_config()
    data = load_data(data_config[config_file_path], data_config[config_separator])

    if data_config[config_multiclass_target]:
        data = transform_target_to_binary_class(
            data,
            data_config[config_target_name],
            data_config[config_multiclass_threshold],
        )

    evaluation_config = {
        config_classifier: config_logistic_regression,
        config_missing_data_mechanism: config_MCAR,
        config_dataset_name: data_config[config_dataset_name],
        config_target_name: data_config[config_target_name],
        config_multiclass_target: data_config[config_multiclass_target],
        config_debug: DEBUG,
    }

    loocv_evaluator = LoocvEvaluator(data, evaluation_config)
    results = loocv_evaluator.perform_loocv_evaluation()

    print(f"config: {evaluation_config}")
    print(f"results: {results}")

    if SAVE_TO_FILE:
        # Pretty print config and results to file
        with open(results_filename, "w") as results_file:
            config_str = "config\n" + "\n".join(
                [f"{item[0]}\t{item[1]}" for item in evaluation_config.items()]
            )
            results_str = "results\n" + "\n".join(
                [f"{item[0]}\t{item[1]}" for item in results.items()]
            )
            results_file.write(f"{config_str}\n\n{results_str}")


if __name__ == "__main__":
    main()
