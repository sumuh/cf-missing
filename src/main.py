import pandas as pd
import numpy as np

from .classifier import Classifier
from .counterfactual_generator import CounterfactualGenerator
from .imputer import Imputer
from .data_utils import (
    load_data,
    explore_data,
    get_cat_indices,
    get_num_indices,
    get_indices_with_missing_values,
    get_target_index,
    get_test_input_1,
    get_test_input_2,
)


def main():
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

    indices_with_missing_values = get_indices_with_missing_values(input)
    if len(indices_with_missing_values) > 0:
        imputer = Imputer()
        input = imputer.mean_imputation(data_np, input, indices_with_missing_values)

    print(f"Input: {input}")
    prediction, probability = classifier.predict_with_proba(input)
    print(f"Prediction was {prediction}, probability of 1 was {probability}\n")

    cf_generator = CounterfactualGenerator(classifier, data, target_index)

    if prediction == 1:
        counterfactuals = cf_generator.generate_explanations(
            input, indices_with_missing_values, 3
        )

    print(f"Counterfactual: {counterfactuals[0]}")
    prediction_cf, probability_cf = classifier.predict_with_proba(counterfactuals[0])
    print(f"Prediction was {prediction_cf}, probability of 1 was {probability_cf}\n")


if __name__ == "__main__":
    main()
