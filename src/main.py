import pandas as pd
import os 

from classifier import Classifier
from counterfactual_generator import CounterfactualGenerator
from data_utils import load_data, explore_data, get_predictor_names, get_target_name, get_test_input_1, get_test_input_2

def main():
    data = load_data()
    explore_data(data)
    target_name = get_target_name()
    predictor_names = get_predictor_names(data, target_name)

    classifier = Classifier(data, predictor_names, target_name)
    classifier.train()

    #input = get_test_input_1()
    input = get_test_input_2()
    print(f"Input: {input}\n")
    prediction, probability = classifier.predict(input)
    print(f"Prediction was {prediction}, probability of 1 was {probability}\n")

    """ cf_generator = CounterfactualGenerator(classifier, data)

    if prediction == 1:
        counterfactuals = cf_generator.generate_explanation(input, 3)
        print(counterfactuals) """

if __name__ == "__main__":
    main()