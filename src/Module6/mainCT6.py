# File: mainCT6.py
# Written by: Angel Hernandez
# Description: Module 6 - Critical Thinking
#
# Requirement(s): Using scikit-learn, write a Naive Bayes classifier in Python. It can be single or multiple features.
# Submit the classifier in the form of an executable Python script alongside basic instructions for testing.
#
# Your Naive Bayes classification script should allow you to do the following:
#
# -- Calculate the posterior probability by converting the dataset into a frequency table.
# -- Create a "Likelihood" table by finding relevant probabilities.
# -- Calculate the posterior probability for each class.
# -- Correct Zero Probability errors using Laplacian correction.
#
# Your classifier may use a Gaussian, Multinomial, or Bernoulli model, depending on your chosen function.
# Your classifier must properly display its probability prediction based on its input data.

import os
import sys
import subprocess

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class DataFactory:
    @staticmethod
    def get_dataset():
        data = {
            "EmailAddress": [
                "alice@gmail.com", "angel@hotmail.com", "john@ms.net",
                "support@techsupport.com", "carol@contoso.com", "customersupport@mysite.io",
                "richard@yahoo.com", "subscription@newsfeed.net", "linda@mail.net",
                "zach@education.io"
            ],
            "ContainsLinks": ["Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No"],
            "SenderKnown": ["Yes", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
            "Spam": ["No", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No"]
        }

        return data

class NaiveBayesClassifierForSpamDetection:
    def __init__(self):
        import numpy as np
        import pandas as pd
        import joblib as jl
        from sklearn.naive_bayes import MultinomialNB as mnb
        self.__np = np
        self.__pd = pd
        self.__jl = jl
        self.__mnb = mnb
        self.__model = None
        self.__target = None
        self.__frequency = None
        self.__posteriors = {}
        self.__likelihoods = {}
        self.__data_frame = None
        self.__encoded_data_frame = None

    def run_orchestration(self):
       self.__load_data()
       self.__show_frequency_table()
       self.__show_likelihood_table_with_laplace_correction()
       self.__train_multinomial_naive_bayes_model()
       self.__test_classifier()
       self.__save_model();

    def __load_data(self):
        self.__data_frame = self.__pd.DataFrame(DataFactory.get_dataset())
        self.__encoded_data_frame = self.__pd.get_dummies(self.__data_frame[["ContainsLinks", "SenderKnown"]])
        self.__target = self.__data_frame["Spam"].map({"No": 0, "Yes": 1})

    def __show_frequency_table(self):
        print("===> Frequency Table\n")
        self.__frequency = self.__data_frame.groupby(["Spam"]).size()
        print(self.__frequency)
        print()

    def __save_model(self):
       save_model = input("Do you want to save the model (y/n)? Default is n: ")
       save_model = "n" if save_model == "n" or not save_model else "y"

       if save_model == "y":
           self.__jl.dump(self.__model, "module6_model.joblib")

    def __test_classifier(self):
        print("\n===> Prediction from Trained Model")

        test_parameter = {
            "ContainsLinks_Yes": 1,
            "ContainsLinks_No": 0,
            "SenderKnown_Yes": 0,
            "SenderKnown_No": 1
        }

        test_data_frame = self.__pd.DataFrame([test_parameter])

        # Let's reindex so training feature order matches
        test_data_frame = test_data_frame.reindex(columns=self.__encoded_data_frame.columns, fill_value=0)

        # let's calculate prediction
        prediction = self.__model.predict(test_data_frame)[0]
        pred_prob = self.__model.predict_proba(test_data_frame)[0]
        print(f"\nPrediction (0=Not Spam, 1=Spam): {prediction}")
        print("Model probability:", pred_prob)
        self.__posterior_calculation()

    # https://en.wikipedia.org/wiki/Additive_smoothing
    # https://towardsdatascience.com/laplace-smoothing-in-naive-bayes-algorithm-9c237a8bdece
    def __show_likelihood_table_with_laplace_correction(self):
        alpha = 1  # Laplace smoothing
        print("\n===> Likelihood Table (with Laplace correction)\n")
        for feature in self.__encoded_data_frame.columns:
            self.__likelihoods[feature] = {}
            for cls in [0, 1]:
                count = self.__encoded_data_frame[feature][self.__target == cls].sum()
                total = sum(self.__target == cls)
                prob = (count + alpha) / (total + 2)
                self.__likelihoods[feature][cls] = prob
        likelihood_dataframe = self.__pd.DataFrame(self.__likelihoods)
        print(likelihood_dataframe)
        print()

    def __train_multinomial_naive_bayes_model(self):
        self.__model = self.__mnb(alpha=1)
        self.__model.fit(self.__encoded_data_frame, self.__target)

    def __posterior_calculation(self):
        print("\n===> Posterior Calculation")
        _dict = {"ContainsLinks": "Yes", "SenderKnown": "No"}
        encoded_input = self.__pd.DataFrame([_dict])
        encoded_input = self.__pd.get_dummies(encoded_input).reindex(columns=self.__encoded_data_frame.columns, fill_value=0)

        priors = {
            0: self.__frequency["No"] / len(self.__data_frame),
            1: self.__frequency["Yes"] / len(self.__data_frame)
        }

        for cls in [0, 1]:
            posterior = priors[cls]
            print(f"\nClass {cls} initial prior: {posterior:.4f}")
            for feature in self.__encoded_data_frame.columns:
                if encoded_input[feature].iloc[0] == 1:
                    posterior *= self.__likelihoods[feature][cls]
                    print(f"  * {feature} contributes {self.__likelihoods[feature][cls]:.4f}")
            self.__posteriors[cls] = posterior
        print("\nPosterior probabilities (unnormalized):", self.__posteriors)
        total = sum(self.__posteriors.values())
        normalized = {cls: self.__posteriors[cls] / total for cls in self.__posteriors}
        print("\nNormalized posterior probabilities:", normalized)
        print()

        return normalized

class TestCaseRunner:
    @staticmethod
    def run_test():
        nb_classifier = NaiveBayesClassifierForSpamDetection()
        nb_classifier.run_orchestration()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'pandas', 'scikit-learn', 'joblib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 6 - Critical Thinking ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
