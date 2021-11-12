from services.SingletonMeta import SingletonMeta
import pandas as pd
import numpy as np
import joblib
from sklearn import linear_model
from sympy.ntheory import primefactors as pf

class EtlService(metaclass=SingletonMeta):

    def generate(self, sample_size):
        x_data = np.array([self.factors_prime_encode(i) for i in range(101, 101 + int(sample_size))])
        y_target = np.array([[self.fizzbuzz(i)] for i in range(101, 101 + int(sample_size))])
        z_samples = np.append(x_data, y_target, axis=1)
        df = pd.DataFrame(data=z_samples, columns=["2", "3", "5", "7", "11", "13", "Class"])
        return df

    # The ground truth (correct) target values of the first 100 numbers
    def generate_first100_fizz_buzz(self):
        sample_fizz_buzz = np.array([self.factors_prime_encode(i) for i in range(1, 100 + 1)])
        target_fizz_buzz = np.array([[self.fizzbuzz(i)] for i in range(1, 100 + 1)])
        z_fizz_buzz = np.append(sample_fizz_buzz, target_fizz_buzz, axis=1)
        df = pd.DataFrame(data=z_fizz_buzz, columns=["2", "3", "5", "7", "11", "13", "Class"])
        df.to_csv('fist100FizzBuzz_ground_truth.csv', index=False)

    def switch_index_encode(self, argument):
        switcher = {
            2: 0,
            3: 1,
            5: 2,
            7: 3,
            11: 4,
            13: 5
        }
        return switcher.get(argument, -1)

    def factors_prime_encode(self, number):
        factors_x = pf(number)
        code = [0] * 6
        for factor in factors_x:
            index = self.switch_index_encode(factor)
            if index != -1:
                code[index] = 1
        return code

    def fizzbuzz(self, i):
        if i % 15 == 0:
            return 1
        if i % 5 == 0:
            return 2
        if i % 3 == 0:
            return 3
        return 4

    def switch_fizz_buzz(self, argument):
        switcher = {
            4: "None",
            3: "Fizz",
            2: "Buzz",
            1: "FizzBuzz"
        }
        return switcher.get(argument, -1)

    def save_object(self, filename, model):
        with open('' + filename, 'wb') as file:
            joblib.dump(model, filename)

    def load_object(self, filename):
        with open('' + filename, 'rb') as f:
            loaded = joblib.load(f)
        return loaded

    def create_model(self):
        model_lR = linear_model.LogisticRegression(
            C=1.0, class_weight=None, dual=False,
            fit_intercept=True, intercept_scaling=1, max_iter=1000,
            multi_class='ovr',
            n_jobs=1, penalty='l2',
            random_state=None,
            solver='liblinear',
            tol=0.0001,
            verbose=0,
            warm_start=False)
        return model_lR