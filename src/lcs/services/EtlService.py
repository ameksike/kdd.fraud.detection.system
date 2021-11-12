from services.SingletonMeta import SingletonMeta
import pandas as pd
import numpy as np
import joblib
from sklearn import linear_model
from sympy.ntheory import primefactors as pf
import os

class EtlService(metaclass=SingletonMeta):
    def setEda(self, eda):
        self.eda = eda

    def readDataSource(self):
        filename = os.path.dirname(__file__) + "../../../../data/data_source.csv"
        filename = os.path.abspath(filename)
        properties = self.eda.getProperties()

        data = pd.read_csv(filename, usecols=properties, dtype={
            'user verification level': str, 
            'email valid': str, 
            'ip vpn': str,
            'phone valid': str
        }, low_memory=False)
        
        return data

    def featureEngineering(self, data):
        return data

    def generate(self):
        data = self.readDataSource()
        data = self.featureEngineering(data)
        
        path = os.path.dirname(__file__) + "../../../../data/"
        data.to_csv(path + 'dataMiningView.csv')
        return data.shape



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