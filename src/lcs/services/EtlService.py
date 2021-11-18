from services.SingletonMeta import SingletonMeta
import pandas as pd
import numpy as np
import jenkspy
import joblib
from sklearn import linear_model
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
        # Missing value
        data = self.missingValue(data)
        # Same format
        data = self.sameFormat(data)
        # Checking outlier values
        data = self.iqrChekOutlierValues(data)
        # Featurization data
        #data = self.featurizationData(data)
        return data

    def generate(self):
        data = self.readDataSource()
        data = self.featureEngineering(data)
        
        path = os.path.dirname(__file__) + "../../../../data/"
        data.to_csv(path + 'dataMiningView.csv')
        return data.shape

    #description set 'none' to empty string, and set 0 to empty number
    def missingValue(self, data):
        dataObj = data.select_dtypes(include=np.object).columns.tolist()
        # data[dataObj] = data[dataObj].astype('string')
        data[dataObj] = data[dataObj].fillna('none')
        obj_columnsFloat = data.select_dtypes(include=np.float64).columns.tolist()
        data[obj_columnsFloat] = data[obj_columnsFloat].fillna(0)
        return data

    def sameFormat(self, data):
        # Select columns which contains any value feature: 'OTHER', 'CURL', 'NONE'
        filter = ((data == 'OTHER') | (data == 'CURL') | (data == 'NONE')).any()
        obj_columnsReplace = data.loc[:, filter].columns.tolist()
        # unify feature value
        data[obj_columnsReplace] = data[obj_columnsReplace].replace(['OTHER'], 'Other')
        data[obj_columnsReplace] = data[obj_columnsReplace].replace(['NONE'], 'none')
        data[obj_columnsReplace] = data[obj_columnsReplace].replace(['CURL'], 'curl')
        filterComprobation = ((data == 'OTHER') | (data == 'CURL') | (data == 'NONE')).any()
        print(len(data.loc[:, filterComprobation].columns.tolist()))
        return data

    def iqrChekOutlierValues(self, dataDeposits):
        # checking outlier values
        q1_amount_transaction_amount = dataDeposits['transaction amount'].quantile(.25)
        q3_amount_transaction_amount = dataDeposits['transaction amount'].quantile(.75)
        IQR_amount = q3_amount_transaction_amount - q1_amount_transaction_amount
        print('transaction amount IQR: ', IQR_amount)

        # defining limits
        sup_amount_transaction_amount = q3_amount_transaction_amount + 1.5 * IQR_amount
        inf_amount_transaction_amount = q1_amount_transaction_amount - 1.5 * IQR_amount
        print('transaction amount Upper limit: ', sup_amount_transaction_amount)
        print('transaction amount Lower limit: ', inf_amount_transaction_amount)

        # checking outlier values
        q1_amount_user_balance = dataDeposits['user balance'].quantile(.25)
        q3_amount_user_balance = dataDeposits['user balance'].quantile(.75)
        IQR_amount = q3_amount_user_balance - q1_amount_user_balance
        print('user balance IQR: ', IQR_amount)
        # defining limits
        sup_amount_user_balance = q3_amount_user_balance + 1.5 * IQR_amount
        inf_amount_user_balance = q1_amount_user_balance - 1.5 * IQR_amount
        print('user balance Upper limit: ', sup_amount_user_balance)
        print('user balance Lower limit: ', inf_amount_user_balance)

        # cleaning the outliers in 'transaction amount' values
        dataDeposits_clean_transaction = dataDeposits.copy()
        dataDeposits_clean_transaction.drop(
            dataDeposits_clean_transaction[dataDeposits_clean_transaction['transaction amount'] >
                                        sup_amount_transaction_amount].index, axis=0, inplace=True)
        dataDeposits_clean_transaction.drop(
            dataDeposits_clean_transaction[dataDeposits_clean_transaction['transaction amount'] <
                                        inf_amount_transaction_amount].index, axis=0, inplace=True)

        # cleaning the outliers in 'user balance' values
        dataDeposits_clean_balance = dataDeposits.copy()
        dataDeposits_clean_balance.drop(dataDeposits_clean_balance[dataDeposits_clean_balance['user balance'] >
                                                                sup_amount_user_balance].index, axis=0, inplace=True)
        dataDeposits_clean_balance.drop(dataDeposits_clean_balance[dataDeposits_clean_balance['user balance'] <
                                                                inf_amount_user_balance].index, axis=0, inplace=True)

        # create nominal intervals 
        dataDeposits = self.jenksBreakMethod('transaction amount', dataDeposits_clean_transaction, dataDeposits)
        dataDeposits = self.jenksBreakMethod('user balance', dataDeposits_clean_balance, dataDeposits)

        return dataDeposits

    def jenksBreakMethod(self, name_feature, dataDepositByFeature, dataDeposits):
        # with cleaning outliers 'transaction amount'
        labels = ['small', 'medium', 'big', 'outlier']
        breaks = jenkspy.jenks_breaks(dataDepositByFeature[name_feature], nb_class=3)
        breaks.append(dataDeposits[name_feature].max()) # tener en cuenta el el rango minimo
        numb_Bins = len(breaks) - 1

        print(breaks)
        print(numb_Bins)

        dataDeposits[name_feature] = pd.cut(dataDeposits[name_feature], bins=breaks, labels=labels, include_lowest=True)
        return dataDeposits


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