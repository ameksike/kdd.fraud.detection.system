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

        # used only class, clean data 
        data = data[((data['fraud state'] == 'APPROVE') | (data['fraud state'] == 'DECLINE'))]
        print('<<< EtlService:EtlService: The shape of data:', data.shape)
        
        return data

    def featureEngineering(self, data, action='train'):
        # Missing value
        data = self.missingValue(data)
        # Same format
        data = self.sameFormat(data)
        # Checking outlier values
        data = self.iqrChekOutlierValues(data, action)
        # Featurization data
        data = self.featurizationData(data)
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

    #replace tags to get same format
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
    
    #format variables with outlier problems, obtain the limits that go from the mean, then create ranges with those values and are labeled
    def iqrChekOutlierValues(self, dataDeposits, action='train'):
        print('>>> EtlService:iqrChekOutlierValues >>>')
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
        if action == 'train':
            dataDeposits = self.jenksBreakMethodTrain('transaction amount', dataDeposits_clean_transaction, dataDeposits)
            dataDeposits = self.jenksBreakMethodTrain('user balance', dataDeposits_clean_balance, dataDeposits)
        else:
            dataDeposits = self.jenksBreakMethodClasify('transaction amount', dataDeposits_clean_transaction, dataDeposits)
            dataDeposits = self.jenksBreakMethodClasify('user balance', dataDeposits_clean_balance, dataDeposits)
            
        return dataDeposits

    def generateOutlierModel(self, name_feature, dataDepositByFeature, dataDeposits):
        print('>>> EtlService:generateOutlierModel >>>')
        labels = ['small', 'medium', 'big']
        breaks = jenkspy.jenks_breaks(dataDepositByFeature[name_feature], nb_class=3)
        minValue = dataDeposits[name_feature].min()
        maxvalue = dataDeposits[name_feature].max()

        if breaks[0] != minValue:
            breaks.insert(0, minValue)
            labels.insert(0, 'outlier-left')

        if breaks[len(breaks)-1] != maxvalue:
            breaks.append(maxvalue)
            labels.append('outlier-right')

        numb_Bins = len(breaks) - 1

        print(breaks)
        print(numb_Bins)
        
        outlierData = {
            "breaks": breaks,
            "labels": labels
        }

        filename = name_feature.replace(" ", "_")
        self.save_object("data/train_outlier_" + filename, outlierData)
        print('>>> EtlService:generateOutlierModel!')
        return outlierData
    
    def avoidOutlier(self, name_feature, dataDeposits, outlierData):
        return pd.cut(dataDeposits[name_feature], bins=outlierData['breaks'], labels=outlierData["labels"], include_lowest=True)
    
    # avoid Outlier from features
    def jenksBreakMethodTrain(self, name_feature, dataDepositByFeature, dataDeposits):
        print('>>> EtlService:jenksBreakMethodTrain >>>')
        # with cleaning outliers 'transaction amount'
        outlierData = self.generateOutlierModel(dataDepositByFeature, dataDeposits)
        dataDeposits[name_feature] =  self.avoidOutlier(name_feature, dataDeposits, outlierData)
        return dataDeposits
        
    # avoid Outlier from features
    def jenksBreakMethodClasify(self, name_feature, dataDepositByFeature, dataDeposits):
        print('>>> EtlService:jenksBreakMethodClasify >>>')
        # with cleaning outliers 'transaction amount'
        filename = name_feature.replace(" ", "_")
        outlierData = self.load_object(filename)
        dataDeposits[name_feature] =  self.avoidOutlier(name_feature, dataDeposits, outlierData)
        return dataDeposits

    # Featurizing the data
    def featurizationData(self, dataDeposits):
        print('>>> EtlService:featurizationData >>>')
        # Separation of columns into numeric and categorical columns
        types = np.array([dt for dt in dataDeposits.dtypes])
        all_columns = dataDeposits.columns.values
        is_num = (types != 'object')
        is_category = (types != 'object') & (types != 'float64') & (types != 'int64')
        isClass = all_columns == 'fraud state'
        isDiscretization = (all_columns == 'user balance') | (all_columns == 'transaction amount')
        num_cols = all_columns[is_num & ~isDiscretization]
        cat_cols = all_columns[~is_num & ~isClass]
        category_cols = all_columns[is_category]

        print(cat_cols)
        print(category_cols)

        # Featurization of categorical data
        # calling the above defined functions
        # categorical columns to perform response coding on

        for col in cat_cols:
            # extracting the dictionary with values corresponding to TARGET variable 0 and 1 for each of the categories
            mapping_dictionary = self.responseFit(dataDeposits, col)
            # mapping this dictionary with our DataFrame
            self.responseTransform(dataDeposits, col, mapping_dictionary)
            # removing the original categorical columns
            _ = dataDeposits.pop(col)

        for col in category_cols:
            # extracting the dictionary with values corresponding to TARGET variable 0 and 1 for each of the categories
            mapping_dictionary = self.responseFit(dataDeposits, col)
            # mapping this dictionary with our DataFrame
            self.responseTransform(dataDeposits, col, mapping_dictionary)
            # removing the original categorical columns
            _ = dataDeposits.pop(col)

        return dataDeposits

    def responseFit(self, data, column):
        '''
        Response Encoding Fit Function
        Function to create a vocabulary with the probability of occurrence of each category for categorical features
        for a given class label.
        Inputs:
            self
            data: DataFrame
                training Dataset
            column: str
                the categorical column for which vocab is to be generated
        Returns:
            Dictionary of probability of occurrence of each category in a particular class label.
        '''
        dict_occurrences = {'APPROVE': {}, 'DECLINE': {}}
        for label in ['DECLINE', 'APPROVE']:
            dict_occurrences[label] = dict(
                (data[column][data['fraud state'] == label].value_counts() / data[column].value_counts()).fillna(0))
        return dict_occurrences

    def responseTransform(self, data, column, dict_mapping):
        '''
        Response Encoding Transform Function
        Function to transform the categorical feature into two features, which contain the probability
        of occurrence of that category for each class label.
        Inputs:
            self
            data: DataFrame
                DataFrame whose categorical features are to be encoded
            column: str
                categorical column whose encoding is to be done
            dict_mapping: dict
                Dictionary obtained from Response Fit function for that particular column
        Returns:
            DataFrame with encoded categorical feature.
        '''
        data[column + '_DECLINE'] = data[column].map(dict_mapping['DECLINE'])
        data[column + '_APPROVE'] = data[column].map(dict_mapping['APPROVE'])
        # print(data[column + '_DECLINE'])
        # print(data[column + '_APPROVE'])

    def replaceClassValue(self, data):
        # Replace class value: 'APPROVE' = 0, 'DECLINE' = 1
        data['fraud state'] = data['fraud state'].replace(['APPROVE'], 0)
        data['fraud state'] = data['fraud state'].replace(['DECLINE'], 1)
        return data

    def save_object(self, filename, model):
        print('>>> EtlService:save_object >>>', 'Done !!!')
        with open(filename, 'wb') as file:
            joblib.dump(model, filename)

    def load_object(self, filename):
        with open(filename, 'rb') as f:
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