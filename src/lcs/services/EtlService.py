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
            'user_verification_level': str, 
            'email_valid': str, 
            'ip_vpn': str,
            'phone_valid': str
        }, low_memory=False)

        # used only class, clean data 
        data = data[((data['fraud_state'] == 'APPROVE') | (data['fraud_state'] == 'DECLINE'))]
        print('<<< EtlService:EtlService: The shape of data:', data.shape)
        
        return data

    def getFilterData(self, data):
        properties = self.eda.getProperties()
        return pd.DataFrame(data, columns=properties )
           
    def featureEngineering(self, data, action='train'):
        # Missing value
        data = self.missingValue(data)
        print('<<< EtlService:featureEngineering: missingValue:', data)
        # Same format
        data = self.sameFormat(data)
        print('<<< EtlService:featureEngineering: sameFormat:', data)
        
        # Checking outlier values
        outlierFields = self.eda.getOutlierFields()
        if action == 'train':
            data = self.iqrChekOutlierValues(data, outlierFields)
        else:
            data = self.jenksBreakMethodClasify(data, outlierFields)
        # Featurization data
        print('<<< EtlService:featureEngineering: jenksBreakMethodClasify:', data)
        data = self.featurizationData(data)
        print('<<< EtlService:featureEngineering: featurizationData:', data)
        return data

    def generate(self):
        data = self.readDataSource()
        data = self.featureEngineering(data)
        
        path = os.path.dirname(__file__) + "../../../../data/"
        data.to_csv(path + 'datamining_view.csv')
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
    def iqrChekOutlierValues(self, data, outlierFields):
        for item in outlierFields:
            # create nominal intervals 
            print('>>> EtlService:iqrChekOutlierValues >>>')
            
            # checking outlier values
            q1_amount = data[item['name']].quantile(.25)
            q3_amount = data[item['name']].quantile(.75)
            IQR_amount = q3_amount - q1_amount
            print('transaction amount IQR: ', IQR_amount)

            # defining limits
            sup_amount = q3_amount + 1.5 * IQR_amount
            inf_amount = q1_amount - 1.5 * IQR_amount
            print('transaction amount Upper limit: ', sup_amount)
            print('transaction amount Lower limit: ', inf_amount)

            # cleaning the outliers in 'transaction amount' values
            data_clean_transaction = data.copy()
            data_clean_transaction.drop( data_clean_transaction[data_clean_transaction[item['name']] > sup_amount].index, axis=0, inplace=True)
            data_clean_transaction.drop( data_clean_transaction[data_clean_transaction[item['name']] < inf_amount].index, axis=0, inplace=True)
    
            data = self.jenksBreakMethodTrain(item, data_clean_transaction, data)
            
        return data
    
    # generate Outlier 
    def generateOutlierModel(self, feature, dataByFeature, data):
        labels = feature['labels']
        breaks = jenkspy.jenks_breaks(dataByFeature[feature['name']], nb_class=len(labels))
        minValue = data[feature['name']].min()
        maxvalue = data[feature['name']].max()

        if breaks[0] != minValue:
            breaks.insert(0, minValue)
            labels.insert(0, 'outlier-left')

        if breaks[len(breaks)-1] != maxvalue:
            breaks.append(maxvalue)
            labels.append('outlier-right')

        numb_Bins = len(breaks) - 1
        
        outlierData = {
            "breaks": breaks,
            "labels": labels
        }

        print('>>> EtlService:generateOutlierModel:breaks >>>', breaks)
        print('>>> EtlService:generateOutlierModel:numb_Bins >>>', numb_Bins)

        filename = feature['name'].replace(" ", "_")
        self.save_object("data/datamining_outlier_" + filename, outlierData)
        
        print('>>> EtlService:generateOutlierModel!')
        return outlierData
    
    # avoid Outlier 
    def avoidOutlier(self, feature, data, outlierData):
        return pd.cut(data[feature['name']], bins=outlierData['breaks'], labels=outlierData["labels"], include_lowest=True)
    
    # avoid Outlier from features
    def jenksBreakMethodTrain(self, feature, dataByFeature, data):
        # with cleaning outliers 'transaction amount'
        outlierData = self.generateOutlierModel(feature, dataByFeature, data)
        data[feature['name']] =  self.avoidOutlier(feature, data, outlierData)
        return data
        
    # avoid Outlier from features based on jenksBreak Method
    def jenksBreakMethodClasify(self, data, features):
        print('>>> EtlService:jenksBreakMethodClasify') 
        # with cleaning outliers 'transaction amount'
        print('>>> EtlService:jenksBreakMethodClasify data >>>', data)
        for item in features:
            filename = item['name'].replace(" ", "_")
            outlierData = self.load_object("data/datamining_outlier_" + filename)
            data[item['name']] =  self.avoidOutlier(item, data, outlierData)
        return data

    # Featurizing the data
    def featurizationData(self, dataDeposits): # revisarlo en funcion de las caracteristicas
        print('>>> EtlService:featurizationData >>>')
        # Separation of columns into numeric and categorical columns
        types = np.array([dt for dt in dataDeposits.dtypes])
        all_columns = dataDeposits.columns.values
        is_num = (types != 'object')
        is_category = (types != 'object') & (types != 'float64') & (types != 'int64')
        isClass = all_columns == 'fraud_state'
        isDiscretization = (all_columns == 'user_balance') | (all_columns == 'transaction_amount')
        num_cols = all_columns[is_num & ~isDiscretization]
        cat_cols = all_columns[~is_num & ~isClass]
        category_cols = all_columns[is_category]

        print('>>> EtlService:featurizationData >>>', cat_cols)
        print('>>> EtlService:featurizationData >>>', category_cols)

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
                (data[column][data['fraud_state'] == label].value_counts() / data[column].value_counts()).fillna(0))
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
        data['fraud_state'] = data['fraud_state'].replace(['APPROVE'], 0)
        data['fraud_state'] = data['fraud_state'].replace(['DECLINE'], 1)
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