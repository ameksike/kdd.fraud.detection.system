from services.SingletonMeta import SingletonMeta
import pandas as pd
import numpy as np
import joblib
import re
from collections import Counter
import pickle

import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

class MlService(metaclass=SingletonMeta):
    def setEtl(self, etl):
        self.etl = etl

    def classify(self, modelname, data):
        dataFrame = pd.DataFrame([data])
        dataFormated = self.etl.featureEngineering(dataFrame, 'clasify')
        print(">>>>>>>>>>>>>>>", dataFormated)

        return { "class": 1 }
    
        load_model_lr = joblib.load(modelname)
        data = self.etl.featureEngineering(dataFrame)
        
        predict = load_model_lr.predict([data])
        predict = predict[0]
        return {
            "class": int(predict),
            "label": "label"
        }

    def readDataMiningView(self, filename):
        pd.set_option('display.max_columns', None)
        print('>>> MLService:readDataMiningView >>> Reading the data AfterPreProc....', end='\n')
        df = pd.read_csv(filename, low_memory=False)
        print('>>> MLService:readDataMiningView >>> The shape of data:', df.shape)
        return df

    # set class valance 
    def randomOverSampling(self, X_train, y_train):
        # instantiating the random over sampler
        ros = RandomOverSampler()
        # resampling X, y
        X_train, y_train = ros.fit_resample(X_train, y_train)
        # new class distribution
        print('>>> MLService:randomOverSampling >>>', Counter(y_train))
        print('>>> MLService:randomOverSampling >>> Shape of X_train After Random Over Sampler Balanced Classes:', X_train.shape)
        print('>>> MLService:randomOverSampling >>> Shape of y_train After Random Over Sampler Balanced Classes:', y_train.shape)
        return X_train, y_train

    # Saving the Dataframes into CSV files for future use balanced classes
    def saveData(self, X_train, X_val, X_test, y, y_train, y_val, y_test):
        X_train.to_csv('data/train_X_train_BalancedClasses.csv')
        X_val.to_csv('data/train_X_val_final_BalancedClasses.csv')
        X_test.to_csv('data/train_X_test_final_BalancedClasses.csv')        
        print('>>> MLService:saveData >>> The shape of data:', X_train.shape)

        # Saving the numpy arrays into text files for future use
        np.savetxt('data/train_y_BalancedClasses.txt', y, fmt='%s')
        np.savetxt('data/train_y_train_BalancedClasses.txt', y_train, fmt='%s')
        np.savetxt('data/train_y_val_BalancedClasses.txt', y_val, fmt='%s')
        np.savetxt('data/train_y_test_BalancedClasses.txt', y_test, fmt='%s')
        print('>>> MLService:saveData: Done Save balanced classes files!')

    # Dividing final data into train, valid and test datasets ... Balanced classes
    def dividingData(self, df_AfterPreProc):
        # Replace class value: 'APPROVE' = 0, 'DECLINE' = 1
        df_AfterPreProc = self.etl.replaceClassValue(df_AfterPreProc)
        print('>>> MLService:dividingData >>>  Shape of data before dividing final data:', df_AfterPreProc.shape)

        # get class labels for fraud state
        y = df_AfterPreProc.pop('fraud state').values
        print('>>> MLService:dividingData >>> class_labels', y)

        # train split
        X_train, X_temp, y_train, y_temp = train_test_split(
            df_AfterPreProc.drop(columns=df_AfterPreProc.columns[0], axis=1),
            y, stratify=y, test_size=0.3, random_state=42
        )
        # validate and test split
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

        # x for vars and y for class
        print('>>> MLService:dividingData >>> Shape of X_train Before Balanced Classes:', X_train.shape)
        print('>>> MLService:dividingData >>> Shape of y_train Before Balanced Classes:', y_train.shape)
        print('>>> MLService:dividingData >>> Shape of X_val:', X_val.shape)
        print('>>> MLService:dividingData >>> Shape of X_test:', X_test.shape)

        # Instantiating the random over sampler and set balanced classes
        X_train, y_train = self.randomOverSampling(X_train, y_train)

        # Saving the Dataframes into CSV files for future use balanced classes
        self.saveData(X_train, X_val, X_test, y, y_train, y_val, y_test)


    def readDataTrainTest(self):
        # Read Data train, test, ... Balanced classes
        print('>>> MLService:readDataTrainTest >>>', 'Hello ML Models with Balanced classes !!!!!!!!!!!!!')
        print('\n')

        X_train_final = pd.read_csv('data/train_X_train_BalancedClasses.csv', header=0)
        X_train_final = X_train_final.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_train_final = X_train_final.drop(X_train_final.columns[0], axis=1)
        print('>>> MLService:readDataTrainTest >>>', X_train_final.shape)

        y_train = pd.read_csv('data/train_y_train_BalancedClasses.txt', header=None)
        print('>>> MLService:readDataTrainTest >>>', y_train.shape)

        X_val_final = pd.read_csv('data/train_X_val_final_BalancedClasses.csv')
        X_val_final = X_val_final.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_val_final = X_val_final.drop(X_val_final.columns[0], axis=1)

        y_val = pd.read_csv('data/train_y_val_BalancedClasses.txt', header=None)
        y_test = pd.read_csv('data/train_y_test_BalancedClasses.txt', header=None)

        X_test_final = pd.read_csv('data/train_X_test_final_BalancedClasses.csv', header=0)
        X_test_final = X_test_final.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_test_final = X_test_final.drop(X_test_final.columns[0], axis=1)

        print('>>> MLService:readDataTrainTest >>>', 'Done Read Data train, test, ... Balanced classes !!!')
        return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test
        
    def selectionFeatures(self, X_train_final, y_train):
        # Selection of features ... Balanced classes
        # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        # https://neptune.ai/blog/lightgbm-parameters-guide
        model_sk = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=7, learning_rate=0.01, n_estimators=2000,
                                    class_weight='balanced', subsample=0.9, colsample_bytree=0.8, n_jobs=-1)

        print('>>> MLService:selectionFeatures >>> y_train', y_train.shape)
        print('>>> MLService:selectionFeatures >>> X_train_final', X_train_final.shape)

        train_features, valid_features, train_y, valid_y = train_test_split(X_train_final, y_train, test_size=0.15, random_state=42)
        model_sk.fit(train_features, train_y, early_stopping_rounds=100, eval_set=[(valid_features, valid_y)], eval_metric='auc', verbose=200)

        print('>>> MLService:selectionFeatures >>>', 'Done!!!!!')

        # Saved features after ... Balanced classes
        feature_imp = pd.DataFrame(sorted(zip(model_sk.feature_importances_, X_train_final.columns)), columns=['Value', 'Feature'])
        features_df = feature_imp.sort_values(by="Value", ascending=False)
        selected_features = list(features_df[features_df['Value'] >= 50]['Feature'])

        # Saving the selected features into pickle file
        with open('data/train_select_featuresBalancedClasses.txt', 'wb') as fp:
            pickle.dump(selected_features, fp)

        print('>>> MLService:selectionFeatures >>>', 'The no. of features selected:', len(selected_features))
        return selected_features

    def logisticRegressionTrain(self, X_train_final, X_val_final, X_test_final, y_train, y_val, selected_features):
        # Logistic regression with selected features ... Balanced classes
        alpha = np.logspace(-4, 4, 9)
        cv_auc_score = {}

        # replace null values for data mean
        X_train_final = X_train_final.fillna(X_train_final.mean())
        X_val_final = X_val_final.fillna(X_val_final.mean())
        X_test_final = X_test_final.fillna(X_test_final.mean())

        # Check if there are missing values​in the data. The following method can be used.
        # Flase: No missing value in the feature value of the corresponding feature
        # True: There are missing values
        print('>>> MLService:logisticRegressionTrain >>>', X_train_final.isnull().any())
        print('>>> MLService:logisticRegressionTrain >>>', X_val_final.isnull().any())
        print('>>> MLService:logisticRegressionTrain >>>', X_test_final.isnull().any())

        # Check if it contains infinite data
        # False: Contains
        # True: Not included
        # print(np.isfinite(X_train_final).all())

        for i in alpha:
            clf = SGDClassifier(alpha=i, penalty='l1', class_weight='balanced', loss='log', random_state=28)
            clf.fit(X_train_final[selected_features], y_train)
            sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
            sig_clf.fit(X_train_final[selected_features], y_train)
            y_pred_prob = sig_clf.predict_proba(X_val_final[selected_features])[:, 1]
            #cv_auc_score.append(roc_auc_score(y_val, y_pred_prob))
            cv_auc_score[ str( i) ] = roc_auc_score(y_val, y_pred_prob)
            print('>>> MLService:logisticRegressionTrain >>>', 'For alpha {0}, cross validation AUC score {1}'.format(i, roc_auc_score(y_val, y_pred_prob)))

        lstScoring = list(cv_auc_score.values())
        maxIndex = np.argmax(lstScoring)
        trainModel = {
            "max": {
                "alpha": alpha[maxIndex],
                "score": lstScoring[maxIndex]
            },
            "list": cv_auc_score,
            "clasifier": SGDClassifier(alpha=alpha[maxIndex], penalty='l1', class_weight='balanced', loss='log', random_state=28)
        }
        print('>>> MLService:logisticRegressionTrain >>>', 'The Optimal C value is:', trainModel['max']['alpha'])
        return trainModel

    def logisticRegressionTest(self, cv_auc_score, X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, selected_features):
        alpha = np.logspace(-4, 4, 9)
        #best_alpha = alpha[np.argmax(cv_auc_score)]
        best_alpha = cv_auc_score
        logreg = SGDClassifier(alpha=best_alpha, class_weight='balanced', penalty='l1', loss='log', random_state=28)

        X_train_final = X_train_final.fillna(X_train_final.mean())
        X_val_final = X_val_final.fillna(X_val_final.mean())
        X_test_final = X_test_final.fillna(X_test_final.mean())

        # Check if there are missing values​in the data. The following method can be used.
        # Flase: No missing value in the feature value of the corresponding feature
        # True: There are missing values
        print('>>> MLService:logisticRegressionTest >>>', X_train_final.isnull().any())
        print('>>> MLService:logisticRegressionTest >>>', X_val_final.isnull().any())
        print('>>> MLService:logisticRegressionTest >>>', X_test_final.isnull().any())

        # Check if it contains infinite data
        # False: Contains
        # True: Not included
        print('>>> MLService:logisticRegressionTest >>>', np.isfinite(X_train_final).all())

        logreg.fit(X_train_final[selected_features], y_train)
        logreg_sig_clf = CalibratedClassifierCV(logreg, method='sigmoid')
        logreg_sig_clf.fit(X_train_final[selected_features], y_train)

        y_pred_prob_train = logreg_sig_clf.predict_proba(X_train_final[selected_features])[:, 1]
        print('>>> MLService:logisticRegressionTest >>>', 'For best alpha {0}, The Train AUC score is {1}'.format(best_alpha, roc_auc_score(y_train, y_pred_prob_train)))
        y_pred_prob_val = logreg_sig_clf.predict_proba(X_val_final[selected_features])[:, 1]
        print('>>> MLService:logisticRegressionTest >>>', 'For best alpha {0}, The Cross validated AUC score is {1}'.format(best_alpha, roc_auc_score(y_val, y_pred_prob_val)))
        y_pred_prob_test = logreg_sig_clf.predict_proba(X_test_final[selected_features])[:, 1]
        print('>>> MLService:logisticRegressionTest >>>', 'For best alpha {0}, The Test AUC score is {1}'.format(best_alpha, roc_auc_score(y_test, y_pred_prob_test)))
        
        y_pred = logreg.predict(X_test_final[selected_features])
        print('>>> MLService:logisticRegressionTest >>>', 'The test AUC score is :', roc_auc_score(y_test, y_pred_prob_test))
        print('>>> MLService:logisticRegressionTest >>>', 'The percentage of misclassified points {:05.2f}% :'.format((1 - accuracy_score(y_test, y_pred)) * 100))
        # IMG plot_confusion_matrix(y_test, y_pred)
        return roc_auc_score(y_train, y_pred_prob_train), roc_auc_score(y_val, y_pred_prob_val), roc_auc_score(y_test, y_pred_prob_test)
        
    def train(self, filename, algorithm="logisticRegression"):
        # Read Mining View
        data = self.readDataMiningView(filename)
        # Dividing Mining View
        self.dividingData(data)
        print('>>> MLService:train >>>', 'Begin models.... ... Balanced classes')
        # Read Data train, test, ... Balanced classes
        X_train_final, X_val_final, X_test_final, y_train, y_val, y_test = self.readDataTrainTest()
        # Selection of features ... Balanced classes
        selected_features = self.selectionFeatures(X_train_final, y_train)

        # Train Model
        model = self.logisticRegressionTrain(X_train_final, X_val_final, X_test_final, y_train, y_val, selected_features)
        # Test Model
        roc_auc_score = self.logisticRegressionTest(model['max']['alpha'], X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, selected_features)
        
        classif_name = 'data/classifier_' + algorithm + '_data_model' + '.pkl'
        self.etl.save_object(classif_name, model["clasifier"])
        del model["clasifier"]

        print('>>> MLService:train >>>', 'Models Done!!!!')
        return {
            "train": model,
            "test": {
                "train": roc_auc_score[0],
                "validation": roc_auc_score[1],
                "value": roc_auc_score[2]
            }
        } 
        

