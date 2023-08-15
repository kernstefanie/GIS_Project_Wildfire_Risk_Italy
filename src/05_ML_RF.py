# -*- coding: utf-8 -*-
"""
Machine Learning with Random Forest

Created on Mon Jul 17 14:24:42 2023

@author: Stefanie Kern & Iris Haake

"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rioxr
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


####################################### ML-Model #################################################

# First, a model for predicting fire areas is created using the training data. 
# The Random Forest method is used to train the model using scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
# Further processing steps to create a model are:
# - Prepare Training Data
# - Generate RF Model
# - Model validation with the independent test dataset

################################ Prepare Training Data ########################################
# Before training of the RF model can begin, the training data (2000-2019) must be divided into `target` and `predictor variables` ("features"), and into `training` and `validation` data.

    # Read training data
    file = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire_nofire/fire_nofire_trainingdata.csv'
    data = pd.read_csv(file,delimiter=',',parse_dates=['time'], index_col="time")
    data
    
    # Separate data into features (independent) and target (dependent)
    features = data.drop('fire status', axis = 1) # independent variable
    target = data['fire status'] # dependent variable 
    
    target
    features
    
    # Create random training and validation datasets of features and target variable
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    X_train
    y_train
    y_train.hist()
    X_test
    y_test
    y_test.hist()
    
    ## Random Forest Model
        # In order to make later predictions of fire/non-fire occurrence in July 2022, a model based on training data (2001-2019) must first be created. In this case, the Random Forest (https://scikit-learn.org/stable/modules/ensemble.html#random-forests) algorithm is used. 
        # This is based on the formation of many decision trees, which use the input data to calculate whether there is a `fire (1)` or `no-fire (0)`. 
        # Since the mean value is drawn from all decision trees, Random Forest, in contrast to the "Decision Tree" method, does not tend very strongly to `overfitting` (= i.e., it can often reproduce the training data but fails when applied to independent test data). [Scikit-Learn 2023](https://scikit-learn.org/stable/modules/ensemble.html#forest)
    
        # In addition, 
        # - the `feature importance`
        # - `forward feature selection` 
        # are performed to improve the model accuracy. 
    
        # Import the model we are using
        from sklearn.ensemble import RandomForestClassifier
        
        # Instantiate the model
        rf_model = RandomForestClassifier(random_state=42, n_jobs= -1) # njobs = number of processors (in this case all for max. of computing power)
        
        # Train the model on training data
        rf_model.fit(X_train, y_train)
        rf_model.score(X_test, y_test)
        
        # Write trained RF-model
        import joblib
        joblib.dump(rf_model, 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev4/model/rf_model.model')
        
########################################## Feature importance ##################################################################################
# The most common method for calculating feature importances in sklearn models (such as Random Forest) is the mean decrease in impurity method. 
# This method measures how much reduction in the impurity criterion was achieved as a result of all the splits the trees made based on that feature.
# Feature importances are provided by the fitted attribute `feature_importances_` and they are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.

    # Calculate feature importance
    rf_model.feature_importances_
    
    # Plot the importances
    feat_imps = pd.DataFrame({"importances" : rf_model.feature_importances_, "features": list(features.columns)})
    feat_imps = feat_imps.sort_values(by="importances")
    feat_imps

    #

    # Here you get a first impression of which variables are a decisive factor in predicting fire. 
    # It becomes clear that parameters from all areas, be it `climate` (sp, pev, VPD, sshf, t2m, ...), `water reserves` (src, rel_hum), `greening` (NDVI), `terrain` (DEM and Slope) or `geographical location` (x,y or proximity to urban areas), have a major influence on the outcome of the calculation.
    

    # Forward feature selection:
        # Another approach to reduce the number of predictors is to use a `forward feature selection`. This can be realized with sklearn's [SequentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn-feature-selection-sequentialfeatureselector). 
        # Here the number of features have to be specified to which the predictor dataset should be reduced. The parameter `n_features_to_select` was set to 19 because the Recursive Feature Selection shows a second maximum (~84 %) with 19 variables in addition to the absolute maximum of the first variable. 

        from sklearn.feature_selection import SequentialFeatureSelector
        seqFeatSel = SequentialFeatureSelector(rf_model, n_features_to_select=19, direction='forward', scoring="accuracy", cv=3, n_jobs=-1)
        
        seqFeatSel.fit(X_train, y_train)
        
        # After fitting the selector to our training data, a list of features that were iteratively added to the model appears. 
        selected_features = X_train.columns[seqFeatSel.support_]
        selected_features

        # So let's go ahead and train/test the RF model again using only the selected predictors:
        rf_model_selFeat_fw = RandomForestClassifier(random_state=42, n_jobs= -1) # njobs = number of processors (with -1 max. computing power)
        rf_model_selFeat_fw.fit(X_train[selected_features], y_train)
        print(rf_model_selFeat_fw.score(X_test[selected_features],y_test))


        # Write trained model 
        import joblib
        joblib.dump(rf_model_selFeat_fw, 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev4/model/rf_final.model')

    
#################################### Model validation with independent test dataset (2018-2022) ###################################################
# In order to validate the RF model, a `baseline model` must first be created. 
# Then the following [evaluation metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) can be used to test the performance of the model: `Confusion Matrix`, `Accuracy`, `Precision`, `Recall or Sensitivity` and `F1 Score`. 

    # Load RF-model
    import joblib
    rf_model_final = joblib.load('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev4/model/rf_final.model')

    # Establish baseline model
        # Before predictions can be made and evaluated, a baseline model must be established, a sensible measure to be outperformed by the RF model. 
        # If the RF model cannot improve upon the baseline, then it will be a failure and a different model should be tried or admit that machine learning is not right for this problem.
        # The baseline prediction for this case will be [sklearn.dummy.DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).
        # The `"stratified"` strategy is used which generates random predictions by respecting the training set class distribution.

        # Import the model we are using
        from sklearn.dummy import DummyClassifier # scikit-learn summarize a lot of machine learning algorithms (classification, regression, ...)
        
        # Instantiate the model
        dummy_clf = DummyClassifier(strategy="stratified") # Dummy to compare with RF model
        
        # Train the model on training data
        dummy_clf.fit(features[selected_features], target)

        # Check model accuracy
        dummy_clf.score(X_test, y_test)
        
        # Read independent test data
        import pandas as pd
        filetestdata = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire_nofire/fire_nofire_testdata.csv'
        testdata = pd.read_csv(filetestdata,delimiter=',',parse_dates=['time'], index_col="time")
        testdata
        
        test_features = testdata.drop(['fire status'], axis = 1)
        test_target = testdata['fire status']

        test_features[selected_features]

    # Confusion Matrix:
        # For classification problems, the confusion or contingency matrix is a good starting point.
        # It can be used with two or more classes and is the basis for calculating the before mentioned metrics. 
        # The rows represent the predicted classes and the columns represent the actual classes in the validation or test data set.

        # RF-Model
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_estimator(rf_model_final, test_features[selected_features],test_target,cmap="Reds")

        # Baseline Model
        ConfusionMatrixDisplay.from_estimator(dummy_clf, test_features[selected_features],test_target,cmap="Reds")
        
    # Evaluation scores:
        
        # Make predictions
        y_pred_rf_model = rf_model_final.predict(test_features[selected_features])
        y_pred_baseline = dummy_clf.predict(test_features[selected_features])
        
        # Accuracy
        from sklearn.metrics import accuracy_score
        print(accuracy_score(test_target, y_pred_rf_model))
        print(accuracy_score(test_target, y_pred_baseline))
        
        # Precision
        from sklearn.metrics import precision_score
        print(precision_score(test_target, y_pred_rf_model))
        print(precision_score(test_target, y_pred_baseline))
        
        # Recall or Sensitivity
        from sklearn.metrics import recall_score
        print(recall_score(test_target, y_pred_rf_model))
        print(recall_score(test_target, y_pred_baseline))
        
        # F1
        from sklearn.metrics import f1_score
        print(f1_score(test_target, y_pred_rf_model))
        print(f1_score(test_target, y_pred_baseline))
        
        # Classification report
        from sklearn.metrics import classification_report
        print(classification_report(test_target, y_pred_rf_model))
        print(classification_report(test_target, y_pred_baseline))
