import sys
import os
import numpy as np
import pandas as pd

from  dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_objects


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
 
@dataclass
class DataTransformationConfig: 
    preprocessing_obj_file_path=os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformer_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''this function is responsible for data transformation'''
        try:
            num_feature=['reading score','writing score']
            cat_feature=['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler())
                ]
            )
            logging.info("numerical feature encoding completed")
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))   #optional
                ]
            )
            logging.info("categorical feature encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num pipeline",num_pipeline,num_feature),
                    ("cat pipeline",cat_pipeline,cat_feature)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading train and test data is completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_col='math score'
            num_feature=['reading score','writing score']

            input_feature_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]

            logging.info("appplying preprocessing object on train and test dataframe")

            preprocessed_input_feature_train=preprocessing_obj.fit_transform(input_feature_train_df)
            preprocessed_input_feature_test=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[preprocessed_input_feature_train,np.array(target_feature_train_df)]
            test_arr=np.c_[preprocessed_input_feature_test,np.array(target_feature_test_df)]
            
            logging.info("saved preprocessing object")

            save_objects(
                file_path=self.data_transformer_config.preprocessing_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessing_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)