import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from catboost import  CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_objects,evaluate_model

@dataclass
class ModelTrainConfig:
    trained_model_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainConfig() 

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and testing data")
            x_train,x_test,y_train,y_test=(
                train_array[ : , :-1],
                test_array[ : , :-1],
                
                train_array[ : ,-1],
                test_array[ : ,-1]
            )

            models={
                "Linear Regression":LinearRegression(),
                "K-Neighbour":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
                "Gradient Boost":GradientBoostingRegressor(),
                "XGBoost":XGBRegressor() 
            }
            model_report:dict=evaluate_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("no best model found")
            
            logging.info("best model found on both training and testing dataset")
            
            save_objects(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model

            )

            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)       