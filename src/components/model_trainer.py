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

            params={
                'Linear Regression':{},
                'K-Neighbour':{
                    'n_neighbours':[5,7,9,11],
                    # 'algorithm':['ball_tree,kd_tree','brute']
                },
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_depth':[]
                    'max_features':['sqrt','log2'],

                },
                'Random Froest':{
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_features':['sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Gradient Boost':{
                    # 'loss':['squared_error','absolute_error','quantile'],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],                    
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_features':['sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'AdaBoost':{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]

                },
                'XGBoost':{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'CatBoost':{
                    'learning_rate':[0.01,0.05,0.001],
                    'depth':[6,8,10],
                    'iterations':[30,50,100]

                }

            }
            model_report:dict=evaluate_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models,parameter=params)

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