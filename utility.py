import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os.path
from datetime import date
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import classification_report, accuracy_score
from category_encoders.target_encoder import TargetEncoder
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import optuna
from scipy import stats
from scipy.cluster import hierarchy as hc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier

# feature creation

def feature_creation_alter(df:pd.DataFrame):
    # log transformation
    
    df['Hillshade'] = df['Hillshade_3pm'] + df['Hillshade_9am'] + df['Hillshade_Noon']
    # df['Distance_To_Hydrology'] = abs(df['Vertical_Distance_To_Hydrology']) + abs(df['Horizontal_Distance_To_Hydrology'])
    df['Distance_Roadways_FirPoints'] = df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']
    # df['Distance_Roadways-FirPoints'] = df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Fire_Points']

    # df['binned_elev'] = [math.floor(v/50.0) for v in df['Elevation']]
    # df['sqrt_elev'] = np.sqrt(df['Elevation'])
    df['Ele_Hillshade'] = df['Elevation'] - df['Hillshade']
    # df['Ele-Hillshade'] = df['Elevation'] + df['Hillshade']

    # area
    df['Wilderness_Area1_soil'] = df['Wilderness_Area1'] + df['Soil_Type29']
    df['Wilderness_Area2_soil'] = df['Wilderness_Area2'] + df['Soil_Type40']
    df['Wilderness_Area3_soil'] = df['Wilderness_Area3'] + df['Soil_Type32']
    df['Wilderness_Area4_soil'] = df['Wilderness_Area4'] + df['Soil_Type3'] # 較不顯著

    # Elevation 
    df['Elevation_FirePoint'] = df['Elevation'] + df['Horizontal_Distance_To_Fire_Points']
    df['Elevation-FirePoint'] = df['Elevation'] - df['Horizontal_Distance_To_Fire_Points']

    df['Elevation_Road'] = df['Elevation'] + df['Horizontal_Distance_To_Roadways']
    df['Elevation-Road'] = df['Elevation'] - df['Horizontal_Distance_To_Roadways']

    # Slope...
    # df['binned_Slope'] = [math.floor(v/2.60) for v in df['Slope']]
    # df['sqrt_Slope'] = np.sqrt(df['Slope'])

    # Aspect...
    # df['binned_Aspect'] = [math.floor(v/18.0) for v in df['Aspect']]
    # df['sqrt_Aspect'] = np.sqrt(df['Aspect'])
    return df 


# train

def train(method:str, x1, x2, y1, y2, param=None):    
    # data, target = df.drop(['Cover_Type'], axis=1), df['Cover_Type']
    # x1, x2, y1, y2 = train_test_split(data, target, random_state=42)
    # y1 -= 1
    # y2 -= 1
    
    if method == 'xgb':
        eval_set = [(x1, y1), (x2, y2)]
        if param: xgb = XGBClassifier(**param)
        else: xgb = XGBClassifier(eval_metric=['merror', 'mlogloss'])
        xgb.fit(x1, y1, eval_set = eval_set, verbose=False, early_stopping_rounds=10)
        return xgb
    elif method == 'rf':
        if param:
            rf = RandomForestClassifier(**param)
        else:
            rf = RandomForestClassifier()
        rf.fit(x1, y1)
        return rf
    elif method == 'hgb':
        if param:
            hgb = HistGradientBoostingClassifier(**param)
        else:
            hgb = HistGradientBoostingClassifier()
        hgb.fit(x1, y1)
        return hgb
    elif method == 'ext':
        if param:
            ext = ExtraTreesClassifier(**param)
        else:
            ext = ExtraTreesClassifier()
        ext.fit(x1, y1)
        return ext
    elif method == 'LGBM':
        eval_set = [(x1, y1), (x2, y2)]
        if param: lgbm = LGBMClassifier(**param)
        else:lgbm = LGBMClassifier()
        lgbm.fit(x1, y1, eval_set = eval_set, verbose=False, early_stopping_rounds=10)
        return lgbm
    else:
        print('method error!')

def plot_error_xgb(xgb):
    results = xgb.evals_result()
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,7))
    plt.plot(results["validation_0"]["mlogloss"], label="Training loss")
    plt.plot(results["validation_1"]["mlogloss"], label="Validation loss")
    plt.xlabel("Number of trees")
    plt.ylabel("Loss")
    plt.legend()

def score(model, x1, x2, y1, y2):
    pred = model.predict(x1)
    print('train acc: ', accuracy_score(y1, pred))
    pred = model.predict(x2)
    print('test acc: ', accuracy_score(y2, pred))
    print('cross validation score: ',np.mean(cross_val_score(model, x2, y2, cv=5)))


def print_feature_importance(x1:pd.DataFrame, model):
    feature_importance = pd.DataFrame({'features':x1.columns, 'Importances':model.feature_importances_}).sort_values(by='Importances',ascending=True)
    for row in feature_importance.iterrows():
        feature, importance = row[1].to_numpy()
        w = 45-len(feature)
        print('{0:}: {1:>{width}.4}'.format(feature, importance, width=w))

