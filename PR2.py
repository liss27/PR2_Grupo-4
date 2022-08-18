import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
data = pd.read_csv('House_Rent_Dataset.csv')
data.head()
data.info()
data.describe()
data.describe(include = 'object')
data.isnull().sum()
# Separar las columnas de pisos en el piso en el que está y los pisos totales
data["Floor_Number"]=data["Floor"].apply(lambda x:str(x).split()[0])
data["Total_Number_of_Floor"]=data["Floor"].apply(lambda x:str(x).split()[-1])
data.head()
# Remover columnas no necesarias
data.drop(["Posted On", "Floor", "Area Locality"], axis="columns", inplace=True)
data.head()
m = data["Floor_Number"]=="Ground"
data[m]
def fun(c):
    return c.replace("Ground", "0")
clm=["Floor_Number"]
for c in clm:
    data[c]=data[c].apply(fun)
data["Floor_Number"].value_counts()
def fun2(v):
    return v.replace("Lower", "1")
for v in clm:
    data[v]=data[v].apply(fun2)
def fun3(b):
    return b
data["Floor_Number"]=data["Floor_Number"].apply(fun3)
data["Floor_Number"]=pd.to_numeric(data["Floor_Number"], errors='coerce')
data["Total_Number_of_Floor"].value_counts()
clm1=["Total_Number_of_Floor"]
for c in clm1:
    data[c]=data[c].apply(fun)
data["Total_Number_of_Floor"]=data["Total_Number_of_Floor"].astype(int)
data['Floor_Number'].fillna(0, inplace = True)
data["Floor_Number"]=data["Floor_Number"].astype(int)
data = pd.get_dummies(data, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
data.head()
data.corr()['Rent'].sort_values(ascending=False)
X = data.drop(columns=['Rent'], axis=1)
y = data['Rent']
X, y = X.astype('int64'),y.astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
y_train = np.array(y_train).reshape(y_train.shape[0], 1)
mms_X = MinMaxScaler()
mms_y = MinMaxScaler()
X_train = mms_X.fit_transform(X_train)
y_train = mms_y.fit_transform(y_train)
model = RandomForestRegressor(n_estimators=700, max_depth=70, min_samples_leaf=3, min_samples_split= 5,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(mms_X.transform(X_test.values))
y_pred = mms_y.inverse_transform(y_pred.reshape(len(y_test), 1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.values.reshape(len(y_test),1)),1))
r2_score(y_pred, y_test)
def fine_tune(model, param_grid):
    grid_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, random_state=42)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, np.sqrt((-1)*grid_search.best_score_)
param_grid = [{'n_estimators': [300, 500, 700], 'max_depth': [40, 60, 80], 'min_samples_leaf': [5, 7, 9], 'min_samples_split': [13, 15, 17]}]
fine_tune(model, param_grid)
plt.figure(figsize=(8, 24))
plt.scatter(x=range(len(y_test)), y=y_test)
plt.scatter(x=range(len(y_test)), y=y_pred)
plt.ylim([100, 1250000])
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()
final_model = RandomForestRegressor(n_estimators=500,
                                    min_samples_split=13,
                                    min_samples_leaf=7,
                                    max_depth=40, random_state=42)
final_model.fit(X_train, y_train)
y_pred_1 = final_model.predict(mms_X.transform(X_test.values))
y_pred_2 = mms_y.inverse_transform(y_pred_1.reshape(len(y_test), 1))
print(np.concatenate((y_pred.reshape(len(y_pred_2),1), y_test.values.reshape(len(y_test),1)),1))
r2_score(y_pred_2, y_test)