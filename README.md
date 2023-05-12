# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:R.K Pragalyaa shree
RegisterNumber:212221040125
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
print("Placement data")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print("Salary data")
data1.head()

print("Checking the null() function")
data1.isnull().sum()

print("Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print("Print data")
data1

x=data1.iloc[:,:-1]
print("Data-status")
x

y=data1["status"]
print("data-status")
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)

print("Prediction of LR")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![ml1](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/ffbdc8c4-5ecb-47aa-a8e2-195cbd5bd7fa)
![ml2](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/f411e96a-4edf-4210-9665-bd4568aa1e84)
![ml3](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/74ad70bc-876a-4b3d-87ca-33b3e72a6f88)
![ml4](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/a37756e1-7dd5-442a-915c-a432dbb7f7f5)
![ml5](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/6db64083-4e11-48c7-bca6-7857826e7058)
![ml6](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/cb8c9858-fd59-46b6-a6fe-bc92693fff06)
![ml7](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/c6bf7bb6-d644-416d-9a16-c632886b9632)
![ml8](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/c3deaa2c-9b98-427e-86af-a85ab1a4346e)
![ml9](https://github.com/pragalyaashree/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135934/85216363-ac60-4ac5-8e4b-d79a369a4c95)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
