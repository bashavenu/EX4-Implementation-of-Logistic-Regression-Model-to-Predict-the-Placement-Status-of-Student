# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect and clean the dataset (handle missing values, encode categorical variables, and scale features).

2.Split the dataset into training and testing sets (e.g., 70% training, 30% testing).

3.Train a logistic regression model on the training data by fitting it to the features and target (placement status).

4.Use the trained model to predict on the test set and evaluate using accuracy, confusion matrix, and other metrics.

5.Adjust model parameters using cross-validation to optimize performance.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BASHA VENU
RegisterNumber:  2305001005

import pandas as pd
import numpy as np
d=pd.read_csv("/content/ex45Placement_Data (1).csv")
d.head()
data1=d.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
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
data1
x=data1.iloc[:, :-1]
x
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print("Acuuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification report:\n",classification)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```
## Output:

![image](https://github.com/user-attachments/assets/f2079f1d-9872-4084-ab7c-1a31b7ab2bb0)
![image](https://github.com/user-attachments/assets/8822cdd7-ce50-4df5-abe3-bc96e0e193e8)
![image](https://github.com/user-attachments/assets/793404a8-6452-4441-ae80-4313dfd451bb)
![image](https://github.com/user-attachments/assets/dfd329a6-50ff-4076-a4a1-f847ea7e36ad)
![image](https://github.com/user-attachments/assets/769e50f7-4757-4829-832d-0d7f53989e7d)
![image](https://github.com/user-attachments/assets/07186c86-36fb-47fe-b398-6c8d0e310481)
![image](https://github.com/user-attachments/assets/d883860e-c42d-4689-ab98-9b5cce5d03c6)
![image](https://github.com/user-attachments/assets/e0cf1415-cd3f-4402-83c9-88fd7ec7be44)
![image](https://github.com/user-attachments/assets/0f87d62d-c58d-41f4-ad9c-fe249d5bd671)
![image](https://github.com/user-attachments/assets/af4d23a6-a57d-4144-b640-ad4174332c9f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
