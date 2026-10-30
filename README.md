# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program and Output:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LAKSHA M
RegisterNumber: 212224220050
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('drive/MyDrive/Dataset-ML/student_scores.csv')
df.head()

```

<img width="280" height="265" alt="417874391-ce69963c-9e81-4531-9936-8ed0667213ad" src="https://github.com/user-attachments/assets/447dd41f-a7f4-451c-bdd3-98a21487310f" />

```
df.tail()
```

<img width="296" height="268" alt="417874436-e8ef2d36-7987-4704-b7fc-3c532d29f7be" src="https://github.com/user-attachments/assets/87734cc6-1566-48b1-b92d-09ef01d5f4c7" />

```
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
X,Y
```
<img width="782" height="599" alt="417874465-9ed3f8f2-f940-4706-8545-a1e8fd9ef941" src="https://github.com/user-attachments/assets/13126847-19b7-4db3-9c06-c9cf1f779343" />

```

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
Y_pred,Y_test
```

<img width="766" height="79" alt="417874489-41a501c9-f572-4f83-9666-d9c481dd37f0" src="https://github.com/user-attachments/assets/d71ef70d-4088-4761-88d7-26da34d64954" />

```
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Hours VS Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```

<img width="785" height="573" alt="417874517-5c9a63c0-1115-4a56-9598-d4c616ad40a7" src="https://github.com/user-attachments/assets/55ed5593-0572-4e20-975a-92f607803e40" />

```
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title('Hours VS Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```

<img width="791" height="578" alt="417874561-decc794b-aa1f-48a4-9370-593ccf6721d8" src="https://github.com/user-attachments/assets/484c86cc-29ad-4ef5-8195-a57ec9978165" />

```
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print("MSE = ",mse)
print('MAE = ',mae)
print('RMSE = ',rmse)
```

<img width="270" height="79" alt="417874597-c82833da-b6f2-43c4-9916-68c248d3baf6" src="https://github.com/user-attachments/assets/b0b54a48-471d-400f-95eb-1891fafef561" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
