#importing all the library or package
import pandas as pd # useful for loading dataset
import numpy as np # to perform array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Loading the dataset
dataset = pd.read_csv(r'./marks.csv')

#Slicing the data into two parts
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#storing data in different variables
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

#Changing the value into smaller one to make our app fast 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)

#Training our Model
clf = LogisticRegression(random_state=0)
clf.fit(X_train,y_train)

#  Predicting result from model
mid = float(input("Enter the marks in Mid-Semester : " ))
end = float(input("Enter the marks in End-Semester : " ))
newinp = [[mid,end]]
res = clf.predict(sc.transform(newinp))
if res==1: 
    print("Student will Pass")
else :
    print("Student will Fail")


# Predicting the Accuracy opf our Model
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test,y_pred)
# print(cm)
print("The accuracy of the model is ",format(accuracy_score(y_test,y_pred) * 100))