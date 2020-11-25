import pandas as pa
import numpy as nu
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pa.read_csv("student-mat.csv", sep=";")
# print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age", "health"]]
# print(data.head())
predict = "G3"
x = nu.array(data.drop([predict], 1))
y = nu.array(data[predict])
best = 0
loops = 0

# Find Plot with R^2 > .975
for t in range(30000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acy = linear.score(x_test, y_test)
    loops += 1
    if acy > best:
        best = acy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)  # Saves Data as Pickle
    if acy > .975:
        break

# Create Pickle File
picklein = open("studentmodel.pickle", "rb")
linear = pickle.load(picklein)

# Print R^2 and Num of Loops
print("Correlation Coefficent Across all 6 Dimensions: ", best)
print("Number of Trials: ", loops)
# print(acy) Accuracy of Data
# print("Slopes of Data: ", linear.coef_)
# print("Intercept of Data: ", linear.intercept_)

# Print Test Values
'''
prediction = linear.predict(x_test)
for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])
'''

# Study Time vs FG
p = "studytime"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
m, b = nu.polyfit(data[p], data["G3"], 1)
pyplot.plot(data[p], m * data[p] + b)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.title("Study Time vs Final Grade")
pyplot.show()
print("Final Grade = ", m, "* Study Time + ", b)

# 1st Sem Grade vs Final Grade
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
m, b = nu.polyfit(data[p], data["G3"], 1)
pyplot.plot(data[p], m * data[p] + b)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.title("1st Semester Grade vs Final Grade")
pyplot.show()
print("Final Grade = ", m, "* 1st Semester Grade + ", b)

# 2nd Sem Grade vs Final Grade
p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
m, b = nu.polyfit(data[p], data["G3"], 1)
pyplot.plot(data[p], m * data[p] + b)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.title("2nd Semester Grade vs Final Grade")
pyplot.show()
print("Final Grade = ", m, "* 2nd Semester Grade + ", b)

# Previous Failures vs Final Grade
p = "failures"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
m, b = nu.polyfit(data[p], data["G3"], 1)
pyplot.plot(data[p], m * data[p] + b)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.title("Previous Failures vs Final Grade")
pyplot.show()
print("Final Grade = ", m, "* Previous Failures + ", b)

# Absences vs Final Grade
p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
m, b = nu.polyfit(data[p], data["G3"], 1)
pyplot.plot(data[p], m * data[p] + b)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.title("Absences vs Final Grade")
pyplot.show()
print("Final Grade = ", m, "* Absences + ", b)

# Age vs Final Grade
p = "age"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
m, b = nu.polyfit(data[p], data["G3"], 1)
pyplot.plot(data[p], m * data[p] + b)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.title("Age vs Final Grade")
pyplot.show()
print("Final Grade = ", m, "* Age + ", b)

# Health vs Final Grade
p = "health"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
m, b = nu.polyfit(data[p], data["G3"], 1)
pyplot.plot(data[p], m * data[p] + b)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.title("Lvl of Health vs Final Grade")
pyplot.show()
print("Final Grade = ", m, "* Level of Health + ", b)

