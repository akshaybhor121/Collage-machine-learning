import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "EducationScore": [60, 62, 65, 68, 70, 73, 76, 80, 83, 85],
    "Salary": [25000, 28000, 32000, 35000, 39000,
               43000, 47000, 50000, 55000, 60000]
}
df = pd.DataFrame(data)
print(df)

X = df[["Experience", "Age", "EducationScore"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual Salary:", y_test.values)
print("Predicted Salary:", y_pred)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

new_person = [[6, 27, 85]]
prediction = model.predict(new_person)
print("Predicated Salary:",prediction[0])