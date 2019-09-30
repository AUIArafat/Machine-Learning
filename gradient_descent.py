import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_scores.csv")
print(df.head())

X = df["Hours"]  # Read values of X from data frame
Y = df["Scores"]  # Read values of Y from data frame

plt.plot(X, Y, 'o')  # 'o' for creating scatter plot
plt.title("Implementing Gradient Descent")
plt.xlabel("Hours Studied")
plt.ylabel("Student Score")
m = 0
b = 0


def grad_desc(X, Y, m, b):
    for point in zip(X, Y):
        x = point[0]
        y_actual = point[1]
        y_prediction = m * x + b
        print(x, " ", y_actual, " ", y_prediction)
        error = y_prediction - y_actual
        delta_m = -1 * error * x * 0.0005
        delta_b = -1 * error * 0.0005
        m = m + delta_m
        b = b + delta_b

        print(m, b)
    return m, b


def plot_regression_line(X, m, b):
    regression_x = X.values
    regression_y = []

    for x in regression_x:
        y = m * x + b
        regression_y.append(y)
    plt.plot(regression_x, regression_y)
    plt.pause(1)


for i in range(0, 10):
    m, b = grad_desc(X, Y, m, b)
    plot_regression_line(X, m, b)
