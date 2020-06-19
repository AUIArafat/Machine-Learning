from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
st.sidebar.header('User Input Parameters')


def load_data():
	data = datasets.load_iris()
	return data


def user_input_features():
	sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
	sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
	petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
	petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
	data = {'sepal_length': sepal_length,
			'sepal_width': sepal_width,
			'petal_length': petal_length,
			'petal_width': petal_width}
	features = pd.DataFrame(data, index=[0])
	return features


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

data_load_state = st.text('Loading data...')
iris = load_data()
data_load_state.text('Loading data...done!')

print(type(iris.data))
iris_data_pd = pd.DataFrame({"classes":iris.target, "Sepal length":iris.data[:,0], "Sepal width":iris.data[:,1],
							 "Petal length":iris.data[:,2], "Petal width":iris.data[:,3]})

if st.checkbox('Show raw data'):
	st.subheader('Iris Raw Dataset')
	st.write(iris_data_pd)

if st.checkbox('Show bar chart of Iris Dataset'):
	st.subheader('Bar chart of Iris Dataset')
	st.bar_chart(iris.data)

x = iris.data
y = iris.target
print(x.shape)

classifier = neighbors.KNeighborsClassifier()  # using KNeighborsClassifier()

# Train the Model.
classifier.fit(x, y)

# Make predictions:
predictions = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

print("prediction : ", predictions)
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[predictions])
print(predictions)

st.subheader('Prediction Probability')
st.write(prediction_proba)
