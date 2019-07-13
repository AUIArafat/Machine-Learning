from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# x contains the features and y contains the labels
x = iris.data
y = iris.target

print(x.shape)
#***Splitting the dataset***#
# x_train contains the training features
# x_test contains the testing features
# y_train contains the training label
# y_test contains the testing labels
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.5)
print(x_train.shape)

# Build The Model
# Useful Link :
# https://hackernoon.com/a-brief-look-at-sklearn-tree-decisiontreeclassifier-c2ee262eab9a
# https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a

# classifier = tree.DecisionTreeClassifier() # using DecisionTreeClassifier()

classifier = neighbors.KNeighborsClassifier() # using KNeighborsClassifier()

# Train the Model.
classifier.fit(x_train,y_train)

# Make predictions:
predictions = classifier.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))

# accuracy score using DecisionTreeClassifier : 0.9466666666666667
# accuracy score using KNeighborsClassifier : 0.9866666666666667