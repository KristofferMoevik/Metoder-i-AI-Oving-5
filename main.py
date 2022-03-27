import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn import tree

with open(file="scikit-learn-data.pickle", mode="rb") as file: 
    data = pickle.load(file)

vectorizer = HashingVectorizer()
x_train_hashed = vectorizer.fit_transform(data["x_train"])
x_test_hashed = vectorizer.fit_transform(data["x_test"])

classifier = BernoulliNB()
classifier.fit(X=x_train_hashed, y=data["y_train"])
prediction = classifier.predict(x_test_hashed)
print("Accuracy Naive bayes",accuracy_score(data["y_test"], prediction))

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X=x_train_hashed, y=data["y_train"])
prediction = classifier.predict(x_test_hashed)
print("Accuracy Decision tree",accuracy_score(data["y_test"], prediction))

# Accuracy Naive bayes 0.7987021941652366
# Accuracy Naive bayes 0.8597235842118166