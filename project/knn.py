from tkinter import Image
import graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

january_path = "C:/Users/Jasmina/Desktop/data/january.csv"

january_pca_path = "C:/Users/Jasmina/Desktop/data/january_pca.csv"

def read_data():
    column_names = ["DAY", "DAY_OF_WEEK", "TAXI_OUT",
             "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN",
             "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY", "AIR_SYSTEM_DELAY",
             "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "IsLate"]
    df = pd.read_csv('data/january3-nostringas.csv', header=0, names=column_names)

    # create design matrix X and target vector y
    x = np.array(df.ix[:, 0:22])  # end index is exclusive
    y = np.array(df['IsLate'])
    return x, y


def knn_basic(k, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    knn = KNeighborsClassifier(n_neighbors=k)   # classifier
    knn.fit(X_train, y_train)  # fitting the model
    prediction = knn.predict(X_test)  # predict the response
    accuracy = accuracy_score(y_test, prediction)  # evaluate accuracy
    return prediction, accuracy


def knn():
    flights = pd.read_csv(january_pca_path)
    flights = flights.fillna(0)

    x = np.array(flights.ix[:, 0:21])
    y = np.array(flights['IS_LATE'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)

    #train, test = train_test_split(flights, test_size=0.2, train_size=0.8)
    #test, validation = train_test_split(test, test_size=0.5)

    # print(train.shape)
    # print(test.shape)
    # print(validation.shape)

    #target = np.array(train['IS_LATE'])
    #predicted_target = test['IS_LATE']

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')  # classifier

    fitness = knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    confusion_mat = confusion_matrix(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)

    print(accuracy)
    print(f1)
    print(precision)
    print(recall)
    print(confusion_mat)


def svm_fun():
    flights = pd.read_csv(january_pca_path)
    flights = flights.fillna(0)

    x = np.array(flights.ix[:, 0:21])
    y = np.array(flights['IS_LATE'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    model = LinearSVC(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    confusion_mat = confusion_matrix(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)

    print(accuracy)
    print(f1)
    print(precision)
    print(recall)
    print(confusion_mat)


def decision_tree_fun():
    flights = pd.read_csv(january_pca_path)
    flights = flights.fillna(0)

    x = np.array(flights.ix[:, 0:21])
    y = np.array(flights['IS_LATE'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    confusion_mat = confusion_matrix(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)

    print(accuracy)
    print(f1)
    print(precision)
    print(recall)
    print(confusion_mat)

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())


    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")

    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                                 feature_names=['DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'ARRIVAL_TIME', 'ORIGIN_AIRPORT_LAT', 'ORIGIN_AIRPORT_LON', 'DESTINATION_AIRPORT_LAT', 'DESTINATION_AIRPORT_LON','AIRLINE_AS', 'AIRLINE_B6', 'AIRLINE_DL', 'AIRLINE_EV', 'AIRLINE_F9','AIRLINE_HA', 'AIRLINE_MQ', 'AIRLINE_NK', 'AIRLINE_OO', 'AIRLINE_UA', 'AIRLINE_US', 'AIRLINE_VX', 'AIRLINE_WN'],
    #                                 class_names=['IS_LATE'],
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)
    #graph = graphviz.Source(dot_data)
    #graph



if __name__ == "__main__":
    # print("Starting k-nn...")
    # k = 1
    # x, y = read_data()
    # prediction, acc = knn_basic(k, x, y)
    # print("Accuracy for k={0} is {1:.2f}".format(k, acc))


    #knn()
    #svm_fun()
    decision_tree_fun()

    print("k-nn finished!")