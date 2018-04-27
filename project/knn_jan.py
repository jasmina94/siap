import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score

january_path = "C:/Users/Jasmina/Desktop/data/january.csv"
january_pca_path = "C:/Users/Jasmina/Desktop/data/january_pca.csv"
june_path = "C:/Users/Jasmina/Desktop/data/june.csv"
june_pca_path = "C:/Users/Jasmina/Desktop/data/june_pca.csv"


def read_data(month_to_read):
    data = None
    if month_to_read == "january":
        data = pd.read_csv(january_path)
    else:
        data = pd.read_csv(june_path)
    data.AIRLINE = data.AIRLINE.astype(float)
    return data

def knn_januar_full():
    jan_flights = read_data("january")
    jan_flights = jan_flights.dropna(axis=0, how='any')

    train, test = train_test_split(jan_flights, test_size=0.2, random_state=42)
    test, validation = train_test_split(test, test_size=0.5, random_state=42)

    test_features = np.array(test.ix[:, 0:26])
    test_labels = np.array(test.iloc[:, 26])

    train_features = np.array(train.ix[:, 0:26])
    train_labels = np.array(train.iloc[:, 26])

    validation_features = np.array(validation.ix[:, 0:26])
    validation_labels = np.array(validation.iloc[:, 26])

    # print(np.any(np.isnan(train_features)))
    # print(np.any(np.isfinite(train_features)))

    knn = KNeighborsClassifier(n_neighbors=3)

    train_features = StandardScaler().fit_transform(train_features)

    # fitting the model
    knn.fit(train_features, train_labels)

    # predict the response
    pred = knn.predict(test_features)

    # evaluate accuracy
    print(accuracy_score(test_labels, pred))

    myList = list(range(1, 50))

    # subsetting just the odd ones
    neighbors = filter(lambda x: x % 2 != 0, myList)

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train_features, train_labels, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print("The optimal number of neighbors is %d" % optimal_k)

    # plot misclassification error vs k
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()




if __name__ == "__main__":
    knn_januar_full()