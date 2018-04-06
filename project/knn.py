import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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


if __name__ == "__main__":
    print("Starting k-nn...")
    k = 1
    x, y = read_data()
    prediction, acc = knn_basic(k, x, y)
    print("Accuracy for k={0} is {1:.2f}".format(k, acc))

    print("k-nn finished!")