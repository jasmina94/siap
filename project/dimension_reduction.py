import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


january_path = "C:/Users/Jasmina/Desktop/data/january.csv"

january_pca_path = "C:/Users/Jasmina/Desktop/data/january_pca.csv"

flights_path = "C:/Users/Jasmina/Desktop/data/flights.csv"

airlines_path = "C:/Users/Jasmina/Desktop/data/airlines.csv"

airports_path = "C:/Users/Jasmina/Desktop/data/airports.csv"


def make_airline_dict():
    airlines_map = {}
    airlines_map['UA'] = 1
    airlines_map['AA'] = 2
    airlines_map['US'] = 3
    airlines_map['F9'] = 4
    airlines_map['B6'] = 5
    airlines_map['OO'] = 6
    airlines_map['AS'] = 7
    airlines_map['NK'] = 8
    airlines_map['WN'] = 9
    airlines_map['DL'] = 10
    airlines_map['EV'] = 11
    airlines_map['HA'] = 12
    airlines_map['MQ'] = 13
    airlines_map['VX'] = 14

    return airlines_map


def prepare_data():
    flights = pd.read_csv(flights_path)
    airports = pd.read_csv(airports_path)
    airlines = pd.read_csv(airlines_path)

    # Uzimamo samo januar i jun
    january_flights = flights[flights['MONTH'] == 1]
    june_flights = flights[flights['MONTH'] == 6]

    # Ne gledamo letove koji su preusmereni ili otkazani
    january_flights = january_flights[january_flights['CANCELLED'] != 1]
    june_flights = june_flights[june_flights['DIVERTED'] != 1]

    # Menjamo airline -> koristimo samo get dummies ostace slova avio kompanije sto je lepse
    #airlines = make_airline_dict()
    #january_flights = january_flights.replace({"AIRLINE": airlines})
    #june_flights = june_flights.replace({"AIRLINE": airlines})

    # Popunjavamo nedostajujuce vrednosti
    january_flights = january_flights.fillna(0)
    june_flights = june_flights.fillna(0)

    # Odbacujemo kodove aerodroma i koristimo geo podatke iz airports.csv
    january_flights['ORIGIN_AIRPORT_LAT'] = january_flights['ORIGIN_AIRPORT'].map(
        airports.set_index('IATA_CODE')['LATITUDE'])
    january_flights['ORIGIN_AIRPORT_LON'] = january_flights['ORIGIN_AIRPORT'].map(
        airports.set_index('IATA_CODE')['LONGITUDE'])
    january_flights['DESTINATION_AIRPORT_LAT'] = january_flights['DESTINATION_AIRPORT'].map(
        airports.set_index('IATA_CODE')['LATITUDE'])
    january_flights['DESTINATION_AIRPORT_LON'] = january_flights['DESTINATION_AIRPORT'].map(
        airports.set_index('IATA_CODE')['LONGITUDE'])

    june_flights['ORIGIN_AIRPORT_LAT'] = june_flights['ORIGIN_AIRPORT'].map(airports.set_index('IATA_CODE')['LATITUDE'])
    june_flights['ORIGIN_AIRPORT_LON'] = june_flights['ORIGIN_AIRPORT'].map(
        airports.set_index('IATA_CODE')['LONGITUDE'])
    june_flights['DESTINATION_AIRPORT_LAT'] = june_flights['DESTINATION_AIRPORT'].map(
        airports.set_index('IATA_CODE')['LATITUDE'])
    june_flights['DESTINATION_AIRPORT_LON'] = june_flights['DESTINATION_AIRPORT'].map(
        airports.set_index('IATA_CODE')['LONGITUDE'])

    # Odbacujemo suvisne kolone
    january_flights = january_flights.drop(columns=['YEAR', 'MONTH', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                                                    'CANCELLATION_REASON', 'CANCELLED', 'DIVERTED', 'ORIGIN_AIRPORT',
                                                    'DESTINATION_AIRPORT'])
    june_flights = june_flights.drop(columns=['YEAR', 'MONTH', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                                              'CANCELLATION_REASON', 'CANCELLED', 'DIVERTED', 'ORIGIN_AIRPORT',
                                              'DESTINATION_AIRPORT'])


    #Categorical to numerical -> da pretvori airlines u numericke
    january_flights = pd.get_dummies(january_flights, columns=['AIRLINE'], prefix=['AIRLINE'])
    june_flights = pd.get_dummies(june_flights, columns=['AIRLINE'], prefix=['AIRLINE'])

    # Oznacavanje podataka isLate
    january_flights.loc[january_flights['ARRIVAL_DELAY'] <= 15, 'IS_LATE'] = 0
    january_flights.loc[january_flights['ARRIVAL_DELAY'] > 15, 'IS_LATE'] = 1
    june_flights.loc[june_flights['ARRIVAL_DELAY'] <= 15, 'IS_LATE'] = 0
    june_flights.loc[june_flights['ARRIVAL_DELAY'] > 15, 'IS_LATE'] = 1

    january_flights.to_csv("data/january.csv")
    june_flights.to_csv("data/june.csv")


def pca():
    flights = pd.read_csv(january_path)
    flights = flights.fillna(0)

    X = flights.iloc[:, 1:].values
    y = flights.iloc[:, 0].values

    ss = StandardScaler()
    X_std = ss.fit_transform(X)

    cov_mat = np.cov(X_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    tot = sum(eigen_vals)

    # var_exp ratio is fraction of eigen_val to total sum
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

    # calculate the cumulative sum of explained variances
    cum_var_exp = np.cumsum(var_exp)

    plot_pca_graph(var_exp, cum_var_exp)

    # PCA primena
    pca_2 = PCA(n_components=2)  # PCA with 3 primary components 80% tacnosti
    pca_3 = PCA(n_components=3)  # PCA with 4 primary components oko 85% tacnosti

    # fit and transform both PCA models
    X_pca_2 = pca_2.fit_transform(X_std)
    X_pca_3 = pca_3.fit_transform(X_std)

    plot_pca_2d_graph(X_pca_2)
    plot_pca_3d_graph(X_pca_3, flights)

    return X_pca_2, X_pca_3


def plot_pca_graph(var_exp, cum_var_exp):
    plt.bar(range(1, 40), var_exp, alpha=0.75, align='center', label='pojedinacna varijansa komponenti')
    plt.step(range(1, 40), cum_var_exp, where='mid', label='kumulativna varijansa komponenti')
    plt.ylim(0, 1.1)
    plt.xlabel('Glavne komponente')
    plt.ylabel('Odnos varijanse')
    plt.legend(loc='best')
    plt.show()


def plot_pca_2d_graph(X):
    # 2d grafik
    plt.scatter(X.T[0], X.T[1], alpha=0.75, c='blue')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


def plot_pca_3d_graph(X, flights):
    # 3d grafik
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # assign x,y,z coordinates from PC1, PC2 & PC3
    xs = X.T[0]
    ys = X.T[1]
    zs = X.T[2]

    # initialize scatter plot and label axes
    ax.scatter(xs, ys, zs, alpha=0.75, c='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    plot = ax.scatter(xs, ys, zs, alpha=0.75,
                      c=flights['ARRIVAL_DELAY'], cmap='viridis', depthshade=True)

    fig.colorbar(plot, shrink=0.6, aspect=9)
    plt.show()


def kmeans():
    flights = pd.read_csv(january_pca_path)
    flights = flights.fillna(0)

    X = flights.iloc[:, 1:].values
    X_std = StandardScaler().fit_transform(X)
    X_pca2, X_pca3 = pca()

    # distortions = []  # sum of squared error within the each cluster
    # for i in range(1, 11):
    #     km = KMeans(n_clusters=i,
    #                 init='k-means++',
    #                 n_init=10,
    #                 max_iter=300,
    #                 random_state=0)
    #     km.fit(X_std)
    #     distortions.append(km.inertia_)
    #
    # plt.plot(range(1, 11), distortions, marker='o', alpha=0.75)
    # plt.xlabel('Broj klastera')
    # plt.ylabel('Distorzija')
    # plt.show()

    km = KMeans(n_clusters=8,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)

    y_km = km.fit_predict(X_pca3)

    plt.scatter(X_pca2[y_km == 0, 0],
                X_pca2[y_km == 0, 1],
                c='lightgreen',
                label='Cluster 1')
    plt.scatter(X_pca2[y_km == 1, 0],
                X_pca2[y_km == 1, 1],
                c='orange',
                label='Cluster 2')
    plt.scatter(X_pca2[y_km == 2, 0],
                X_pca2[y_km == 2, 1],
                c='lightblue',
                label='Cluster 3')
    plt.scatter(X_pca2[y_km == 3, 0],
                X_pca2[y_km == 3, 1],
                c='red',
                label='Cluster 4')
    plt.scatter(X_pca2[y_km == 4, 0],
                X_pca2[y_km == 4, 1],
                c='pink',
                label='Cluster 5')
    plt.scatter(X_pca2[y_km == 5, 0],
                X_pca2[y_km == 5, 1],
                c='yellow',
                label='Cluster 6')
    plt.scatter(X_pca2[y_km == 6, 0],
                X_pca2[y_km == 6, 1],
                c='purple',
                label='Cluster 7')
    plt.scatter(X_pca2[y_km == 7, 0],
                X_pca2[y_km == 7, 1],
                c='gray',
                label='Cluster 8')
    plt.scatter(km.cluster_centers_[:, 0],
                km.cluster_centers_[:, 1],
                s=85,
                alpha=0.75,
                marker='o',
                c='black',
                label='Centroids')

    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


if __name__ == "__main__":
    #prepare_data()
    #pca()
    kmeans()