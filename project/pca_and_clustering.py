import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

january_path = "C:/Users/Jasmina/Desktop/classification/january.csv"


def prepare_data_for_rapid_miner():
    path = os.path.abspath("C:/Users/Jasmina/Documents/Faks/SIAP/avioni/flight-delays/flights.csv")
    flights = pd.read_csv(path)
    airports = pd.read_csv("data/airports.csv")

    flights = flights[flights['CANCELLED'] != 1]
    flights = flights[flights['DIVERTED'] != 1]

    flights = flights.fillna(0)

    flights = flights.drop(columns=['YEAR', 'DAY', 'FLIGHT_NUMBER', 'TAIL_NUMBER','LATE_AIRCRAFT_DELAY', 'AIRLINE_DELAY', 'WEATHER_DELAY', 'SECURITY_DELAY',
                                    'AIR_SYSTEM_DELAY','CANCELLATION_REASON', 'CANCELLED', 'DIVERTED', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'AIRLINE'])

    flights = flights.sample(n=500000)


    #flights.to_csv("data/pca.csv")
    return flights


def read_data_and_replace_missing():
    path = os.path.abspath("C:/Users/Jasmina/Documents/Faks/SIAP/avioni/flight-delays/flights.csv")
    flights = pd.read_csv(path)
    return flights


def prepare_dataset():
    flights = read_data_and_replace_missing()
    #flights['DATE'] = pd.to_datetime(flights[['YEAR', 'MONTH', 'DAY']]) #date time value ne prolazi

    #flights = flights.dropna(axis=0, how='any')
    flights = flights[flights['CANCELLED'] != 1]
    flights = flights[flights['DIVERTED'] != 1]
    flights = flights.fillna(0)
    flights = flights.drop(columns=['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                                    'LATE_AIRCRAFT_DELAY', 'AIRLINE_DELAY', 'WEATHER_DELAY', 'SECURITY_DELAY', 'AIR_SYSTEM_DELAY',
                                    'CANCELLATION_REASON', 'CANCELLED', 'DIVERTED',
                                    'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
    print(flights.head())
    print("***")
    print(flights.shape)
    return flights


def prepare_for_pca():
    flights = prepare_dataset()
    X = flights.iloc[:, 1:].values
    y = flights.iloc[:, 0].values

    ss = StandardScaler()
    X_std = ss.fit_transform(X)

    cov_mat = np.cov(X_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    return eigen_vals, eigen_vecs, cov_mat, X_std, flights


def prep():
   # flights = prepare_data_for_rapid_miner()
    flights = pd.read_csv(january_path)
    flights = flights.fillna(0)
    X = flights.iloc[:, 1:].values
    y = flights.iloc[:, 0].values

    ss = StandardScaler()
    X_std = ss.fit_transform(X)

    cov_mat = np.cov(X_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    return eigen_vals, eigen_vecs, cov_mat, X_std, flights


# index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6',
# 'PC-7', 'PC-8', 'PC-9', 'PC-10', 'PC-11', 'PC-12', 'PC-13', 'PC-14']

def alternative_pca():
    flights = prepare_dataset()
    data_scaled = pd.DataFrame(preprocessing.scale(flights), columns=flights.columns)
    # PCA
    pca = PCA(n_components=3)
    pca.fit_transform(data_scaled)

    # Dump components relations with features:
    print(pd.DataFrame(pca.components_, columns=data_scaled.columns,
                       index=['PC-1', 'PC-2', 'PC-3']))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # assign x,y,z coordinates from PC1, PC2 & PC3
    xs = pca.components_[0]
    ys = pca.components_[1]
    zs = pca.components_[2]

    # initialize scatter plot and label axes
    ax.scatter(xs, ys, zs, alpha=0.75, c='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.show()


def first_pca():
    eigen_values, eigen_vectors, cov_mat, X_std, flights = prep()
    print("AAAAA SHAPE")
    print(flights.shape)
    tot = sum(eigen_values)
    # var_exp ratio is fraction of eigen_val to total sum
    var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
      # calculate the cumulative sum of explained variances
    cum_var_exp = np.cumsum(var_exp)

    #Prvi grafik
    plt.bar(range(1, 27), var_exp, alpha=0.75, align='center', label='pojedinacna varijansa komponenti')
    plt.step(range(1, 27), cum_var_exp, where='mid', label='kumulativna varijansa komponenti')
    plt.ylim(0, 1.1)
    plt.xlabel('Glavne komponente')
    plt.ylabel('Odnos varijanse')
    plt.legend(loc='best')
    plt.show()

    #PCA primena
    pca_2 = PCA(n_components=2)  # PCA with 3 primary components 80% tacnosti
    pca_3 = PCA(n_components=3)  # PCA with 4 primary components oko 85% tacnosti

    # fit and transform both PCA models
    X_pca_2 = pca_2.fit_transform(X_std)
    X_pca_3 = pca_3.fit_transform(X_std)

    # 2d grafik
    plt.scatter(X_pca_2.T[0], X_pca_2.T[1], alpha=0.75, c='blue')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    # 3d grafik
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # assign x,y,z coordinates from PC1, PC2 & PC3
    xs = X_pca_3.T[0]
    ys = X_pca_3.T[1]
    zs = X_pca_3.T[2]

    #initialize scatter plot and label axes
    ax.scatter(xs, ys, zs, alpha=0.75, c='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    plot = ax.scatter(xs, ys, zs, alpha=0.75,
                      c=flights['ARRIVAL_DELAY'], cmap='viridis', depthshade=True)

    fig.colorbar(plot, shrink=0.6, aspect=9)
    plt.show()
    return X_pca_2


def treca():
    data = prepare_dataset()
    pca = PCA(0.95)
    data = StandardScaler().fit_transform(data)
    pca.fit(data)
    pca_data = pca.transform(data)

    plt.semilogy(pca.explained_variance_ratio_, '--o')
    plt.show()

    pca_headers = ['target']
    for i in range(pca.n_components_):
        pca_headers.append('PCA_component_' + str(i + 1))

   # pca_data.to_csv('data/new.csv')


    print(pca)
    print(pca_data)
    print(pca_headers)


def k_means():
    eigen_values, eigen_vectors, cov_mat, X_std, flights = prep()
    X_pca = first_pca()
    distortions = []  # sum of squared error within the each cluster
    for i in range(1, 11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X_std)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker='o', alpha=0.75)
    plt.xlabel('Broj klastera')
    plt.ylabel('Distorzija')
    plt.show()

    km = KMeans(n_clusters=6,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)

    y_km = km.fit_predict(X_pca)

    plt.scatter(X_pca[y_km == 0, 0],
                X_pca[y_km == 0, 1],
                c='lightgreen',
                label='Cluster 1')
    plt.scatter(X_pca[y_km == 1, 0],
                X_pca[y_km == 1, 1],
                c='orange',
                label='Cluster 2')
    plt.scatter(X_pca[y_km == 2, 0],
                X_pca[y_km == 2, 1],
                c='lightblue',
                label='Cluster 3')
    plt.scatter(X_pca[y_km == 3, 0],
                X_pca[y_km == 3, 1],
                c='red',
                label='Cluster 4')
    plt.scatter(X_pca[y_km == 4, 0],
                X_pca[y_km == 4, 1],
                c='pink',
                label='Cluster 5')
    plt.scatter(X_pca[y_km == 5, 0],
                X_pca[y_km == 5, 1],
                c='yellow',
                label='Cluster 6')
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


def dbscan():
    X_pca = first_pca()
    dbs = DBSCAN(eps=0.75,
                 min_samples=5)

    y_dbs = dbs.fit_predict(X_pca)
    plt.scatter(X_pca[y_dbs == -1, 0],
                X_pca[y_dbs == -1, 1],
                c='lightgreen',
                label='Cluster 1')
    plt.scatter(X_pca[y_dbs == 0, 0],
                X_pca[y_dbs == 0, 1],
                c='orange',
                label='Cluster 2')
    plt.scatter(X_pca[y_dbs == 1, 0],
                X_pca[y_dbs == 1, 1],
                c='lightblue',
                label='Cluster 3')
    plt.scatter(X_pca[y_dbs == 2, 0],
                X_pca[y_dbs == 2, 1],
                c='yellow',
                label='Cluster 4')
    plt.scatter(X_pca[y_dbs == 3, 0],
                X_pca[y_dbs == 3, 1],
                c='pink',
                label='Cluster 5')
    plt.scatter(X_pca[y_dbs == 4, 0],
                X_pca[y_dbs == 4, 1],
                c='purple',
                label='Cluster 6')

    plt.legend(loc='best')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


def dbscan_test():
    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    print("Labels")
    print(labels_true)

    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def dbscan_final():
    flights = prepare_data_for_rapid_miner()
    X = StandardScaler().fit_transform(flights)
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.8, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    print(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


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


def data_info():
    path = os.path.abspath("C:/Users/Jasmina/Documents/Faks/SIAP/avioni/flight-delays/flights.csv")
    flights = pd.read_csv(path)
    airports = pd.read_csv("data/airports.csv")
    airlines = pd.read_csv("data/airlines.csv")

    #Uzimamo samo januar i jun
    january_flights = flights[flights['MONTH'] == 1]
    june_flights = flights[flights['MONTH'] == 6]

    #Ne gledamo letove koji su preusmereni ili otkazani
    january_flights = january_flights[january_flights['CANCELLED'] != 1]
    june_flights = june_flights[june_flights['DIVERTED'] != 1]

    #Menjamo airline
    airlines = make_airline_dict()
    january_flights = january_flights.replace({"AIRLINE" : airlines})
    june_flights = june_flights.replace({"AIRLINE": airlines})

    #Popunjavamo nedostajujuce vrednosti
    january_flights = january_flights.fillna(0)
    june_flights = june_flights.fillna(0)

    #Odbacujemo kodove aerodroma i koristimo geo podatke iz airports.csv
    january_flights['ORIGIN_AIRPORT_LAT'] = january_flights['ORIGIN_AIRPORT'].map(airports.set_index('IATA_CODE')['LATITUDE'])
    january_flights['ORIGIN_AIRPORT_LON'] = january_flights['ORIGIN_AIRPORT'].map(airports.set_index('IATA_CODE')['LONGITUDE'])
    january_flights['DESTINATION_AIRPORT_LAT'] = january_flights['DESTINATION_AIRPORT'].map(airports.set_index('IATA_CODE')['LATITUDE'])
    january_flights['DESTINATION_AIRPORT_LON'] = january_flights['DESTINATION_AIRPORT'].map(airports.set_index('IATA_CODE')['LONGITUDE'])

    june_flights['ORIGIN_AIRPORT_LAT'] = june_flights['ORIGIN_AIRPORT'].map(airports.set_index('IATA_CODE')['LATITUDE'])
    june_flights['ORIGIN_AIRPORT_LON'] = june_flights['ORIGIN_AIRPORT'].map(airports.set_index('IATA_CODE')['LONGITUDE'])
    june_flights['DESTINATION_AIRPORT_LAT'] = june_flights['DESTINATION_AIRPORT'].map(airports.set_index('IATA_CODE')['LATITUDE'])
    june_flights['DESTINATION_AIRPORT_LON'] = june_flights['DESTINATION_AIRPORT'].map(airports.set_index('IATA_CODE')['LONGITUDE'])

    #Odbacujemo suvisne kolone
    january_flights = january_flights.drop(columns=['YEAR', 'MONTH', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                                                    'CANCELLATION_REASON', 'CANCELLED', 'DIVERTED', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
    june_flights = june_flights.drop(columns=['YEAR', 'MONTH', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                                                    'CANCELLATION_REASON', 'CANCELLED', 'DIVERTED', 'ORIGIN_AIRPORT',
                                                    'DESTINATION_AIRPORT'])

    #Oznacavanje podataka isLate
    january_flights.loc[january_flights['ARRIVAL_DELAY'] <= 15, 'IS_LATE'] = 0
    january_flights.loc[january_flights['ARRIVAL_DELAY'] > 15, 'IS_LATE'] = 1
    june_flights.loc[june_flights['ARRIVAL_DELAY'] <= 15, 'IS_LATE'] = 0
    june_flights.loc[june_flights['ARRIVAL_DELAY'] > 15, 'IS_LATE'] = 1

    print(january_flights.head(1))
    print(january_flights.shape)
    print(june_flights.head(1))
    print(june_flights.shape)

    january_flights.to_csv("data/january.csv")
    june_flights.to_csv("data/june.csv")


if __name__ == "__main__":
    first_pca()
    #alternative_pca()
    #treca()
    #prepare_data_for_rapid_miner()
    #k_means()
    #dbscan_test()
    #dbscan_final()
    #data_info()









