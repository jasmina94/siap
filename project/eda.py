import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
from scipy.stats import norm
import os


pd.options.display.float_format = '{:.0f}'.format


def read_data_and_replace_missing():
    path = os.path.abspath("C:/Users/Jasmina/Documents/Faks/SIAP/avioni/flight-delays/flights.csv")
    flights = pd.read_csv(path)
    flights = flights.fillna(0)
    print(flights.head())
    return flights


def distribution_arrival_delay():
    #Histogram za sve
    flights = read_data_and_replace_missing()
    ax = sns.distplot(flights['ARRIVAL_DELAY'])
    plt.show()
    skewness = flights['ARRIVAL_DELAY'].skew()
    kurtosis = flights['ARRIVAL_DELAY'].kurt()
    print("Skewness for arrival delay all: %.2f" % skewness)
    print("Kurtosis for arrival delay all: %.2f" % kurtosis)


def correlation_matrix_all():
    flights = read_data_and_replace_missing()
    flights.insert(0, 'NoName', range(0, 0 + len(flights)))
    corrmat = flights.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()


def fix_date():
    flights = read_data_and_replace_missing()
    flights['DEPARTURE_DATE'] = pd.to_datetime(flights['YEAR'] * 10000 + flights['MONTH'] * 100 + flights['DAY'], format='%Y%m%d')
    print(flights.head())
    flights = flights.drop("YEAR", 1)  # Converted to date
    flights = flights.drop("MONTH", 1)  # Converted to date
    flights = flights.drop("DAY", 1)  # Converted to date
    flights = flights.drop("DAY_OF_WEEK", 1)  # Converted to date
    return flights


def define_flight_status():
    flights = read_data_and_replace_missing()
    for flight in flights:
        flights.loc[flights['ARRIVAL_DELAY'] <= 15, 'Status'] = 0  # on time
        flights.loc[flights['ARRIVAL_DELAY'] >= 15, 'Status'] = 1  # slightly delayed
        flights.loc[flights['ARRIVAL_DELAY'] >= 60, 'Status'] = 2  # highly delayed
        flights.loc[flights['DIVERTED'] == 1, 'Status'] = 3        # diverted
        flights.loc[flights['CANCELLED'] == 1, 'Status'] = 4       # cancelled
    return flights


def define_flight_status_and_plot_statistics():
    flights = define_flight_status()
    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    flights['Status'].value_counts().plot.pie(explode=[0.05, 0.05, 0.05, 0, 0], autopct='%1.1f%%', ax=ax[0],shadow=True)
    ax[0].set_title('Status')
    ax[0].set_ylabel('')
    sns.countplot('Status', order=flights['Status'].value_counts().index, data=flights, ax=ax[1])
    ax[1].set_title('Status')
    plt.show()


def define_delayed_reason():
    LATE_AIRCRAFT_DELAY = 0
    AIRLINE_DELAY = 0
    WEATHER_DELAY = 0
    SECURITY_DELAY = 0
    AIR_SYSTEM_DELAY = 0

    flights = define_flight_status()
    delayed_flights = flights[(flights.Status >= 1) & (flights.Status < 3)]
    quantity, att = delayed_flights.shape # 1063439 zakasnelih


    total_arrival_delay = delayed_flights["ARRIVAL_DELAY"].sum()
    AIR_SYSTEM_DELAY = delayed_flights["AIR_SYSTEM_DELAY"].sum()
    SECURITY_DELAY = delayed_flights["SECURITY_DELAY"].sum()
    WEATHER_DELAY = delayed_flights["WEATHER_DELAY"].sum()
    AIRLINE_DELAY = delayed_flights["AIRLINE_DELAY"].sum()
    LATE_AIRCRAFT_DELAY = delayed_flights["LATE_AIRCRAFT_DELAY"].sum()

    print("Total: %.2f" % total_arrival_delay)
    print("SYS: %.2f" % AIR_SYSTEM_DELAY)
    print("SEC: %.2f" % SECURITY_DELAY)
    print("WH: %.2f" % WEATHER_DELAY)
    print("AI: %.2f" % AIRLINE_DELAY)
    print("LA: %.2f" % LATE_AIRCRAFT_DELAY)


def define_cancellation_reason():
    #Get flights with statuses
    flights = define_flight_status()
    cancelled_flights = flights[(flights.Status == 4)]
    cancelled_flights.loc[cancelled_flights["CANCELLATION_REASON"] == "A", 'CANCELLATION_REASON'] = "AIRLINE/CARRIER"  # 0 Airline or carrier
    cancelled_flights.loc[cancelled_flights["CANCELLATION_REASON"] == "B", 'CANCELLATION_REASON'] = "WEATHER"  # 1 Weather
    cancelled_flights.loc[cancelled_flights["CANCELLATION_REASON"] == "C", 'CANCELLATION_REASON'] = "NAS"  # 2 National Air System
    cancelled_flights.loc[cancelled_flights["CANCELLATION_REASON"] == "D", 'CANCELLATION_REASON'] = "SECURITY"  # 3 Security

    #Fix date
    cancelled_flights['DEPARTURE_DATE'] = pd.to_datetime(flights['YEAR'] * 10000 + flights['MONTH'] * 100 + flights['DAY'],
                                               format='%Y%m%d')
    cancelled_flights = cancelled_flights.drop("YEAR", 1)
    cancelled_flights = cancelled_flights.drop("MONTH", 1)
    cancelled_flights = cancelled_flights.drop("DAY", 1)
    cancelled_flights = cancelled_flights.drop("DAY_OF_WEEK", 1)

    #Plot data
    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    cancelled_count = cancelled_flights['CANCELLATION_REASON'].value_counts()
    print(cancelled_count)
    cancelled_count.plot.pie(explode=[0.05, 0.05, 0.05, 0], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_ylabel('')
    sns.countplot('CANCELLATION_REASON', order=cancelled_flights['CANCELLATION_REASON'].value_counts().index, data=cancelled_flights, ax=ax[1])
    plt.show()

    cancelled_flights[['DEPARTURE_DATE', 'CANCELLATION_REASON']].groupby(['DEPARTURE_DATE']).count().plot()
    plt.show()


def distribution_arrival_delay_delayed_flights():
    #Histogram za kasnjenje
    flights = define_flight_status()
    delayed_flights = flights[(flights.Status >= 1) & (flights.Status < 3)]
    sns.distplot(delayed_flights['ARRIVAL_DELAY'])
    plt.show()

    skewness = delayed_flights['ARRIVAL_DELAY'].skew()
    kurtosis = delayed_flights['ARRIVAL_DELAY'].kurt()
    print("Skewness: %f" % skewness)
    print("Kurtosis: %f" % kurtosis)


def define_avg_and_sum_delay_by_month():
    # Procena po mesecima prosecno kasnjenje u minutima i broj minuta ukupno u jednom mesecu
    flights = define_flight_status()
    delayed_flights = flights[(flights.Status >= 1) & (flights.Status < 3)]

    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    delayed_flights[['MONTH', 'ARRIVAL_DELAY']].groupby(['MONTH']).mean().plot(ax=ax[0])
    ax[0].set_title('Average delay by month')
    delayed_flights[['MONTH', 'ARRIVAL_DELAY']].groupby(['MONTH']).sum().plot(ax=ax[1])
    ax[1].set_title('Broj minuta kašnjenja u dolasku po mesecu')
    plt.show()


def relation_scheduled_dep_and_arrival_delay():
    # Dijagram za odnos planiranog polaska i kasnjenja u dolasku -> regresija
    flights = define_flight_status()
    delayed_flights = flights[(flights.Status >= 1) & (flights.Status < 3)]
    sns.jointplot(x='SCHEDULED_DEPARTURE', y='ARRIVAL_DELAY', data=delayed_flights, kind='reg', color='y', fit_reg=True)
    plt.show()


def get_max_and_min_delay():
    # Ako je arrival delay manji od nule -> dosao je pre planiranog vremena
    flights = read_data_and_replace_missing()
    max_arrival_delay = max(flights["ARRIVAL_DELAY"])
    min_arrival_delay = min(flights["ARRIVAL_DELAY"])
    max_departure_delay = max(flights["DEPARTURE_DELAY"])
    min_departure_delay = min(flights["DEPARTURE_DELAY"])
    return max_arrival_delay, min_arrival_delay, max_departure_delay, min_departure_delay


def get_max_arrival_delay_info():
    flights = read_data_and_replace_missing()
    max_flight_info = flights.loc[flights["ARRIVAL_DELAY"].idxmax()]
    return max_flight_info


def correlation_matrix_for_delayed():
    # Korelacija izmedju onih koji su kasnili
    flights = define_flight_status()
    delayed_flights = flights[(flights.Status >= 1) & (flights.Status < 3)]
    delayed_correlation_matrix = delayed_flights.corr()

    k = 10  # number of variables for heatmap
    f, ax = plt.subplots(figsize=(12, 9))
    cols = delayed_correlation_matrix.nlargest(k, 'ARRIVAL_DELAY')['ARRIVAL_DELAY'].index
    print(cols)
    cm = np.corrcoef(delayed_flights[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values,xticklabels=cols.values)
    plt.show()

    df2 = delayed_flights.filter(['MONTH', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], axis=1)
    df2 = df2.groupby('MONTH')['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'].sum().plot()
    df2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
    plt.show()

    # Scatterplot - kakva je zavisnost izmedju razloga
    sns.set()
    cols = ['ARRIVAL_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'WEATHER_DELAY']
    sns.pairplot(delayed_flights[cols], size=2.5)
    plt.show()


def airlines_statistics():
    flights = define_flight_status()
    delayed_flights = flights[(flights.Status >= 1) & (flights.Status < 3)]
    airline_counts = flights['AIRLINE'].value_counts()  #14 avio-kompanija

    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.barplot('AIRLINE', 'AIRLINE_DELAY', data=delayed_flights, ax=ax[0],order=['UA', 'AA', 'US', 'F9', 'B6', 'OO', 'AS', 'NK','WN', 'DL', 'EV', 'HA', 'MQ', 'VX'])
    ax[0].set_title('Prosečno kašnjenje po avio-kompaniji')

    sns.boxplot('AIRLINE', 'AIRLINE_DELAY', data=delayed_flights, ax=ax[1],order=['UA', 'AA', 'US', 'F9', 'B6', 'OO', 'AS', 'NK','WN', 'DL', 'EV', 'HA', 'MQ', 'VX'])
    ax[1].set_title('Raspodela kašnjenja po avio-kompaniji')

    plt.close(2)
    plt.show()


def airports_statistics():
    # Having more than 300 airports on the dataset, we are going to focus on the top20.
    flights = define_flight_status()
    delayed_flights = flights[(flights.Status >= 1) & (flights.Status < 3)]

    top_20_airports = delayed_flights[(delayed_flights.ORIGIN_AIRPORT == 'ATL') | (delayed_flights.ORIGIN_AIRPORT == 'LAX') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'ORD') | (delayed_flights.ORIGIN_AIRPORT == 'DFW') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'JFK') | (delayed_flights.ORIGIN_AIRPORT == 'DEN') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'SFO') | (delayed_flights.ORIGIN_AIRPORT == 'LAS') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'CLT') | (delayed_flights.ORIGIN_AIRPORT == 'SEA') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'PHX') | (delayed_flights.ORIGIN_AIRPORT == 'MIA') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'MCO') | (delayed_flights.ORIGIN_AIRPORT == 'IAH') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'EWR') | (delayed_flights.ORIGIN_AIRPORT == 'MSP') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'BOS') | (delayed_flights.ORIGIN_AIRPORT == 'DTW') |
                                   (delayed_flights.ORIGIN_AIRPORT == 'PHL') | (delayed_flights.ORIGIN_AIRPORT == 'LGA')]
    print(top_20_airports['ORIGIN_AIRPORT'].value_counts())

    #Origin airport - AIR_SYSTEM_DELAY
    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.barplot('ORIGIN_AIRPORT', 'AIR_SYSTEM_DELAY', data=top_20_airports, ax=ax[0],
                order=['ATL', 'LAX', 'ORD', 'DFW', 'JFK', 'DEN', 'SFO', 'LAS', 'CLT','SEA',
                       'PHX', 'MIA', 'MCO', 'IAH', 'EWR', 'MSP', 'BOS', 'PHL', 'DTW', 'LGA'])
    ax[0].set_title('Prosečno kašnjenje po polaznom aerodromu')

    sns.boxplot('ORIGIN_AIRPORT', 'AIR_SYSTEM_DELAY', data=top_20_airports, ax=ax[1],
                order=['ATL', 'LAX', 'ORD', 'DFW', 'JFK', 'DEN', 'SFO', 'LAS', 'CLT', 'SEA',
                       'PHX', 'MIA', 'MCO', 'IAH', 'EWR', 'MSP', 'BOS', 'PHL', 'DTW', 'LGA'])

    ax[1].set_title('Raspodela kašnjenja po polaznom aerodromu')
    plt.close(2)
    plt.show()

    #Destination airport - weather delay
    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.barplot('DESTINATION_AIRPORT', 'WEATHER_DELAY', data=top_20_airports, ax=ax[0],
                order=['ATL', 'LAX', 'ORD', 'DFW', 'JFK', 'DEN', 'SFO', 'LAS', 'CLT', 'SEA',
                       'PHX', 'MIA', 'MCO', 'IAH', 'EWR', 'MSP', 'BOS', 'PHL', 'DTW', 'LGA'])
    ax[0].set_title('Prosečno kašnjenje po polaznom aerodromu')

    sns.boxplot('DESTINATION_AIRPORT', 'WEATHER_DELAY', data=top_20_airports, ax=ax[1],
                order=['ATL', 'LAX', 'ORD', 'DFW', 'JFK', 'DEN', 'SFO', 'LAS', 'CLT', 'SEA',
                       'PHX', 'MIA', 'MCO', 'IAH', 'EWR', 'MSP', 'BOS', 'PHL', 'DTW', 'LGA'])

    ax[1].set_title('Raspodela kašnjenja po polaznom aerodromu')
    plt.close(2)
    plt.show()



if __name__ == "__main__":
    print("EDA Air delays")
    flights = read_data_and_replace_missing()
    x, y = flights.shape
    print(x)
    print(y)
    #distribution_arrival_delay()
    #correlation_matrix_all()
    #flights = fix_date()
    #define_flight_status_and_plot_statistics()
    #define_cancellation_reason()
    #correlation_matrix_for_delayed()
    #max_arr, min_arr, max_dep, min_dep = get_max_and_min_delay()
    #print(max_arr, min_arr, max_dep, min_dep)
    #max_info = get_max_arrival_delay_info()
    #print(max_info)
    #airlines_statistics()
    #define_cancellation_reason()
    #define_delayed_reason()