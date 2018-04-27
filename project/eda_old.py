import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests


flights_path = "C:/Users/Jasmina/Desktop/data/flights.csv"
airlines_path = "C:/Users/Jasmina/Desktop/data/airlines.csv"
airports_path = "C:/Users/Jasmina/Desktop/data/airports.csv"


def late_or_cancelled(x):
    if x['CANCELLED'] == 1 or x['ARRIVAL_DELAY'] > 15:
        return 1
    else:
        return 0


def day_31_to_365(x):
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_365 = days_in_month[:x['MONTH']-1].sum() + x['DAY']
    return days_365


def load_data():
    n = 5819078                                     # total number of datapoints = 5819078
    skip = random.sample(range(1, n), n - 20000)    # skip = sorted(random.sample(range(1,n),n-2000))
    flights = pd.read_csv(flights_path, skiprows=skip,
                          usecols=['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                                   'SCHEDULED_DEPARTURE','DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'CANCELLED'], low_memory=False)
    airports = pd.read_csv(airlines_path)
    airlines = pd.read_csv(airlines_path)
    flights = flights.sample(frac=1).reset_index(
        drop=True)                      # here I randomize rows so that data is not chronologically sorted

    all_features = flights.columns.values
    return flights, airports, airlines, all_features


def make_aircode_dict():
    url1 = "https://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_AIRPORT"
    s1 = requests.get(url1).content
    url2 = "https://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_AIRPORT_ID"
    s2 = requests.get(url2).content

    aircode1 = pd.read_csv(io.StringIO(s1.decode('utf-8')))
    aircode2 = pd.read_csv(io.StringIO(s2.decode('utf-8')))

    # Format the airport codes
    aircode1 = aircode1.reset_index()
    aircode2 = aircode2.reset_index()
    aircodes = pd.merge(aircode1, aircode2, on='Description')
    aircode_dict = dict(zip(aircodes['Code_y'].astype(str), aircodes['Code_x']))

    return aircode_dict


def change_airport_codes(data, aircode_dict):
    for i in range(len(data)):
        if len(data['ORIGIN_AIRPORT'][i]) != 3:
            to_replace = data['ORIGIN_AIRPORT'][i]
            value = aircode_dict[data['ORIGIN_AIRPORT'][i]]
            data = data.replace(to_replace, value)
    for i in range(len(data)):
        if len(data['DESTINATION_AIRPORT'][i]) != 3:
            to_replace = data['DESTINATION_AIRPORT'][i]
            value = aircode_dict[data['DESTINATION_AIRPORT'][i]]
            data = data.replace(to_replace, value)

    return data


if __name__ == "__main__":
    flights, airports, airlines, all_features = load_data()

    # Make sure all Origin and departing airports are strings
    flights['ORIGIN_AIRPORT'] = flights['ORIGIN_AIRPORT'].values.astype(str)
    flights['DESTINATION_AIRPORT'] = flights['DESTINATION_AIRPORT'].values.astype(str)

    aircode_dict = make_aircode_dict()

    flights = change_airport_codes(flights, aircode_dict)

    Depart_airport = str(input("Specify the departure airport (three letters) or put ALL:  ")).upper()
    filter_flights = []
    if Depart_airport in list(flights['ORIGIN_AIRPORT']):
        filter_flights = flights[flights['ORIGIN_AIRPORT'] == Depart_airport]
    elif Depart_airport == 'ALL':
        print('Analyzing all airports')
    else:
        print("incorrect airport code")

    print(filter_flights)  # isprintava samo za prosledjeni aerodrom

    #Zakasnio ili ne
    flights['late or cancelled'] = flights.apply(late_or_cancelled, axis=1)
    flights = flights[flights['CANCELLED'] == 0]

    flights['DAY'] = flights.apply(day_31_to_365, axis=1)
    flights['WEEK'] = flights['DAY'] // 7  #zaokruzeno deljenje bez decimale week je sada broj nedelje u celoj godini
    del flights['DAY']

    flights['SCHEDULED_DEPARTURE'] = np.ceil(flights['SCHEDULED_DEPARTURE'] / 600).apply(int)

    del flights['ARRIVAL_DELAY']
    del flights['DEPARTURE_DELAY']
    del flights['CANCELLED']

    Delay_vs_Day_of_Week = pd.DataFrame(
        {'delays': flights.groupby(['DAY_OF_WEEK'])['late or cancelled'].mean()}).reset_index()
    Delay_vs_WEEK = pd.DataFrame({'delays': flights.groupby(['WEEK'])['late or cancelled'].mean()}).reset_index()
    Delay_vs_AIRLINE = pd.DataFrame(
        {'delays': flights.groupby(['AIRLINE'])['late or cancelled'].mean()})  # .reset_index()
    Delay_vs_SCHEDULED_DEPARTURE = pd.DataFrame(
        {'delays': flights.groupby(['SCHEDULED_DEPARTURE'])['late or cancelled'].mean()}).reset_index()


    print(Delay_vs_AIRLINE)


    fig = plt.figure(figsize=(15.0, 10.0), edgecolor='b', dpi=80)

    sub1 = fig.add_subplot(221)  # instead of plt.subplot(2, 2, 1)
    sub1.set_title('Week of Year', fontsize=12, color="green")
    sns.barplot(x="WEEK", y="delays", data=Delay_vs_WEEK, palette="BuGn_d", ax=sub1)
    sub1.set_xticks(list(range(0, 52, 10)))
    sub1.set_xticklabels(list(range(0, 52, 10)))

    sub2 = fig.add_subplot(222)
    sub2.set_title('Day of Week', fontsize=12, color="green")
    sns.barplot(x="DAY_OF_WEEK", y="delays", data=Delay_vs_Day_of_Week, palette="BuGn_d", ax=sub2)
    sub2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    sub3 = fig.add_subplot(223)
    sub3.set_title('Airline', fontsize=12, color="green")
    sns.barplot(y=list(range(len(Delay_vs_AIRLINE))), x=Delay_vs_AIRLINE['delays'], palette="BuGn_d", ax=sub3,
                orient="h")
    sub3.set_yticks(range(len(Delay_vs_AIRLINE)))
    sub3.set_yticklabels(Delay_vs_AIRLINE.index)

    sub4 = fig.add_subplot(224)
    sns.barplot(x=list(range(len(Delay_vs_SCHEDULED_DEPARTURE))), y=Delay_vs_SCHEDULED_DEPARTURE['delays'],
                palette="BuGn_d", ax=sub4)
    sub4.set_title('Scheduled Departure Time', fontsize=12, color="green")
    sub4.set_xticks([0, 1, 2, 3])
    sub4.set_xticklabels(['00:00 - 06:00', '06:00 - 12:00', '12:00 - 18:00', '18:00 - 00:00'])

    plt.show()