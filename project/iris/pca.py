from plotly.graph_objs import Scatter, Line, Marker, Data, Layout, Figure, YAxis, XAxis
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly
import pandas as pd
from sklearn.decomposition import PCA

plotly.tools.set_credentials_file(username='JasminaEminovski', api_key='XRGrENKXzfi4aiPnavfi')

def pca():
    df = pd.read_csv(
        filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        sep=',')

    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    df.dropna(how="all", inplace=True)  # drops the empty line at file-end

    df.tail()

    X = df.ix[:, 0:4].values
    y = df.ix[:, 4].values
    X_std = StandardScaler().fit_transform(X)

    sklearn_pca = sklearnPCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(X_std)

    traces = []

    for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
        trace = Scatter(
            x=Y_sklearn[y == name, 0],
            y=Y_sklearn[y == name, 1],
            mode='markers',
            name=name,
            marker=Marker(
                size=12,
                line=Line(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8))
        traces.append(trace)

    data = Data(traces)
    layout = Layout(xaxis=XAxis(title='PC1', showline=False),
                    yaxis=YAxis(title='PC2', showline=False))
    fig = Figure(data=data, layout=layout)
    plotly.plotly.iplot(fig)


def pcaMoj():
    df = pd.read_csv(
        filepath_or_buffer='../data/1.csv',
        header=0,
        sep=',')
    print(df)

    df.columns = ["DAY","DAY_OF_WEEK","SCHEDULED_DEPARTURE","DEPARTURE_TIME","DEPARTURE_DELAY","TAXI_OUT",
                  "WHEELS_OFF","SCHEDULED_TIME","ELAPSED_TIME","AIR_TIME","DISTANCE","WHEELS_ON","TAXI_IN",
                  "SCHEDULED_ARRIVAL","ARRIVAL_TIME","ARRIVAL_DELAY","AIR_SYSTEM_DELAY",
                  "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","IsLate"]
    df.dropna(how="all", inplace=True)  # drops the empty line at file-end

    df.tail()

    X = df.ix[:, 0:21].values
    y = df.ix[:, 21].values
    X_std = StandardScaler().fit_transform(X)

    sklearn_pca = sklearnPCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(X_std)

    traces = []

    for name in ('Late', 'Not late'):
        trace = Scatter(
            x=Y_sklearn[y == name, 0],
            y=Y_sklearn[y == name, 1],
            mode='markers',
            name=name,
            marker=Marker(
                size=12,
                line=Line(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8))
        traces.append(trace)

    data = Data(traces)
    layout = Layout(xaxis=XAxis(title='PC1', showline=False),
                    yaxis=YAxis(title='PC2', showline=False))
    fig = Figure(data=data, layout=layout)
    plotly.plotly.iplot(fig)