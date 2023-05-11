import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
import matplotlib.pyplot as plt
import seaborn as sns
from knn import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def Data_clean():

    # Lectura de datos
    df = pd.read_csv('dataset_phishing.csv')
    scaler = StandardScaler()

    # Transformación de datos
    df.loc[df['status'] == 'phishing', 'status'] = -1
    df.loc[df['status'] == 'legitimate', 'status'] = 1
    df = df.drop('url', axis=1)
    df_temp = df
    df['status'] = df['status'].astype(int)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Cantidad de elementos por clase
    value_counts = df['status'].value_counts()

    return df, df_temp


def Analisis_exploratorio(df):

    # Distribución de los datos
    with plt.style.context(style="fivethirtyeight"):
        plt.pie(x=dict(df['status'].value_counts()).values(),
                labels=dict(df['status'].value_counts()).keys(),
                autopct="%.2f%%",
                colors=['red', 'orangered'],
                startangle=90,
                explode=[0, 0.05])
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(label="Analysing status feature using donut-chart")
        plt.show(block=True)

    corr = df.corr()['status']
    columns = corr[corr > 0.1].index.tolist()
    linear_coefficients = []
    model = LinearRegression()
    df_without_status = df.drop('status', axis=1)
    for column in df_without_status.columns:
        model.fit(df[['status']], df[column])
        if model.score(df[['status']], df[column]) > 0.2:
            linear_coefficients.append(column)

    columns = list(set(linear_coefficients + columns + ['status']))
    df = df[columns]

    print("\n Las variables que se usaron: ")
    a = ""
    for c in columns:
        a += c+", "

    print(a)
    print("**********************************")

    return df


"""
El dataset si esta balanceado por lo que si es posible utilizar la métrica de accuracy.

Para la selección de variables se hizo un análisis de correlación entre las variables 
y aquellas que tuvieran un idice de correlación mayor a 0.1 se tomaron en cuenta.
Luego se realizo una regresión lineal y aquellas que presentaran un score mayor a 0.2 se tomaron en cuenta.
Estos fueron los criterios que uilitizamos para la selección de variables.
    
"""
