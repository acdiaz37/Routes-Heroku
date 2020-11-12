import math
import time
from datetime import date

import numpy as np
import pandas as pd
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from sklearn.cluster import KMeans, OPTICS
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize, StandardScaler

from .models import *

# Create your views here.

csv_filename = None
xlx_filename = None
ruta_waypts = []


def index(request):
    if request.user.is_authenticated:
        return redirect('routes/')
    else:
        return redirect('login/')


@login_required
def map_mainpage(request):
    global csv_filename
    global xlx_filename
    global ruta_waypts

    csv_filename = Archivos.objects.get(id=1).csv.path
    xlx_filename = Archivos.objects.get(id=1).xlx.path
    # print(type(csv_filename), type(xlx_filename))

    data = pd.read_csv(csv_filename)
    print(data)

    try:
        city = str(Vehiculo.objects.get(conductor_id=request.user.id).ciudad)
        driver = Despacho.objects.filter(vehiculo__conductor=request.user.id).latest().vehiculo.conductor

        print('city: ', city)
        print('driver: ', driver, str(driver.id), '\n')

        drop_indexes = data[(data['City'] != city) | (data['driver_id'] != driver.id)].index
        # print(city_indexes, '\n')

        data.drop(drop_indexes, inplace=True)
        data = data[
            ['ID', 'Client_Depot', 'Lat_[y]', 'Lon_[x]', 'Q_[Kg]', 'Si_[min]', 'ai_[min]', 'bi_[min]', 'City']
        ]

        datanew = data.rename(
            columns={
                "Lat_[y]": "Lat",
                "Lon_[x]": "Lon",
                "Q_[Kg]": "Q",
                "Si_[min]": "Si",
                "ai_[min]": "ai",
                "bi_[min]": "bi",
            }
        )

    except:

        messages.warning(request, 'Se ha detectado que este usuario no tiene despachos o vehiculo asignado')

        print('\nEs necesario asignar un vehículo y un despacho al usuario\n')

        data = data[
            ['ID', 'Client_Depot', 'Lat_[y]', 'Lon_[x]', 'Q_[Kg]', 'Si_[min]', 'ai_[min]', 'bi_[min]', 'City']
        ]

        datanew = data.rename(
            columns={
                "Lat_[y]": "Lat",
                "Lon_[x]": "Lon",
                "Q_[Kg]": "Q",
                "Si_[min]": "Si",
                "ai_[min]": "ai",
                "bi_[min]": "bi",
            }
        )

    # Importar base de datos.

    print(datanew)

    # datanew.head()

    # print("Total de nodos incluído el depósito:", len (datanew))

    # Leer base de datos vehiculos.
    dataveh = pd.read_excel(xlx_filename, sheet_name='vehiculos')

    # dataveh.head()

    # dataveh.info()
    # print("Total de vehículos disponibles:", len (dataveh))

    # Asignación ejes.
    datanew.dropna(axis=0, how='any', subset=['Lat', 'Lon'], inplace=True)

    # Asignación variables de latitud y longitud.
    X = datanew.loc[:, ['Client_Depot', 'Lat', 'Lon']]

    # Clusterización K-means
    K_clusters = range(1, 10)

    kmeans = [KMeans(n_clusters=i) for i in K_clusters]
    Y_axis = datanew[['Lat']]
    X_axis = datanew[['Lon']]

    score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

    # Número de clusters.

    # Carga total a transportar.
    Qtotal = datanew['Q'].sum()
    # print("Carga total a transportar:", Qtotal)

    # Vehiculos requeridos.
    Capcarga = dataveh['Capacity_[kg]'].mean()
    vehiculosr = math.ceil(Qtotal / Capcarga)
    # print("Vehiculos requeridos inicialmente (tentativo):", vehiculosr * 2)

    # Número de clusters.
    Ncluster = math.ceil(vehiculosr / 2)
    # print("Número de clusters K-means inicial:", Ncluster)

    # Predefinir el número de clusters.
    km = KMeans(n_clusters=Ncluster)

    # Predecir el cluster al que debe ir cada nodo con base en su ubicación.
    y_predicted = km.fit_predict(datanew[['Lon', 'Lat']])
    # y_predicted

    # Incluir a la base de datos la columna del cluster asignado.
    datanew['ClusterK'] = y_predicted
    dn = datanew
    # dn.head()

    # Centroides.
    # km.cluster_centers_

    # plt.rcParams['figure.figsize'] = [15, 13]

    data0 = datanew[datanew.ClusterK == 0]
    data1 = datanew[datanew.ClusterK == 1]
    data2 = datanew[datanew.ClusterK == 2]
    data3 = datanew[datanew.ClusterK == 3]
    data4 = datanew[datanew.ClusterK == 4]
    data5 = datanew[datanew.ClusterK == 5]

    # plt.scatter(data0.Lon, data0['Lat'],color='yellow')
    # plt.scatter(data1.Lon, data1['Lat'],color='red')
    # plt.scatter(data2.Lon, data2['Lat'],color='blue')
    # plt.scatter(data3.Lon, data3['Lat'],color='green')
    # plt.scatter(data4.Lon, data4['Lat'],color='gray')
    # plt.scatter(data5.Lon, data5['Lat'],color='orange')

    # plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*',label='centroid')
    # plt.legend()
    # plt.xlabel('Coordenada x = Longitud')
    # plt.ylabel('Coordenada y = Latitud')
    # plt.title('Clusterización', family='Arial', fontsize=14)

    # Filtrar base de datos con cluster "n".
    data0 = datanew[(datanew['ClusterK'] == 0)]
    d0 = data0  # .sort_values(by = 'Q', ascending = False)

    data1 = datanew[(datanew['ClusterK'] == 1)]
    d1 = data1  # .sort_values(by = 'Q', ascending = False)

    data2 = datanew[(datanew['ClusterK'] == 2)]
    d2 = data2  # .sort_values(by = 'Q', ascending = False)

    data3 = datanew[(datanew['ClusterK'] == 3)]
    d3 = data3  # .sort_values(by = 'Q', ascending = False)

    data4 = datanew[(datanew['ClusterK'] == 4)]
    d4 = data4  # .sort_values(by = 'Q', ascending = False)

    data5 = datanew[(datanew['ClusterK'] == 5)]
    d5 = data5  # .sort_values(by = 'Q', ascending = False)

    data6 = datanew[(datanew['ClusterK'] == 6)]
    d6 = data6  # .sort_values(by = 'Q', ascending = False)

    data7 = datanew[(datanew['ClusterK'] == 7)]
    d7 = data7  # .sort_values(by = 'Q', ascending = False)

    # Carga total a transportar en el cluster "n".
    Qt0 = data0['Q'].sum()
    # print("Total carga a transportar cluster 0:", Qt0)

    Qt1 = data1['Q'].sum()
    # print("Total carga a transportar cluster 1:", Qt1)

    Qt2 = data2['Q'].sum()
    # print("Total carga a transportar cluster 2:", Qt2)

    Qt3 = data3['Q'].sum()
    # print("Total carga a transportar cluster 3:", Qt3)

    Qt4 = data4['Q'].sum()
    # print("Total carga a transportar cluster 4:", Qt4)

    Qt5 = data5['Q'].sum()
    # print("Total carga a transportar cluster 5:", Qt5)

    Qt6 = data6['Q'].sum()
    # print("Total carga a transportar cluster 6:", Qt6)

    Qt7 = data7['Q'].sum()
    # print("Total carga a transportar cluster 7:", Qt7)

    # Filtrar base de datos con cluster "n".
    dataC0 = datanew[(datanew['ClusterK'] == 0)]
    dataDepot = datanew[(datanew['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, dataC0]
    concat0 = pd.concat(concat, sort='False', ignore_index='True')
    concat0.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    data0 = concat0
    # data0.head()

    # In[18]:

    # print("Cantidad de clientes en el cluster:", len(data0))

    # Selección y eliminación de columnas irrelevantes en para el analisis
    drop_features = ['Lat', 'Lon', 'ClusterK', 'ID', 'Client_Depot', 'City']
    X = data0.drop(drop_features, axis=1)

    # Filtro de los valores perdidos (si los hay)
    X.fillna(method='ffill', inplace=False)

    # X.head()

    # Escalar los datos para llevar todos los atributos a un nivel comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalizar los datos para que los datos sigan aproximadamente una distribución gaussiana.
    X_normalizada = normalize(X_scaled)

    # Conversión de la matriz numpy en un DataFrame de pandas
    X_normalizada = pd.DataFrame(X_normalizada)

    # Renombre de las columnas
    X_normalizada.columns = X.columns
    # X_normalizada.head()

    # Construcción del modelo de clusterización Optics para el cluster "n"
    optics_model = OPTICS(min_samples=11, xi=0.06, min_cluster_size=0.08)

    # Entrenamiento del modelo
    optics_model.fit(X_normalizada)

    # Creando una matriz numpy con números en espacios iguales hasta el rango especificado
    space = np.arange(len(X_normalizada))

    # Almacenamiento de la distancia de accesibilidad de cada punto
    reachability = optics_model.reachability_[optics_model.ordering_]

    # Almacenamiento de las etiquetas de grupo de cada punto
    labels = optics_model.labels_[optics_model.ordering_]
    # labels

    X1 = data0
    X1['ClusterO'] = labels
    dn = X1
    # dn.head()
    # print("Confirmación cantidad de clientes en el cluster:", len(dn))

    # Filtrar base de datos con cluster "n".
    data1n = dn[(dn['ClusterO'] == -1)]
    d1n = data1n  # .sort_values(by = 'Q', ascending = False)

    data0 = dn[(dn['ClusterO'] == 0)]
    d0 = data0  # .sort_values(by = 'Q', ascending = False)

    data1 = dn[(dn['ClusterO'] == 1)]
    d1 = data1  # .sort_values(by = 'Q', ascending = False)

    data2 = dn[(dn['ClusterO'] == 2)]
    d2 = data2  # .sort_values(by = 'Q', ascending = False)

    data3 = dn[(dn['ClusterO'] == 3)]
    d3 = data3  # .sort_values(by = 'Q', ascending = False)

    data4 = dn[(dn['ClusterO'] == 4)]
    d4 = data4  # .sort_values(by = 'Q', ascending = False)

    data5 = dn[(dn['ClusterO'] == 5)]
    d5 = data5  # .sort_values(by = 'Q', ascending = False)

    data6 = dn[(dn['ClusterO'] == 6)]
    d6 = data6  # .sort_values(by = 'Q', ascending = False)

    data7 = dn[(dn['ClusterO'] == 7)]
    d7 = data7  # .sort_values(by = 'Q', ascending = False)

    # Carga total a transportar en el cluster "n".
    Qt1n = data1n['Q'].sum()
    # print("Total carga a transportar cluster 1n:", Qt1n)

    Qt0 = data0['Q'].sum()
    # print("Total carga a transportar cluster 0:", Qt0)

    Qt1 = data1['Q'].sum()
    # print("Total carga a transportar cluster 1:", Qt1)

    Qt2 = data2['Q'].sum()
    # print("Total carga a transportar cluster 2:", Qt2)

    Qt3 = data3['Q'].sum()
    # print("Total carga a transportar cluster 3:", Qt3)

    Qt4 = data4['Q'].sum()
    # print("Total carga a transportar cluster 4:", Qt4)

    Qt5 = data5['Q'].sum()
    # print("Total carga a transportar cluster 5:", Qt5)

    Qt6 = data6['Q'].sum()
    # print("Total carga a transportar cluster 6:", Qt6)

    Qt7 = data7['Q'].sum()
    # print("Total carga a transportar cluster 7:", Qt7)

    # Cantidad de carga a transportar por ruta.
    QR0 = Qt1n
    QR1 = Qt0
    QR2 = Qt1

    # In[30]:

    datak0 = dn

    # # Cluster 1 K-means a Optics

    # In[31]:

    # Filtrar base de datos con cluster "n".
    dataC1 = datanew[(datanew['ClusterK'] == 1)]
    dataDepot = datanew[(datanew['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, dataC1]
    concat1 = pd.concat(concat, sort='False', ignore_index='True')
    concat1.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    data1 = concat1
    # data1.head()

    # In[32]:

    # print("Cantidad de clientes en el cluster:", len(data1))

    # Selección y eliminación de columnas irrelevantes en para el analisis
    drop_features = ['Lat', 'Lon', 'ClusterK', 'ID', 'Client_Depot', 'City']
    X = data1.drop(drop_features, axis=1)

    # Filtro de los valores perdidos (si los hay)
    X.fillna(method='ffill', inplace=False)

    # X.head()

    # Escalar los datos para llevar todos los atributos a un nivel comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalizar los datos para que los datos sigan aproximadamente una distribución gaussiana.
    X_normalizada = normalize(X_scaled)

    # Conversión de la matriz numpy en un DataFrame de pandas
    X_normalizada = pd.DataFrame(X_normalizada)

    # Renombre de las columnas
    X_normalizada.columns = X.columns
    # X_normalizada.head()

    # Construcción del modelo de clusterización Optics para el cluster "n"
    optics_model = OPTICS(min_samples=11, xi=0.06, min_cluster_size=0.08)

    # Entrenamiento del modelo
    optics_model.fit(X_normalizada)

    # In[36]:

    # Creando una matriz numpy con números en espacios iguales hasta el rango especificado
    space = np.arange(len(X_normalizada))

    # Almacenamiento de la distancia de accesibilidad de cada punto
    reachability = optics_model.reachability_[optics_model.ordering_]

    # Almacenamiento de las etiquetas de grupo de cada punto
    labels = optics_model.labels_[optics_model.ordering_]
    # labels

    # Incluir a la base de datos la columna del cluster asignado.
    ##  ACA LO CONVERTI EN 2 PARA QUE FUNCIONARA
    X1 = data2
    X1['ClusterO'] = labels
    dn = X1
    # dn.head()

    # Filtrar base de datos con cluster "n".
    data1n = dn[(dn['ClusterO'] == -1)]
    d1n = data1n  # .sort_values(by = 'Q', ascending = False)

    data0 = dn[(dn['ClusterO'] == 0)]
    d0 = data0  # .sort_values(by = 'Q', ascending = False)

    data1 = dn[(dn['ClusterO'] == 1)]
    d1 = data1  # .sort_values(by = 'Q', ascending = False)

    data2 = dn[(dn['ClusterO'] == 2)]
    d2 = data2  # .sort_values(by = 'Q', ascending = False)

    data3 = dn[(dn['ClusterO'] == 3)]
    d3 = data3  # .sort_values(by = 'Q', ascending = False)

    data4 = dn[(dn['ClusterO'] == 4)]
    d4 = data4  # .sort_values(by = 'Q', ascending = False)

    data5 = dn[(dn['ClusterO'] == 5)]
    d5 = data5  # .sort_values(by = 'Q', ascending = False)

    data6 = dn[(dn['ClusterO'] == 6)]
    d6 = data6  # .sort_values(by = 'Q', ascending = False)

    data7 = dn[(dn['ClusterO'] == 7)]
    d7 = data7  # .sort_values(by = 'Q', ascending = False)

    # In[42]:

    # Carga total a transportar en el cluster "n".
    Qt1n = data1n['Q'].sum()
    # print("Total carga a transportar cluster 1n:", Qt1n)

    Qt0 = data0['Q'].sum()
    # print("Total carga a transportar cluster 0:", Qt0)

    Qt1 = data1['Q'].sum()
    # print("Total carga a transportar cluster 1:", Qt1)

    Qt2 = data2['Q'].sum()
    # print("Total carga a transportar cluster 2:", Qt2)

    Qt3 = data3['Q'].sum()
    # print("Total carga a transportar cluster 3:", Qt3)

    Qt4 = data4['Q'].sum()
    # print("Total carga a transportar cluster 4:", Qt4)

    Qt5 = data5['Q'].sum()
    # print("Total carga a transportar cluster 5:", Qt5)

    Qt6 = data6['Q'].sum()
    # print("Total carga a transportar cluster 6:", Qt6)

    Qt7 = data7['Q'].sum()
    # print("Total carga a transportar cluster 7:", Qt7)

    # In[43]:

    # Cantidad de carga a transportar por ruta.
    QR3 = Qt1n
    QR4 = Qt0
    QR5 = Qt1

    # In[44]:

    datak1 = dn

    # # Cluster 2 K-means a Optics

    # In[45]:

    # Filtrar base de datos con cluster "n".
    dataC2 = datanew[(datanew['ClusterK'] == 2)]
    dataDepot = datanew[(datanew['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, dataC2]
    concat2 = pd.concat(concat, sort='False', ignore_index='True')
    concat2.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    data2 = concat2
    # data2.head()

    # In[46]:

    # print("Cantidad de clientes en el cluster:", len(data2))

    # In[47]:

    # Selección y eliminación de columnas irrelevantes en para el analisis
    drop_features = ['Lat', 'Lon', 'ClusterK', 'ID', 'Client_Depot', 'City']
    X = data2.drop(drop_features, axis=1)

    # Filtro de los valores perdidos (si los hay)
    X.fillna(method='ffill', inplace=False)

    # X.head()

    # In[48]:

    # Escalar los datos para llevar todos los atributos a un nivel comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalizar los datos para que los datos sigan aproximadamente una distribución gaussiana.
    X_normalizada = normalize(X_scaled)

    # Conversión de la matriz numpy en un DataFrame de pandas
    X_normalizada = pd.DataFrame(X_normalizada)

    # Renombre de las columnas
    X_normalizada.columns = X.columns
    # X_normalizada.head()

    # In[49]:

    # Construcción del modelo de clusterización Optics para el cluster "n"
    optics_model = OPTICS(min_samples=11, eps=0.06, xi=0.08)

    # Entrenamiento del modelo
    optics_model.fit(X_normalizada)

    # In[50]:

    # Creando una matriz numpy con números en espacios iguales hasta el rango especificado
    space = np.arange(len(X_normalizada))

    # Almacenamiento de la distancia de accesibilidad de cada punto
    reachability = optics_model.reachability_[optics_model.ordering_]

    # Almacenamiento de las etiquetas de grupo de cada punto
    labels = optics_model.labels_[optics_model.ordering_]
    # labels

    # In[51]:

    X1 = data2
    X1['ClusterO'] = labels
    dn = X1
    # dn.head()

    # Filtrar base de datos con cluster "n".
    data1n = dn[(dn['ClusterO'] == -1)]
    d1n = data1n  # .sort_values(by = 'Q', ascending = False)

    data0 = dn[(dn['ClusterO'] == 0)]
    d0 = data0  # .sort_values(by = 'Q', ascending = False)

    data1 = dn[(dn['ClusterO'] == 1)]
    d1 = data1  # .sort_values(by = 'Q', ascending = False)

    data2 = dn[(dn['ClusterO'] == 2)]
    d2 = data2  # .sort_values(by = 'Q', ascending = False)

    data3 = dn[(dn['ClusterO'] == 3)]
    d3 = data3  # .sort_values(by = 'Q', ascending = False)

    data4 = dn[(dn['ClusterO'] == 4)]
    d4 = data4  # .sort_values(by = 'Q', ascending = False)

    data5 = dn[(dn['ClusterO'] == 5)]
    d5 = data5  # .sort_values(by = 'Q', ascending = False)

    data6 = dn[(dn['ClusterO'] == 6)]
    d6 = data6  # .sort_values(by = 'Q', ascending = False)

    data7 = dn[(dn['ClusterO'] == 7)]
    d7 = data7  # .sort_values(by = 'Q', ascending = False)

    # In[56]:

    # Carga total a transportar en el cluster "n".
    Qt1n = data1n['Q'].sum()
    # print("Total carga a transportar cluster 1n:", Qt1n)

    Qt0 = data0['Q'].sum()
    # print("Total carga a transportar cluster 0:", Qt0)

    Qt1 = data1['Q'].sum()
    # print("Total carga a transportar cluster 1:", Qt1)

    Qt2 = data2['Q'].sum()
    # print("Total carga a transportar cluster 2:", Qt2)

    Qt3 = data3['Q'].sum()
    # print("Total carga a transportar cluster 3:", Qt3)

    Qt4 = data4['Q'].sum()
    # print("Total carga a transportar cluster 4:", Qt4)

    Qt5 = data5['Q'].sum()
    # print("Total carga a transportar cluster 5:", Qt5)

    Qt6 = data6['Q'].sum()
    # print("Total carga a transportar cluster 6:", Qt6)

    Qt7 = data7['Q'].sum()
    # print("Total carga a transportar cluster 7:", Qt7)

    # In[57]:

    # Cantidad de carga a transportar por ruta.
    # QR6 = Qt1n
    # QR7 = Qt0
    # QR8= Qt1

    # In[58]:

    datak2 = dn

    # Conversion de longitud y latitud en radianes para obtener un escalar basado en la formula del
    # semiverseno para el cálculo de la matriz de distancias.

    # Conversión de latitud y longitud a radianes
    data['Lat_[y]'] = np.radians(data['Lat_[y]'])
    data['Lon_[x]'] = np.radians(data['Lon_[x]'])

    # Formula del semiverseno, calculo entre dos puntos de una esfera.
    dist = DistanceMetric.get_metric('haversine')

    # Matriz de latitud y longitud multiplicada por el radio de la tierra en km.
    EM = dist.pairwise(data[['Lat_[y]', 'Lon_[x]']].to_numpy()) * 6371.01  # Radio de la tierra en km.

    # Mostrar matriz de distancias.
    ExportMatriz = pd.DataFrame(EM, columns=data.Client_Depot.unique(), index=data.Client_Depot.unique())
    # ExportMatriz

    # Filtrar base de datos con cluster "n".
    DtF = datak0[(datak0['ClusterO'] == -1)]
    dataDepot = datak0[(datak0['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    # DF.head()

    # In[62]:

    # len(DF)

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    # noinspection PyStatementEffect
    nodos

    # In[64]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Matriz de distancia en un diccionario.
    # Se obtiene un escalar a partir de la matriz de distancias con la función del semiverseno, con ello
    # se obtiene un cálculo de distancia del 98.9% cercano a la distancia real entre nodos (distancia
    # lineal, no contempla disttribución de calles ni avenidas, p.ej).
    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + distancia[(i, j)]

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    y0 = y
    sol0 = sol
    d0 = d
    distancia0 = distancia
    tiempo0 = time_final - time_inicio
    DF0 = DF

    # Filtrar base de datos con cluster "n".
    DtF = datak0[(datak0['ClusterO'] == 0)]
    dataDepot = datak0[(datak0['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    # DF.head()

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    # nodos

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + distancia[(i, j)]

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    # Parámetros gráfica general de rutas
    # Ruta 1
    x1 = x
    y1 = y
    sol1 = sol
    d1 = d
    distancia1 = distancia
    tiempo1 = time_final - time_inicio
    DF1 = DF

    # # Cluster K-means 0, Optics 1

    # In[71]:

    # Filtrar base de datos con cluster "n".
    DtF = datak0[(datak0['ClusterO'] == 1)]
    dataDepot = datak0[(datak0['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    DF.head()

    # In[72]:

    len(DF)

    # In[73]:

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    # noinspection PyStatementEffect
    nodos

    # In[74]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Matriz de distancia en un diccionario.
    # Se obtiene un escalar a partir de la matriz de distancias con la función del semiverseno, con ello
    # se obtiene un cálculo de distancia del 98.9% cercano a la distancia real entre nodos.
    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + distancia[(i, j)]

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    from platform import python_version

    print(python_version())

    # Parámetros gráfica general de rutas
    # Ruta 2
    x2 = x
    y2 = y
    sol2 = sol
    d2 = d
    distancia2 = distancia
    tiempo2 = time_final - time_inicio
    DF2 = DF

    # # Cluster K-means 1, Optics -1

    # In[76]:

    # Filtrar base de datos con cluster "n".
    DtF = datak1[(datak1['ClusterO'] == -1)]
    dataDepot = datak1[(datak1['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    DF.head()

    # In[77]:

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    nodos

    # In[78]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Matriz de distancia en un diccionario.
    # Se obtiene un escalar a partir de la matriz de distancias con la función del semiverseno, con ello
    # se obtiene un cálculo de distancia del 98.9% cercano a la distancia real entre nodos.
    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + 1

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    # Parámetros gráfica general de rutas
    # Ruta 3
    x3 = x
    y3 = y
    sol3 = sol
    d3 = d
    distancia3 = distancia
    tiempo3 = time_final - time_inicio
    DF3 = DF

    # # Cluster K-means 1, Optics 0

    # In[80]:

    # Filtrar base de datos con cluster "n".
    DtF = datak1[(datak1['ClusterO'] == 0)]
    dataDepot = datak1[(datak1['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    DF.head()

    # In[81]:

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    nodos

    # In[82]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + +1

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    # Parámetros gráfica general de rutas
    # Ruta 4
    x4 = x
    y4 = y
    sol4 = sol
    d4 = d
    distancia4 = distancia
    tiempo4 = time_final - time_inicio
    DF4 = DF

    # # Cluster K-means 1, Optics 1

    # In[84]:

    # Filtrar base de datos con cluster "n".
    DtF = datak1[(datak1['ClusterO'] == 1)]
    dataDepot = datak1[(datak1['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    DF.head()

    # In[85]:

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    nodos

    # In[86]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Matriz de distancia en un diccionario.
    # Se obtiene un escalar a partir de la matriz de distancias con la función del semiverseno, con ello
    # se obtiene un cálculo de distancia del 98.9% cercano a la distancia real entre nodos.
    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + 1

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    # Parámetros gráfica general de rutas
    # Ruta 5
    x5 = x
    y5 = y
    sol5 = sol
    d5 = d
    distancia5 = distancia
    tiempo5 = time_final - time_inicio
    DF5 = DF

    # # Cluster K-means 2, Optics -1

    # In[88]:

    # Filtrar base de datos con cluster "n".
    DtF = datak2[(datak2['ClusterO'] == -1)]
    dataDepot = datak2[(datak2['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    DF.head()

    # In[89]:

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    nodos

    # In[90]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Matriz de distancia en un diccionario.
    # Se obtiene un escalar a partir de la matriz de distancias con la función del semiverseno, con ello
    # se obtiene un cálculo de distancia del 98.9% cercano a la distancia real entre nodos.
    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + 1

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    # Parámetros gráfica general de rutas
    # Ruta 6
    x6 = x
    y6 = y
    sol6 = sol
    d6 = d
    distancia6 = distancia
    tiempo6 = time_final - time_inicio
    DF6 = DF

    # # Cluster K-means 2, Optics 0

    # In[92]:

    # Filtrar base de datos con cluster "n".
    DtF = datak2[(datak2['ClusterO'] == 0)]
    dataDepot = datak2[(datak2['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.
    DF = concatt
    DF.head()

    # In[93]:

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    nodos

    # In[94]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Matriz de distancia en un diccionario.
    # Se obtiene un escalar a partir de la matriz de distancias con la función del semiverseno, con ello
    # se obtiene un cálculo de distancia del 98.9% cercano a la distancia real entre nodos.
    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + 1

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    # Parámetros gráfica general de rutas
    # Ruta 7
    x7 = x
    y7 = y
    sol7 = sol
    d7 = d
    distancia7 = distancia
    tiempo7 = time_final - time_inicio
    DF7 = DF

    # # Cluster K-means 2, Optics 1

    # In[96]:

    # Filtrar base de datos con cluster "n".
    DtF = datak2[(datak2['ClusterO'] == 1)]
    dataDepot = datak2[(datak2['Client_Depot'] == 'Depot')]

    # Unir base de datos de los clusters y el depósito.
    concat = [dataDepot, DtF]
    concatt = pd.concat(concat, sort='False', ignore_index='True')
    concatt.drop_duplicates('Client_Depot', inplace=True)  # Eliminar duplicado del depot en caso de que el
    # cluster incluya el depósito.

    DF = concatt
    DF.head()

    # In[97]:

    # Extraer cantidad de nodos.
    Client_Depot = DF.loc[:, 'Client_Depot']
    IDMatriz = np.array(Client_Depot)
    nodos = IDMatriz

    # Cantidad de nodos.
    nodos = len(nodos)
    nodos

    # In[98]:

    # Creación de la red de datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = DF.loc[:, 'Lon']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = DF.loc[:, 'Lat']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Matriz de distancia en un diccionario.
    # Se obtiene un escalar a partir de la matriz de distancias con la función del semiverseno, con ello
    # se obtiene un cálculo de distancia del 98.9% cercano a la distancia real entre nodos.
    distancia = {(i, j): np.hypot(x[i] - x[j], y[i] - y[j]) * 111.138746329478 for i, j in arcos}

    # Función Nearest Neighbor.
    def Nearest_neighbor(starting_node, clientes, distancia):
        NN = [starting_node]
        n = len(clientes)

        while len(NN) < n:
            k = NN[-1]
            nn = {(k, j): distancia[(k, j)] for j in clientes if j != k and j not in NN}
            nn.items()
            new = min(nn.items(), key=lambda x: x[1])
            NN.append(new[0][1])

        NN.append(starting_node)

        return NN

    # Calculo de la distancia total de la ruta.
    def total_distancia(lista, distancia):

        dist = 0
        for n in range(len(lista) - 1):
            i = lista[n]
            j = lista[n + 1]
            dist = dist + 1

        return dist

    # Solución inicial con Nearest Neighbor.
    # Para iniciar una solución con local search 2 opt, se necesita de una solución previa.
    starting_node = 0
    NN = Nearest_neighbor(starting_node, clientes, distancia)

    # Distancia total de la solución Nearest Neighbor.
    d = total_distancia(NN, distancia)

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # Implementación Local search 2-opt.
    # Parte 1, buscar la mejor combinación.
    def LS_2opt(NN, distancia):

        min_change = 0

        for i in range(len(NN) - 2):
            for j in range(i + 2, len(NN) - 1):

                costo_actual = distancia[(NN[i], NN[i + 1])] + distancia[(NN[j], NN[j + 1])]
                costo_nuevo = distancia[(NN[i], NN[j])] + distancia[(NN[i + 1], NN[j + 1])]
                change = costo_nuevo - costo_actual

                if change < min_change:
                    min_change = change
                    min_i = i
                    min_j = j

        if min_change < 0:
            NN[min_i + 1: min_j + 1] = NN[min_i + 1: min_j + 1][::-1]

        return NN

    # Parte 2, hace los cambios, eliminar cruces.
    time_inicio = time.time()
    sol = NN.copy()

    cambio = 1
    count = 0

    while cambio != 0:
        count = count + 1
        inicial = total_distancia(sol, distancia)
        sol = LS_2opt(sol, distancia).copy()
        final = total_distancia(sol, distancia)

        cambio = np.abs(final - inicial)

    time_final = time.time()

    # Parámetros gráfica general de rutas
    # Ruta 8
    x8 = x
    y8 = y
    sol8 = sol
    d8 = d
    distancia8 = distancia
    tiempo8 = time_final - time_inicio
    DF8 = DF

    # Importar base de datos.
    data = pd.read_csv(csv_filename)

    # Extraer lista de nodos.
    nodolist = pd.read_csv(csv_filename, usecols=['ID', 'Client_Depot'])

    # Cantidad de nodos.
    nodos = len(nodolist)

    # Creación de gráfica y datos.
    n = nodos
    clientes = [i for i in range(n)]
    arcos = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Extraer columna de longitud.
    Longitud = data.loc[:, 'Lon_[x]']
    Lonmatriz = np.array(Longitud)
    x = Lonmatriz

    # Extraer columna de latitud.
    Latitud = data.loc[:, 'Lat_[y]']
    Latmatriz = np.array(Latitud)
    y = Latmatriz

    # Creación de coordenadas.
    x = Lonmatriz
    y = Latmatriz

    # Extraer depósito.
    depotmatriz = np.column_stack((Lonmatriz, Latmatriz))
    depot = depotmatriz[:1]

    # Latitud y longitud del depósito.
    depotlon = depot[:, 0]
    depotlat = depot[:, 1]

    # arreglos finales con latitudes y longitudes

    finalsol1 = []
    for row in sol1:
        temp = [y[row], x[row]]
        finalsol1.append(temp)

    finalsol2 = []
    for row in sol2:
        temp = [y[row], x[row]]
        finalsol2.append(temp)

    finalsol3 = []
    for row in sol3:
        temp = [y[row], x[row]]
        finalsol3.append(temp)

    finalsol4 = []
    for row in sol4:
        temp = [y[row], x[row]]
        finalsol4.append(temp)

    finalsol5 = []
    for row in sol5:
        temp = [y[row], x[row]]
        finalsol5.append(temp)

    finalsol6 = []
    for row in sol5:
        temp = [y[row], x[row]]
        finalsol6.append(temp)

    finalsol7 = []
    for row in sol7:
        temp = [y[row], x[row]]
        finalsol7.append(temp)

    finalsol8 = []
    for row in sol8:
        temp = [y[row], x[row]]
        finalsol8.append(temp)

    ruta_waypts = finalsol1
    # print(ruta_waypts)

    dataframe = pd.read_csv(csv_filename)

    args = {'title': 'Trazado de rutas'}
    return render(request, 'main.html', args)


@login_required
def route_calculator(request):

    global ruta_waypts

    try:

        if len(ruta_waypts) > 8:
            waypoints = ruta_waypts[1: 9]
        else:
            waypoints = ruta_waypts

        start = str(Vehiculo.objects.get(conductor_id=request.user.id).direccion) + ' ' + \
                str(Vehiculo.objects.get(conductor_id=request.user.id).ciudad)

        end = str(Despacho.objects.filter(vehiculo__conductor=request.user.id).
                  filter(fecha_despacho=date.today()).filter(vehiculo__ciudad=Vehiculo.objects.
                                                             get(conductor_id=request.user.id).ciudad).
                  latest().pedido.cliente.direccion) \
              + ' ' + str(Vehiculo.objects.get(conductor_id=request.user.id).ciudad)

        print('start:', start)
        print('end:', end, '\n')
        print('waypoints: ', waypoints)
        print('waypoints count: ', len(waypoints))

    except(ValueError, Exception):
        messages.error(request, 'Se requiere validar la ruta nuevamente. Es posible que exista algún inconveniente, '
                                  'que no tenga vehículo asignado en el sistema  '
                                  'o que su ruta de despacho para el día de hoy aún no haya sido generada.\n'
                                  'Por favor comuniquese con el administrador del sitio.')
        return redirect('/routes/')

    args = {
        'title': 'Trazado de recorrido',

        'start': start,
        'end': end,
        'prueba2': waypoints
    }

    return render(request, 'main.html', args)
