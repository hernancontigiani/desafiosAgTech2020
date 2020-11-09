import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np
import csv
#from osgeo import gdal,ogr,osr
import ee
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix

# ARCHIVOS  A UTILIZAR
# ==================================================================================
workdir="/home/alfredo/Escritorio/desafiosAgTech2020/"
# image_name = workdir+"S2_20200117_B020304081112.tif"  
train_csv_name = workdir+"data_train_r.csv"
test_csv_name = workdir+"data_test_r.csv"

# ABRO LA IMAGEN RASTER DE GEE
# ==================================================================================
ee.Initialize()

S2_collection = ee.ImageCollection("COPERNICUS/S2_SR") \
                    .filterBounds(ee.Geometry.Point(-61.9055,-33.6756)) \
                    .filterDate('2020-01-01', '2020-01-31') \
                    .sort('CLOUDY_PIXEL_PERCENTAGE') \
                    .first() \
                     
S2_info = S2_collection.getInfo()['id']

im1 = ee.Image(S2_info)

# ABRO LOS PUNTOS DE ENTRENAMIENTO Y LOS DE TESTEO
# ==================================================================================
puntos_train=list()
puntos_test=list()
print("Busco datos para los puntos de entrenamiento")
with open(train_csv_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['Campania']=='19/20'):
            p = ee.Geometry.Point(float(row['Longitud']),float(row['Latitud']))
            data = im1.select("B2","B3","B4","B8","B11","B12").reduceRegion(ee.Reducer.first(),p,10).getInfo()
            datos = np.asarray(list(data.values()))
            puntos_train.append({'lat':row['Latitud'],'lon':row['Longitud'],
                                'cultivo':row['Cultivo'],'camp':row['Campania'],
                                'datos':datos})

print("Busco datos para los puntos de testeo")
with open(test_csv_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['Campania']=='19/20'):
            p = ee.Geometry.Point(float(row['Longitud']),float(row['Latitud']))
            data = im1.select("B2","B3","B4","B8","B11","B12").reduceRegion(ee.Reducer.first(),p,10).getInfo()

            datos = np.asarray(list(data.values()))
            puntos_test.append({'lat':row['Latitud'],'lon':row['Longitud'],
                                'cultivo':row['Cultivo'],'camp':row['Campania'],
                                'datos':datos})

# OBTENGO LOS VALORES DE LOS PIXELES
# =================================================================================
valores_pixeles_entrenamiento = np.asarray([d['datos']   for d in puntos_train])
clase_entrenamiento = [d['cultivo'] for d in puntos_train]

# CORRO SVC
# ==================================================================================
SupportVectorClassModel = SVC()
SupportVectorClassModel.fit(valores_pixeles_entrenamiento,clase_entrenamiento)

puntos_predichos = SupportVectorClassModel.predict([d['datos']   for d in puntos_test])

print(puntos_predichos)


