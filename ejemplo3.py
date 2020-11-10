#DISCLAIRMER: ESTE CODIGO ES A MODO DE EJEMPLO DIDÁCTICO, NO CONTIENE CONTROL DE ERRORES, NI SOFISTICACIONES, NI MEJORAS DE
# PERFORMANCE. TODOS LOS USOS DE LIBRERIAS EXTERNAS PUEDEN SER MEJORADAS EN SU IMPLEMENTACIÓN.
# ===================================================================================
 
import matplotlib.pyplot as plt 
import numpy as np
import csv
import ee


# ARCHIVOS  A UTILIZAR
# ==================================================================================
workdir="/home/alfredo/Escritorio/desafiosAgTech2020/"
train_csv_name = workdir+"data_train_r.csv"


# ABRO LA IMAGEN RASTER DE GEE
# ==================================================================================
ee.Initialize()

S2_collection = ee.ImageCollection("COPERNICUS/S2_SR") \
                    .filterBounds(ee.Geometry.Point(-61.9055,-33.6756)) \
                    .filterDate('2020-01-01', '2020-01-31') \
                    .sort('CLOUDY_PIXEL_PERCENTAGE') \
                    .first() \
                     
S2_info = S2_collection.getInfo()['id']

imagen = ee.Image(S2_info)

# ABRO LOS PUNTOS DE ENTRENAMIENTO Y LOS DE TESTEO
# ==================================================================================
puntos_train=list()

print("Busco datos para los puntos de entrenamiento")
# Esta parte es lenta porque se busca de a un punto! Los invito a mejorarla.
with open(train_csv_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['Campania']=='19/20'):
            p = ee.Geometry.Point(float(row['Longitud']),float(row['Latitud']))
            data = imagen.select("B2","B3","B4","B8","B11","B12").reduceRegion(ee.Reducer.first(),p,10).getInfo()
            datos = np.asarray(list(data.values()))
            puntos_train.append({'lat':row['Latitud'],'lon':row['Longitud'],
                                'cultivo':row['Cultivo'],'camp':row['Campania'],
                                'datos':datos[[2,3,4,5,0,1]]}) # reordeno los datos porque GEE me entregaba primero el SWIR


# OBTENGO LOS VALORES DE LOS PIXELES
# =================================================================================
valores_pixeles_entrenamiento = np.asarray([d['datos']   for d in puntos_train])
clase_entrenamiento = [d['cultivo'] for d in puntos_train]

# GRAFICO
# =================================================================================
plt.plot(np.array(np.transpose(valores_pixeles_entrenamiento[np.array(clase_entrenamiento)=='M',:])),'r',alpha=0.3)
plt.plot(np.array(np.transpose(valores_pixeles_entrenamiento[np.array(clase_entrenamiento)=='S',:])),'g',alpha=0.3)
plt.xticks(np.arange(6),("B","G","R","NIR","SWIR1","SWIR2"))
plt.show()

