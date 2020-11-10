import matplotlib.pyplot as plt 
import csv
from osgeo import gdal,ogr,osr
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.svm import SVC
import random

# ARCHIVOS  A UTILIZAR
# ==================================================================================
workdir="/home/alfredo/Escritorio/desafiosAgTech2020/"
image_name = workdir+"S2_20200117_B020304081112.tif"  
train_csv_name = workdir+"data_train_r.csv"

# ABRO LA IMAGEN RASTER
# ==================================================================================
raster_ds=gdal.Open(image_name) 
raster_gt=raster_ds.GetGeoTransform()
raster_dataPixel=np.zeros((raster_ds.RasterYSize,raster_ds.RasterXSize,raster_ds.RasterCount,))
for i in range(raster_ds.RasterCount):
    banddataraster = raster_ds.GetRasterBand(i+1)
    raster_dataPixel[:,:,i]= banddataraster.ReadAsArray(0,0, raster_ds.RasterXSize, raster_ds.RasterYSize).astype(np.float)

# ABRO LOS PUNTOS DE ENTRENAMIENTO Y LOS DE TESTEO
# ==================================================================================

train_SR = osr.SpatialReference()
train_SR.ImportFromEPSG(4326)
target_SR = osr.SpatialReference()
target_SR.ImportFromWkt(raster_ds.GetProjectionRef())

puntos_train=list()
puntos_test=list()

with open(train_csv_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['Campania']=='19/20') and row['Cultivo'] in ['M','S','m','s']:
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(float(row['Latitud']),float(row['Longitud']))
                coordTrans = osr.CoordinateTransformation(train_SR,target_SR)
                point.Transform(coordTrans)
                transf_x,transf_y=point.GetX(), point.GetY()

                px = int((transf_x - raster_gt[0]) / raster_gt[1]) #x pixel
                py = int((transf_y - raster_gt[3]) / raster_gt[5]) #y pixel

                puntos_train.append({'lat':row['Latitud'],'lon':row['Longitud'],'px':px,'py':py,'cultivo':row['Cultivo'],'camp':row['Campania']})


# OBTENGO LOS VALORES DE LOS PIXELES
# =================================================================================
valores_pixeles = np.asarray([raster_dataPixel[d['py'],d['px'],:]   for d in puntos_train])
clase = [d['cultivo'] for d in puntos_train]

X_train, X_test, y_train, y_test = train_test_split(valores_pixeles, clase, test_size=0.33, random_state=42)


# CORRO RANDOM FOREST, SVM y Aleatorio
# ==================================================================================
puntos_predichos_aleatorio=([random.choice(['M','S']) for i in range(len(y_test))]) 

classifier_rf = RandomForestClassifier(n_estimators=5)
classifier_rf.fit(X_train,y_train)
puntos_predichos_rf = classifier_rf.predict(X_test)

classifier_svm = SVC()
classifier_svm.fit(valores_pixeles,clase)
puntos_predichos_svm = classifier_svm.predict(X_test)


# Observo metricas
# ==================================================================================
print("---------------------------------")
print("Accuracy_score")
print("---------------------------------")
print('RANDOM',accuracy_score(y_test, puntos_predichos_aleatorio))
print('RF:',accuracy_score(y_test, puntos_predichos_rf))
print('SVM:',accuracy_score(y_test, puntos_predichos_svm))

print("---------------------------------")
print("Balanced Accuracy_score")
print("---------------------------------")
print('RANDOM',balanced_accuracy_score(y_test, puntos_predichos_aleatorio))
print('RF:',balanced_accuracy_score(y_test, puntos_predichos_rf))
print('SVM:',balanced_accuracy_score(y_test, puntos_predichos_svm))

plot_confusion_matrix(classifier_svm,X_test, y_test)
plt.title("CM SVM")
plt.show()




