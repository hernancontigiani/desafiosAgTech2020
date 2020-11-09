import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np
import csv
from osgeo import gdal,ogr,osr
from sklearn.ensemble import RandomForestClassifier

# ARCHIVOS  A UTILIZAR
# ==================================================================================
workdir="/home/alfredo/Escritorio/desafiosAgTech2020/"
image_name = workdir+"S2_20200117_B020304081112.tif"  
train_csv_name = workdir+"data_train_r.csv"
test_csv_name = workdir+"data_test_r.csv"

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
        if (row['Campania']=='19/20'):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(float(row['Latitud']),float(row['Longitud']))
                coordTrans = osr.CoordinateTransformation(train_SR,target_SR)
                point.Transform(coordTrans)
                transf_x,transf_y=point.GetX(), point.GetY()

                px = int((transf_x - raster_gt[0]) / raster_gt[1]) #x pixel
                py = int((transf_y - raster_gt[3]) / raster_gt[5]) #y pixel

                puntos_train.append({'lat':row['Latitud'],'lon':row['Longitud'],'px':px,'py':py,'cultivo':row['Cultivo'],'camp':row['Campania']})

with open(test_csv_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['Campania']=='19/20'):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(row['Latitud']),float(row['Longitud']))
            coordTrans = osr.CoordinateTransformation(train_SR,target_SR)
            point.Transform(coordTrans)
            transf_x,transf_y=point.GetX(), point.GetY()

            px = int((transf_x - raster_gt[0]) / raster_gt[1]) #x pixel
            py = int((transf_y - raster_gt[3]) / raster_gt[5]) #y pixel

            puntos_test.append({'lat':row['Latitud'],'lon':row['Longitud'],'px':px,'py':py,'cultivo':row['Cultivo'],'camp':row['Campania']})


# OBTENGO LOS VALORES DE LOS PIXELES
# =================================================================================
valores_pixeles_entrenamiento = np.asarray([raster_dataPixel[d['py'],d['px'],:]   for d in puntos_train])
clase_entrenamiento = [d['cultivo'] for d in puntos_train]

# CORRO RANDOM FOREST
# ==================================================================================
classifier = RandomForestClassifier(n_estimators=5)
classifier.fit(valores_pixeles_entrenamiento,clase_entrenamiento)

puntos_predichos = [classifier.predict([raster_dataPixel[p['py'],p['px'],:]]) for p in puntos_test]


F,C,bands = raster_dataPixel.shape
img_reshape = np.reshape(raster_dataPixel,(F*C,bands),order='C')
puntos_predichos = np.array(classifier.predict(img_reshape))
img_clasif_num = np.zeros((F*C))
img_clasif_num[puntos_predichos=='M']=3
img_clasif_num[puntos_predichos=='S']=2
img_clasif_num[puntos_predichos=='m']=3
img_clasif_num[puntos_predichos=='s']=2
img_clasif_num[puntos_predichos=='B']=1

img_clasif_num = np.reshape(img_clasif_num,(F,C))


plt.imshow(np.array(img_clasif_num,dtype='int'),cmap='jet')
plt.colorbar()
plt.show()

