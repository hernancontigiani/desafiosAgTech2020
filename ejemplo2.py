#DISCLAIRMER: ESTE CODIGO ES A MODO DE EJEMPLO DIDÁCTICO, NO CONTIENE CONTROL DE ERRORES, NI SOFISTICACIONES, NI MEJORAS DE
# PERFORMANCE. TODOS LOS USOS DE LIBRERIAS EXTERNAS PUEDEN SER MEJORADAS EN SU IMPLEMENTACIÓN.
# ===================================================================================

import matplotlib.pyplot as plt
import numpy as np
import csv
from osgeo import gdal,ogr,osr 
from sklearn.cluster import KMeans

# ARCHIVOS  A UTILIZAR
# ==================================================================================
workdir="/home/alfredo/Escritorio/desafiosAgTech2020/"
image_name = workdir+"S2_20200117_B020304081112.tif"  

# ABRO LA IMAGEN RASTER Y ARMO HIPERMATRIZ
# B0=BLUE, B1=GREEN, B2=RED, B3=NIR, B4=SWIR1, B5=SWIR
# ==================================================================================
raster_ds=gdal.Open(image_name) 
raster_gt=raster_ds.GetGeoTransform()
raster_dataPixel=np.zeros((raster_ds.RasterYSize,
                        raster_ds.RasterXSize,
                        raster_ds.RasterCount))

for i in range(raster_ds.RasterCount):
    banddataraster = raster_ds.GetRasterBand(i+1)
    raster_dataPixel[:,:,i]= banddataraster.ReadAsArray(0,0, raster_ds.RasterXSize, raster_ds.RasterYSize).astype(np.float)

# CALCULO KMEANS
# ==================================================================================
raster_reshape = np.reshape(raster_dataPixel,
                        (raster_ds.RasterYSize*raster_ds.RasterXSize,raster_ds.RasterCount),
                        order='C')
print("Empiezo la clasificación!!!")
kmeans = KMeans(n_clusters=5, random_state=0).fit(raster_reshape)
img_clasif = np.reshape(kmeans.labels_,(raster_ds.RasterYSize,raster_ds.RasterXSize))

# MUESTRO LAS IMGS
# ==================================================================================
plt.subplot(2,2,1) , plt.imshow(raster_dataPixel[:,:,0:3]*.0001*4), plt.title("RGB")
plt.subplot(2,2,2) , plt.imshow(raster_dataPixel[:,:,[5,4,3]]*.0001*3) ,plt.title("SWIR-NIR-R")
plt.subplot(2,2,3) , plt.imshow(img_clasif, cmap='jet'),  plt.title("Clasificacion Kmeans")

plt.show()
