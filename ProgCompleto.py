# Este programa realiza lo siguiente 
# Recorte
# segmentación
# Extracción de características
# Análisis de PCA
import re
from ast import Try
from email.mime import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from sklearn.metrics.cluster import entropy
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

magenesAn=["\Anngie-1A.jpg","\Anngie-2A.jpg","\Anngie-3A.jpg","\Anngie-4A.jpg","\Anngie-5A.jpg","\Anngie-6A.jpg","\Anngie-7A.jpg","\Anngie-8A.jpg","\Anngie-9A.jpg","\Anngie-10A.jpg",
"\Anngie-11A.jpg","\Anngie-12A.jpg","\Anngie-13A.jpg","\Anngie-14A.jpg","\Anngie-15A.jpg","\Anngie-16A.jpg","\Anngie-17A.jpg","\Anngie-18A.jpg","\Anngie-19A.jpg","\Anngie-20A.jpg",
"\Anngie-21A.jpg","\Anngie-22A.jpg","\Anngie-23A.jpg","\Anngie-24A.jpg","\Anngie-25A.jpg","\Anngie-26A.jpg","\Anngie-27A.jpg","\Anngie-28A.jpg","\Anngie-29A.jpg","\Anngie-30A.jpg" ]
imagenesMa=["\Mariana-1A.jpg","\Mariana-2A.jpg","\Mariana-3A.jpg","\Mariana-4A.jpg","\Mariana-5A.jpg","\Mariana-6A.jpg","\Mariana-7A.jpg","\Mariana-8A.jpg","\Mariana-9A.jpg","\Mariana-10A.jpg",
"\Mariana-11A.jpg","\Mariana-12A.jpg","\Mariana-13A.jpg","\Mariana-14A.jpg","\Mariana-15A.jpg","\Mariana-16A.jpg","\Mariana-17A.jpg","\Mariana-18A.jpg","\Mariana-19A.jpg","\Mariana-20A.jpg",
"\Mariana-21A.jpg","\Mariana-22A.jpg","\Mariana-23A.jpg","\Mariana-24A.jpg","\Mariana-25A.jpg","\Mariana-26A.jpg","\Mariana-27A.jpg","\Mariana-28A.jpg","\Mariana-29A.jpg","\Mariana-30A.jpg"]
imagenes11M=["\Anngie-15A.jpg","\Anngie-12A.jpg","\Birads2-A.jpg","\Camilo-A.jpg","\Edinson-A.jpg","\FabioE5-A.jpg","\Fibrocistica-A.jpg","\Julio-A.jpg","\Mariana-11A.jpg","\Masa_Antes-A.jpg","\Mastectomia-A.jpg","\ormal_2-A.jpg","\ormal_3-A.jpg","\ormal_4-A.jpg", "\ormal_6-A.jpg", "\ormal_7-A.jpg" ]
ImgeDef=[ "\Mariana.jpg","\Anngie.jpg", "\Alejandra.jpg", "\Consuelo.jpg", "\Alejandra.jpg", "\SamiraAlejandra.jpg", "\Pilar.jpg", "\Fabio.jpg"]

imgMarAnn=["\Anngie-1A.jpg","\Anngie-2A.jpg","\Anngie-3A.jpg","\Anngie-4A.jpg","\Anngie-5A.jpg","\Anngie-6A.jpg","\Anngie-7A.jpg","\Anngie-8A.jpg","\Anngie-9A.jpg","\Anngie-10A.jpg",
"\Anngie-11A.jpg","\Anngie-12A.jpg","\Anngie-13A.jpg","\Anngie-14A.jpg","\Anngie-15A.jpg","\Anngie-16A.jpg","\Anngie-17A.jpg","\Anngie-18A.jpg","\Anngie-19A.jpg","\Anngie-20A.jpg",
"\Anngie-21A.jpg","\Anngie-22A.jpg","\Anngie-23A.jpg","\Anngie-24A.jpg","\Anngie-25A.jpg","\Anngie-26A.jpg","\Anngie-27A.jpg","\Anngie-28A.jpg","\Anngie-29A.jpg","\Anngie-30A.jpg" ,"\Mariana-1A.jpg","\Mariana-2A.jpg","\Mariana-3A.jpg","\Mariana-4A.jpg","\Mariana-5A.jpg","\Mariana-6A.jpg","\Mariana-7A.jpg","\Mariana-8A.jpg","\Mariana-9A.jpg","\Mariana-10A.jpg",
"\Mariana-11A.jpg","\Mariana-12A.jpg","\Mariana-13A.jpg","\Mariana-14A.jpg","\Mariana-15A.jpg","\Mariana-16A.jpg","\Mariana-17A.jpg","\Mariana-18A.jpg","\Mariana-19A.jpg","\Mariana-20A.jpg",
"\Mariana-21A.jpg","\Mariana-22A.jpg","\Mariana-23A.jpg","\Mariana-24A.jpg","\Mariana-25A.jpg","\Mariana-26A.jpg","\Mariana-27A.jpg","\Mariana-28A.jpg","\Mariana-29A.jpg","\Mariana-30A.jpg"]

imgPrueba=["\Anngie-1A.jpg","\Anngie-2A.jpg","\Anngie-3A.jpg"]

caracteriticas=["Des.EstandarRojo","VarianzaRojo", "CovarianzaRojo", "MediaRojo", "MedianaRojo", "MínimoRojo", "MáximoRojo", "EntropiaRojo", "KurtosisRojo", "UniformidadRojo", "Des.EstandarBlanco","VarianzaBlanco", "CovarianzaBlanco", "MediaBlanco", "MedianaBlanco", "MínimoBlanco", "MáximoBlanco", "EntropiaBlanco", "KurtosisBlanco", "UniformidadBlanco", "Des.EstandarAmarillo","VarianzaAmarillo", "CovarianzaAmarillo", "MediaAmarillo", "MedianaAmarillo", "MínimoAmarillo", "MáximoAmarillo", "EntropiaAmarillo", "KurtosisAmarillo", "UniformidadAmarillo","etiqueta"]
#fLIR=r"G:\Unidades compartidas\FLIR\PruebaMensual"
fLIR=r"G:\Unidades compartidas\FLIR\PruebaMensual" 
matriz=[]
columnas=len(caracteriticas)
filas= len(imagenesMa)
for i in range(filas):
    matriz.append([0]*columnas)

con=0
etiqt=[]
NumE=[]
q=200
r=310


for imagen in imagenesMa: 
    original = cv2.imread(fLIR+imagen)    
    try:

        l,a,rgbnum=original.shape
        print("Leyó imagen correctamente")
        guion= imagen.find('-')
        etiq= imagen[1:guion]
        numEtiq=imagen[guion+1:guion+3]
        if 'A' in numEtiq:
            numEtiq= numEtiq[0]
        print(numEtiq)

        # inicia el recorte   
        if l==1440:
            images = original[103:1000,0 :1080]
        images = original
        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        ret,th=cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
        contornos,jerarquia=cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tamaño=len(contornos)
        
        cnt=contornos[-1]
        M=cv2.moments(cnt) #momento para el último objeto (objeto principal para los termogramas)
        if M["m00"] ==0.0:
            cnt=contornos[-2]
            M=cv2.moments(cnt)
        if M["m00"]== 0.0:
            cnt =contornos[-3]
            M=cv2.moments(cnt)
        if M["m00"]== 0.0:
            cnt =contornos[-4]
            M=cv2.moments(cnt)
        if M["m00"]== 0.0:
            cnt =contornos[-5]
            M=cv2.moments(cnt)
        if M["m00"]== 0.0:
            cnt =contornos[-6]
            M=cv2.moments(cnt)
        #breakpoint()
        cX=int(M["m10"]/M["m00"]); cY=int(M["m01"]/M["m00"])
        area= cv2.contourArea(cnt)
        perimetro= cv2.arcLength(cnt,False)
        #cv2.putText(images," x="+str(cX)+", y="+str(cY),(cX,cY),1,1,(0,0,0),2)
        #cv2.circle(images,(cX,cY),5,(0,0,0),-1)
        # Recorte funciona bien con imágenes frontales 
        
        rArriba= cY-q
        print("valor arriba")
        print(rArriba)
        rAbajo=cY+r
        print("valor abajo")
        print(rAbajo)
        if rArriba < 0:
            rArriba=0
            rAbajo=510

        if rAbajo > 897:
            rAbajo=897
            rArriba=387

        recorte= images[rArriba:rAbajo, 0:1080]

        plt.imshow(recorte)
        plt.show()
        #Preguntar al usuario si el recorte fue correcto 
        print("El recorte es correcto, poner s para SI y n para NO")
        pregunta= input()
        varA= 0
        while pregunta == "n":
            print("Debe recortar hacia arriba (poner u), debe hacer el recorte abajo (poner d)")
            modificar= input()
            varA= varA + 20
            if modificar == "u":
                rArriba= rArriba-varA 
                rAbajo= rAbajo-varA
                if rArriba>= 0 & rAbajo <= 897:
                    recorte= images[rArriba:rAbajo, 0:1080]
                else: 
                    ("No es posible recortar más arriba")
            else: 
                rArriba= rArriba+varA 
                rAbajo=rAbajo+varA
                if rArriba>= 0 & rAbajo <= 897:
                    recorte= images[rArriba:rAbajo, 0:1080]
                else: 
                    ("No es posible recortar más hacia abajo")
            
            
            plt.imshow(recorte)
            plt.show()

            print("El recorte es correcto, poner s para SI y n para NO")
            pregunta= input()

        q=cY-rArriba
        r=rAbajo-cY
        #der=abs(a-cX);
        #dif=abs(cX-der)
        #if cX>der:
        #   recorte= images[rArriba:rAbajo, dif:1080]
        #else:
        #   recorte= images[rArriba:rAbajo, 0:1080-dif]
            
        grayR = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        retR,thR=cv2.threshold(grayR,90,255,cv2.THRESH_BINARY)
        contornosR, jerarquia=cv2.findContours(thR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_img=cv2.bitwise_and(recorte,recorte,mask=thR.astype(np.uint8))
        newRGB=cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        #Hasta acá llega el recorte. El retorno es new_img
        # Inicia la segmentación 

        I_shape=np.shape(newRGB)
        
        #Image normalization
        #Image normalization
        norm_img=np.zeros(I_shape) #Create an empty array
        Im=cv2.normalize(newRGB,  norm_img, 0.0, 1.0, cv2.NORM_MINMAX) #Better contrast in RGB in values between 0.0 and 1.0

        #Take each layer from the matrix Im
        Lay1=Im[:,:,0]
        Lay2=Im[:,:,1]
        Lay3=Im[:,:,2]
        # Use np.concatenate to make each matrix a row vector
        Lay1_c=np.concatenate(Lay1)
        Lay2_c=np.concatenate(Lay2)
        Lay3_c=np.concatenate(Lay3)

        #Make a new matrix with the previous vectors
        color=[Lay1_c,Lay2_c,Lay3_c]
        color_t=np.transpose(color) #Transpose
        sample_test=color_t 

        #Training Matrix
        color_name=["Rojo","Azul","Blanco","Amarillo"] #Nunca lo utilicé 
        train_color=[[1,0,0],[0,0,1],[1,1,1],[1,1,0]] #Vectorial form of each color
        number_color=[1,2,3,4] #Number of each color 
        train_number=np.transpose(number_color)

        #KNN Classificator
        knn_class=KNeighborsClassifier(n_neighbors=1, metric='euclidean') #Apply the classificator with 1 neighbor and the euclidean metric
        knn_fit=knn_class.fit(train_color, train_number) #Uses the classificator with the training vectors 
        prediction=knn_fit.predict(sample_test) #Predicts the value of each pixel in matrix sample with the knn model
        Im_prediction=np.reshape(prediction,[I_shape[0],I_shape[1]]) #Reshape the matrix with the size of the original one

        #Plot the matrix as a color map to look at the cluster division
        #colors_map=matplotlib.colors.ListedColormap(("black","pink","blue","yellow"),name="colors")
        #plt.imshow(Im_prediction,cmap=colors_map)
        #plt.show()

        #Get each cluster 
        cl_n=[1,2,3,4] #Cluster number

        for a in cl_n: 
            Im_0=Im_prediction.copy()    
            Im_0[Im_0 != cl_n[a-1]]=0 #Assign a value of 0 to each pixel that is not the cluster number
            segmentacion=newRGB.copy() #Make segmentation a copy of the Image matrix
            
            #Assign a [0,0,0] to each element in the matrix Im_0 that is not the cluster number 
            for k in range(newRGB.shape[0]): 
                for l in range (newRGB.shape[1]):
                    if Im_0[k,l] != cl_n[a-1]:
                        segmentacion[k,l,:]=[0,0,0]
                        
            if a==1:
                cluster1= segmentacion.copy()
            elif a==2:
                cluster2= segmentacion.copy()
            elif a==3:
                cluster3= segmentacion.copy()
            else:
                cluster4= segmentacion.copy()
        plt.figure(1)
        plt.imshow(cluster1)
        plt.figure(2)
        plt.imshow(cluster2)
        plt.figure(3)
        plt.imshow(cluster3)
        plt.figure(4)
        plt.imshow(cluster4)
        plt.show() #Cluster

        #Fin segmentación
        # Inicia la extracción de características
        #Hace falta poder decirle al programa que me saque características por cada cluster
        # Obtener características manualmente 
        clusters= [cluster1, cluster3, cluster4]
        valor=0
        au=0
        for ii,cls in enumerate (clusters):
            clu_gray=cv2.cvtColor(cls, cv2.COLOR_RGB2GRAY) #Convert the image of the cluster in gray scale 
            vect_cl_red=clu_gray[np.nonzero(clu_gray)]
            stddev=np.std(vect_cl_red) #Standard deviation --> Contrast 
            var=np.var(vect_cl_red) #Variance
            covar=np.cov(vect_cl_red) #Covariance
            mean=np.mean(vect_cl_red) #Mean --> Shine
            med=np.median(vect_cl_red) #Median 
            min=np.amin(vect_cl_red) #Minimum
            max=np.amax(vect_cl_red) #Maximum
            c_corr=np.corrcoef(vect_cl_red) #Correlation coeficient
            entr=entropy(vect_cl_red) #Entropy
            mod=stats.mode(vect_cl_red) #Mode
            kurt=kurtosis(vect_cl_red) #Kurtosis
            ske=skew(vect_cl_red) #Skewness
            carac=[stddev, var, covar, mean, med, min, max, entr, kurt, ske]
            
            for c in range(len(carac)): 
                matriz[con][c+au]=carac[c]
            au=(ii+1)*len(carac)    
            
        matriz[con][-1]=etiq
        etiqt.append(etiq)
        NumE.append(numEtiq)

       
        con+=1
    except:
        
        print("Fallo cargando la imagen llamada ", imagen )

S = pd.DataFrame(matriz, columns= caracteriticas)
print("Escriba el nombre del archivo para guardar datos")
nombreArchivo= input()
S.to_csv(nombreArchivo+'.txt', sep=" ", 
          quoting=csv.QUOTE_NONE, escapechar=" ")

# Análisis de componentes principales 

df = pd.read_csv(nombreArchivo+'.txt', sep=" ")

df.tail()
# iloc divide colunmas y filas de matriz de pandas
X = np.delete(matriz, -1, axis=1)

#X= df.drop(df.columns[[-1]], axis= 1)
y = df.iloc[:,-1].values
#annotations = df.iloc[:,0].values
annotations = NumE


#Normalización
#Aplicamos una transformación de los datos para poder aplicar las propiedades de la distribución normal

X_std = StandardScaler().fit_transform(X)

# Calculamos la matriz de covarianza
cov_mat = np.cov(X_std.T)
#Calculamos los autovalores y autovectores de la matriz y los mostramos

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#  Hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# A partir de los autovalores, calculamos la varianza explicada
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
tamaño= len(var_exp)
cum_var_exp = np.cumsum(var_exp)

# Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada

#with plt.style.context('seaborn-pastel'):
with plt.style.context('seaborn-whitegrid'):
    plt.figure(1)
    plt.bar(range(1, len(var_exp)+1), var_exp, align='center',
            label='Varianza individual explicada', color='r')
    plt.step(range(1, len(var_exp)+1), cum_var_exp, where='mid', linestyle='--',color='b', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.legend(loc='best')
    plt.tight_layout()
   
matrix_w = np.hstack((eig_pairs[0][1].reshape(len(var_exp),1),
                      eig_pairs[1][1].reshape(len(var_exp),1)))

Y = X_std.dot(matrix_w)
eet=set(etiqt)
etiqts=list(eet)
et= tuple(etiqts)
colorsPeople=['cyan', 'magenta', 'yellow', 'blue', 'green', 'red', 'black', 'pink', 'orange', 'indigo', 'lime', 'brown']
tam= len(et)
colorsPP= colorsPeople[0:tam]
cPP=tuple(colorsPP)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(2)
    for lab, col in zip(et,cPP):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)

    exis= Y[:, 0]
    ye=Y[:, 1]
    for i, label in enumerate(annotations):
       plt.text(exis[i], ye[i],label)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()