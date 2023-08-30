import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from sklearn.metrics.cluster import entropy
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import column_or_1d

fLIR=r"G:\Unidades compartidas\FLIR\PruebaMensual"

def recorteyfondo( imagen):
    original = cv2.imread(fLIR+imagen)
    #original = cv2.imread(imagen) 
    l,a,rgbnum=original.shape
    print(a)
  
    if l==1440:
        images = original[103:1000,0 :1080]
    images = original[103:1000,0 :1080]
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    ret,th=cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
    contornos,jerarquia=cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tamaño=len(contornos)
    
    cnt=contornos[-1]
    print("El numero de contornos es " , tamaño)
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

    #rArriba= cY-160
    rAbajo=cY+350
    rArriba= cY-220
    #rAbajo=cY+290
    der=abs(a-cX);
    dif=abs(cX-der)
    if cX>der:
        recorte= images[rArriba:rAbajo, dif:1080]
    else:
        recorte= images[rArriba:rAbajo, 0:1080-dif]

    l1,a1,rgbnum1=recorte.shape
    print("l", l1, "ancho", a1)

    
    grayR = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    retR,thR=cv2.threshold(grayR,90,255,cv2.THRESH_BINARY)
    contornosR, jerarquia=cv2.findContours(thR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Leyó y recortó IMG")
    #print("contorno es " , len(contornosR))
    #print(contornosR)
    #print("jerarquia es " , len(jerarquia))
    #envoltura = cv2.convexHull(thR[-1])
    #print(envoltura)

    #print(len(img))
    new_img=cv2.bitwise_and(recorte,recorte,mask=thR.astype(np.uint8))
    #cv2.imshow("nueva imagen sin fondo" , new_img)
    #cv2.waitKey(0)
    #cv2.imshow("nueva imagen gray" , grayR)
    #cv2.waitKey(0)
    newRGB=cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    #plt.figure()
    #plt.imshow(newRGB)
    #plt.show()
    #plt.imshow(grayR)
    #plt.show()
    return newRGB, grayR


def puntosMax(imgRGB, grayImg, d,h):

    max=np.amax(grayImg)
    print("el valor máximo en escala de grises de esta imágen es: ", max)
    for num in range(d,h):
        s=max-num
        ar,i = np.where(grayImg==s)
        
        c=np.vstack((ar,i))
        m=np.transpose(c)
        #print("el valor máximo es: ",max-num)
        #print(ar, i)
        #print("el calor m es ", m) 

        filas=len(m)
        columnas=len(m[0])
        #print(filas)
        #print(columnas)
        color=3+num
        color2=num
        color3=num+6
       

        for fila in m:
            a1, a2 = fila
            #print("la fila es", a1, a2)
            #print("El valor RGB del pixel es:", imgRGB[a1][a2])

            new= cv2.circle(imgRGB,(a2,a1),1,(color2,color,color3),-1)
            graypoint= cv2.circle(grayImg,(a2,a1),2,(0,0,0),-1)
    #plt.figure(0)
    #plt.imshow(new)
    #plt.show() 
    print("El numero de pixeles que tienen los 10 valores máximos es de: ", filas)
    return new


from skimage import filters
def borde(imagen_g):
    #imagen = io.imread(img)
    #imagen_g = rgb2gray(imagen)

    # Filtros: sobel, roberts, prewitt
    filtros = [filters.sobel, filters.roberts, filters.prewitt]

    for filtro in filtros:
        # Aplicamos cada uno de los filtros
        img_fil = filtro(imagen_g)
        
        # Mostramos los resultados 
        plt.imshow(img_fil)
        plt.show()

def caracteristicas(imagenRGB):
    caracteriticas=["Des.Estandar1","Varianza1", "Covarianza1", "Media1", "Mediana1", "Mínimo1", "Máximo1", "Entropia1", "Kurtosis1", "Uniformidad1"]
    matriz=[]
    con=0
    clu_red_gray=cv2.cvtColor(imagenRGB, cv2.COLOR_RGB2GRAY) #Convert the image of the cluster in gray scale
    vect_cl_red=clu_red_gray[np.nonzero(clu_red_gray)]

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

    matriz[con][0]=stddev
    matriz[con][1]=var
    matriz[con][2]=covar
    matriz[con][3]=mean
    matriz[con][4]=med
    matriz[con][5]=min
    matriz[con][6]=max
    matriz[con][7]=entr
    matriz[con][8]=kurt
    matriz[con][9]=ske
    con+=1
    print(matriz)
    S = pd.DataFrame(matriz, columns= caracteriticas)
    S.to_csv('DATOSSINCLUSTERMariana.txt', sep=" ", 
          quoting=csv.QUOTE_NONE, escapechar=" ")
    



def cluster(imagenlista):

    I_shape=np.shape(imagenlista)
    
    #Image normalization
    #Image normalization
    norm_img=np.zeros(I_shape) #Create an empty array
    Im=cv2.normalize(imagenlista,  norm_img, 0.0, 1.0, cv2.NORM_MINMAX) #Better contrast in RGB in values between 0.0 and 1.0

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
        segmentacion=imagenlista.copy() #Make segmentation a copy of the Image matrix
        
        #Assign a [0,0,0] to each element in the matrix Im_0 that is not the cluster number 
        for k in range(imagenlista.shape[0]): 
            for l in range (imagenlista.shape[1]):
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
    return cluster1, cluster2, cluster3, cluster4
        
    #clu_red_gray=cv2.cvtColor(clu_1, cv2.COLOR_RGB2GRAY) #Convert the image of the cluster in gray scale
    #cv2.imshow('Cluster in gray', clu_red_gray)

imagenesPrueba=[ "\Mariana-12B.jpg","\Anngie-18A.jpg","\Anngie-19A.jpg","\Anngie-20A.jpg","\Anngie-2A.jpg", "\Mariana-1A.jpg","\Mariana-2A.jpg",
"\Mariana-22A.jpg","\Mariana-23A.jpg","\Mariana-24A.jpg","\Mariana-25A.jpg","\Mariana-26A.jpg","\Mariana-27A.jpg","\Mariana-28A.jpg","\Mariana-29A.jpg" ]
for imagen in imagenesPrueba:
    y,x=recorteyfondo(imagen)
    #borde(x)
    plt.figure(1) #mostrar puntos blancos, sin cluser
    plt.imshow(y)
    plt.figure(2)
    #plt.imshow(x)
    
    #a=puntosMax(y,x,0,20)
    #a=puntosMax(y,x,10,20)
    b=puntosMax(y,x,0,25)
    #plt.figure(2)
    plt.imshow(b)
    plt.show()  #mostrar puntos blancos, sin cluser

    plt.figure(1) #Cluster
    plt.imshow(y)
    c1, c2, c3, c4 = cluster(y)
    
    plt.figure(2)
    plt.imshow(c1)
    plt.figure(3)
    plt.imshow(c2)
    plt.figure(4)
    plt.imshow(c3)
    plt.figure(5)
    plt.imshow(c4)
    plt.show() #Cluster

    #histo(x)
    

    
    #a=puntosMax(y,x,30,40)
    #b=puntosMax2(y,x) 