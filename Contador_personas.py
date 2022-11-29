
import numpy as np
import cv2 as cv
import Person
import time
from time import sleep
from pyo import *

try:
    log = open('log.txt',"w")
except:
    print( "No se puede abrir el archivo log")
#Contadores de entrada y salida
cnt_up   = 0
cnt_down = 0
aforo_max = 4

#Fuente de video
cap = cv.VideoCapture('Test Files/aeropuerto.mp4')


h = 480
w = 640
frameArea = h*w
areaTH = frameArea/300 #250
print( 'Area Threshold', areaTH)

#Lineas de entrada/salida
line_up =  420
line_down   = 440

up_limit =   330
down_limit = 530


line_down_color = (255,0,0)
line_up_color = (0,0,255)
pt1 =  [0, line_down];
pt2 =  [w, line_down];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#Substractor de fondo
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

#Elementos estructurantes para filtros morfoogicos
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

#Variables
font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while(cap.isOpened()):

    ret, frame = cap.read()


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Dibujamos un rectángulo en frame, para señalar el estado
    # del área en análisis (movimiento detectado o no detectado)
    cv.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
    color = (0, 255, 0)


    # Especificamos los puntos extremos del área a analizar
    area_pts = np.array([[240,320], [480,320], [620,frame.shape[0]], [50,frame.shape[0]]])
    
    
    # Con ayuda de una imagen auxiliar, determinamos el área
    # sobre la cual actuará el detector de movimiento
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv.bitwise_and(gray, gray, mask=imAux)


    
    #Aplica substraccion de fondo
    fgmask = fgbg.apply(image_area)

    #Binariazcion para eliminar sombras (color gris)
    try:
        #Opening (erode->dilate) para quitar ruido.
        mask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        #Closing (dilate -> erode) para juntar regiones blancas.
        mask =  cv.morphologyEx(mask , cv.MORPH_CLOSE, kernel)
        mask = cv.dilate(mask, None, iterations=2)

    except:
        break

    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    contours0, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > areaTH:

            
            #Falta agregar condiciones para multipersonas, salidas y entradas de pantalla.
            
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        # el objeto esta cerca de uno que ya se detecto antes
                        new = False
                        i.updateCoords(cx,cy)   #actualiza coordenadas en el objeto and resets age
                        if i.going_UP(line_down,line_up) == True:
                            cnt_up += 1;
                            print( "ID:",i.getId(),'crossed going up at',time.strftime("%c"))
                            log.write("ID: "+str(i.getId())+' crossed going up at ' + time.strftime("%c") + '\n')
                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1;
                            print( "ID:",i.getId(),'crossed going down at',time.strftime("%c"))
                            log.write("ID: " + str(i.getId()) + ' crossed going down at ' + time.strftime("%c") + '\n')
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        #sacar i de la lista persons
                        index = persons.index(i)
                        persons.pop(index)
                        del i     #liberar la memoria de i
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
            

    for i in persons:

        cv.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv.LINE_AA)
        

    cv.drawContours(frame, [area_pts], -1, color, 2)
    
    str_up = 'IN: '+ str(cnt_up)
    str_down = 'OUT: '+ str(cnt_down)
    aforo = cnt_up - cnt_down
    aforo_txt = 'AFORO: ' + str(aforo)
    frame = cv.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame = cv.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv.LINE_AA)
    cv.putText(frame, str_down ,(10,70),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_down ,(10,70),font,0.5,(255,0,0),1,cv.LINE_AA)

    if (aforo < aforo_max - 2):
        cv.putText(frame, aforo_txt ,(250,50),font,2,(0,255,0),4,cv.LINE_AA)
    elif ((aforo >= aforo_max - 2) and (aforo < aforo_max)):
        cv.putText(frame, aforo_txt ,(250,50),font,2,(200,0,100),4,cv.LINE_AA)
    else:
        cv.putText(frame, aforo_txt ,(250,50),font,2,(0,100,255),4,cv.LINE_AA)

        server = Server().boot()
        server.start()
        sine = Sine(261.63, mul=0.1).out()
        sleep(3)
        server.stop()
        
    cv.imshow('Frame',frame)
    cv.imshow('Mask',mask)    
    
    k = cv.waitKey(50) & 0xff
    if k == 27:
        break

log.flush()
log.close()
cap.release()
cv.destroyAllWindows()
