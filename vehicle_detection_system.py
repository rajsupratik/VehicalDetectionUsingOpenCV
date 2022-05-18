import cv2 
import numpy as np

#video file
video=cv2.VideoCapture('video_truck_Trim.mp4')
'''video=cv2.VideoCapture('trim.mp4')'''

#pre trained algorithm
classifier_car='cars.xml'
classifier_pedestrians='pedestrian.xml'
classifier_bike='bike.xml'
classifier_bus='bus.xml'


while True:
    #read current frame
    read_frame,frame= video.read()
     
    #safe coding
    if read_frame:
        #converting it to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #creating a classifier file
    trained_car_data = cv2.CascadeClassifier(classifier_car)
    trained_pedestrian_data=cv2.CascadeClassifier(classifier_pedestrians)
    trained_bike_data=cv2.CascadeClassifier(classifier_bike)
    trained_bus_data=cv2.CascadeClassifier(classifier_bus)

    #detecting cars pedestrians and other vihecles
    car_coordinates = trained_car_data.detectMultiScale(gray_img)
    pedestrian_coordinates=trained_pedestrian_data.detectMultiScale(gray_img)
    bike_coordinates=trained_bike_data.detectMultiScale(gray_img)
    bus_coordinates=trained_bus_data.detectMultiScale(gray_img)
    
    #print(car_coordinates,pedestrian_coordinates and other vihecles_coordinates) and draw rectangle around
    for (x,y,w,h) in car_coordinates:
         cv2.rectangle( frame ,(x,y), (x+w,y+h), (0,0,255), 2)
    
    for (x,y,w,h) in pedestrian_coordinates:
         cv2.rectangle( frame ,(x,y), (x+w,y+h), (0,225,255), 2)
    
    for (x,y,w,h) in bike_coordinates:
         cv2.rectangle( frame ,(x,y), (x+w,y+h), (0,0,255), 2)
    
    for (x,y,w,h) in bus_coordinates:
         cv2.rectangle( frame ,(x,y), (x+w,y+h), (0,0,255), 2)

    cv2.imshow('car Detector', frame)
    key= cv2.waitKey(1)
    
    # stop if Q button is hit
    if key==81 or key==113:
        break
#release the video
video.release()        
