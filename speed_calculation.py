# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:26:40 2020

@author: skasm
"""
import cv2
import math
import numpy as np
def estimateSpeed(location1, location2,maxWidth, maxHeight,bbox,class_name):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    length_pixels_2 = math.sqrt(math.pow(bbox[0] -bbox[2], 2) + math.pow(bbox[1] - bbox[3], 2))
    # length_car_pixel_2 = abs (bbox[0] -bbox[2])
    if length_car_pixel_2<50:
        
        # d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2))
        # if d_pixels<1:
        #     speed = 0.0
        # else:
        alpha = length_car_pixel_2/50
        if class_name =='car':
            
            car_width = 3.5*alpha
        elif class_name =='Truck':
            car_width = 4.5*alpha
            
        else:
             car_width = 6.5*alpha
        d_meters = d_pixels *car_width/50
    else:
            
        # d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2))
        # if d_pixels<1:
        #     speed = 0.0
        # else:
        if class_name =='car':
            
            car_width = 3.5
        elif class_name =='Truck':
            car_width = 4.5
            
        else:
             car_width = 6.5
        d_meters = d_pixels *car_width/length_car_pixel_2
    # ppm = maxHeight / 50
    # ppm = 11.56
    # d_meters = d_pixels / ppm
    
    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 29
    
    speed = (d_meters * fps * 3.6)/5
    if speed<5:
        speed = 0 
    return int(speed)
def speed_estimation_module(frame,warped_,cars_list, track,bbox,maxWidth, maxHeight):
    cars_list[int(track.track_id),0] = int(track.track_id)
    spd = -1
    if track.track_id:
        if warped_.any():
            location_ = np.argwhere(warped_ > 1)
            location_1 = np.zeros(2)
            location_1[0] = int(np.mean(location_[:,0]))
            location_1[1] = int(np.mean(location_[:,1]))
            if cars_list[int(track.track_id),1]>0:
                if cars_list[int(track.track_id),3]>0:
    
                    if cars_list[int(track.track_id),5]%5==0 and cars_list[int(track.track_id),5]>0:
                        cars_list[int(track.track_id),1] = cars_list[int(track.track_id),3]
                        cars_list[int(track.track_id),2] = cars_list[int(track.track_id),4]
                        
                        cars_list[int(track.track_id),3] = location_1[0]
                        cars_list[int(track.track_id),4] = location_1[1]
                        
                        spd = int (estimateSpeed(cars_list[int(track.track_id),1:3], cars_list[int(track.track_id),3:5],maxWidth, maxHeight,bbox,track.get_class()))
                        cars_list[int(track.track_id),6] = spd
                    # elif cars_list[int(track.track_id),6]>0:
                    #     cv2.putText(frame, str(cars_list[int(track.track_id),6]) ,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                else:
                    cars_list[int(track.track_id),3] = location_1[0]
                    cars_list[int(track.track_id),4] = location_1[1]
                    # cv2.putText(frame,  (int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                cars_list[int(track.track_id),5] = cars_list[int(track.track_id),5]+1
                    
            else:
                cars_list[int(track.track_id),1] = location_1[0]
                cars_list[int(track.track_id),2] = location_1[1]
            
            
            if cars_list[int(track.track_id),6]>0:
                cv2.putText(frame, str(cars_list[int(track.track_id),6]) ,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            # else:
            #     cv2.putText(frame, '-',(int(bbox[0]), int(bbox[1])),0, 0.75, (255,255,255),2)

    return cars_list, frame
# print (cars_list)