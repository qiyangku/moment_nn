# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:17:02 2023

@author: dell
"""

import time
import numpy as np

def perf_timer(func):
    '''a decorator for recording execution time of func'''
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        y=func(*args, **kwargs)
        dt = time.perf_counter()-t0
        #print('Time elasped: ', dt)
        return y, dt
    return wrapper



def medianFilter(img):
    width, height= img.shape    
    #create an array for resultant image
    result = np.zeros((height, width))    
    
    # calculate 95% quantile
    #cut_off = np.percentile(img.flatten(), 100)
    #traverse through the pixels of the image
    for i in range(width):
        for j in range(height):        
            #initialize variables
            currentElement=0; left=0; right=0; top=0; bottom=0; topLeft=0; 
            topRight=0; bottomLeft=0; bottomRight=0;          
            
            #get current pixel
            currentElement = img[i,j]
            
            # if currentElement < cut_off:
            #     #skip this pixel if its value is not outlier
            #     result[j,i] = currentElement
            #     continue
            
            #offset is equal to 1 in a 3x3 filter
            offset=1
            
            #get left, right, bottom and top pixels
            #with respect to current pixel
            if not i-offset < 0:
              left = img[i-offset,j]
            if not i+offset > width-offset:
              right = img[i+offset,j]
            if not j-offset < 0:
              top = img[i,j-offset]
            if not j+offset > height-1:
              bottom = img[i,j+offset]
            
            #get top left, top right, bottom left and bottom right pixels
            #with respect to current pixel
            if not i-offset < 0 and not j-offset < 0:
              topLeft = img[i-offset,j-offset]
            if not j-offset < 0 and not i+offset > width-1:
              topRight = img[i+offset,j-offset]
            if not j+offset > height-1 and not i-offset < 0:
              bottomLeft = img[i-offset,j+offset]
            if not j+offset > height-1 and not i+offset > width-1:
              bottomRight = img[i+offset,j+offset]
            
            #get median of all pixels retrieved
            tmp = [currentElement,left,right,top,bottom,topLeft,topRight,bottomLeft,bottomRight]
            cut_off = np.percentile(tmp, 85)
            if currentElement > cut_off:# == np.max(tmp):
                #put median in the same position in resultant array
                result[i,j] = np.median(tmp)      
                #result[i,j] = np.max([left,right,top,bottom,topLeft,topRight,bottomLeft,bottomRight])
            else:
                result[i,j] = currentElement
    #return resultant array  
    return result