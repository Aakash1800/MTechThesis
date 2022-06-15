#Grain size distribution
'''
Step 1 : Read image and define pixel size (if needed to convert into microns)
Step 2 : Denoising, if required and threshold image to seperate grain from boundaries
Step 3 : Clean up image if needed (erode,etc.) and create a mask for grain
Step 4 : label grain in masked image
Step 5 : Measure the properties of each grain (object)
Step 6 : Output the result into csv file
'''

from multiprocessing.sharedctypes import Value
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pandas import value_counts
from scipy import ndimage
from skimage import io,color,measure

for k in range(0,1):

    #Step 1
    img_feed = cv.imread("Grain size distribution/VoronoiG"+str(k)+'.jpg',0)    #0 for reading gray level image
    img = cv.bitwise_not(img_feed)  
    #img = img[2:(img.shape[1]-2),2:(img.shape[0]-2)]
    pixels_to_um = 3     # ratio of microns to pixels

    #Step 2
    #plt.hist(img.flat,bins = 100 ,range = (0,255))  #image is 2D array thus img.flat convert it to 1D array
    threshold,thresh = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)  #thresh is 8 bit image we need to convert it into binary
    cv.imwrite('Grain size distribution/threshed'+str(k)+'.jpg',thresh)
    # cv.imshow('Thresh',thresh)
    
    #Step 3
    # kernel = np.ones((3,3),np.uint8)
    # eroded = cv.erode(thresh,kernel,iterations=1)
    # dilated = cv.dilate(eroded,kernel,iterations=1)
    # mask = dilated == 255
    # #mask = np.array(mask,dtype = 'int8')

    #Step 4
    s = [[1,1,1],[1,1,1],[1,1,1]]  #structure factor specifies when to consider a pixel to be connected by nearby pixel, image J specifies
    labeled_mask, num_labels = ndimage.label(thresh,structure=s)  #labels unconnected grains, means if grains are connected by single pixel then it is one object, assign label to all unconnected objects
    print(num_labels)
    img2 = color.label2rgb(labeled_mask,bg_label=0)
    # cv.imshow('Img2',img2)
    # cv.waitKey(0)

    #Step 5
    clusters = measure.regionprops(labeled_mask,img)
    # print(clusters[0])
    # print(clusters[0]['perimeter'])
    # for prop in clusters:
    #     print('Label: {} Area: {}'.format(prop.label,prop.area))

    #Step 6
    propList = ['Area','equivalent_diameter','orientation','MajorAxisLength','MinorAxisLength','Perimeter','MinIntensity','MeanIntensity','MaxIntensity']
    output_file = open('Grain size distribution/imagemeasurements'+str(k)+'.csv','w')   #'w' for write mode, default is text mode
    output_file.write((','+','.join(propList)+'\n'))

    for cluster_props in clusters:
        output_file.write(str(cluster_props['Label']))
        for i,prop in enumerate(propList):
            if (prop=='Area'):
                to_print = cluster_props[prop]*(pixels_to_um*pixels_to_um)   #convert pixel square to um
            elif (prop=='orientation'):
                to_print = cluster_props[prop]*57.2958     #convert to degree from radians
            elif (prop.find('Intensity')<0):
                to_print = cluster_props[prop]*pixels_to_um
            else:
                to_print = cluster_props[prop]
            output_file.write(','+str(to_print))
        output_file.write('\n')

