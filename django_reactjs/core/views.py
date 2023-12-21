from django.shortcuts import render
from .models import Files
from rest_framework import viewsets
from .serializers import FilesSerializer
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework import status
from tensorflow.keras.models import Model, load_model
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from rest_framework.renderers import JSONRenderer
import os
from django.conf import settings
from pathlib import Path



# Create your views here.

class FilesViewSet(viewsets.ModelViewSet):
    queryset = Files.objects.all()
    serializer_class = FilesSerializer
    @action(detail=False, methods=["get"], url_path="predict",)
    def predict(seft, request):
        imgurl =  Files.objects.all().values()[0]['img']
        model_path = os.path.join( settings.MODEL_ROOT, 'cnn-parameters-improvement-23-0.91.model')
        model = load_model(filepath=model_path)
        image = os.path.join(settings.MEDIA_ROOT, imgurl)
        ex_img = cv2.imread(image)
        new_img = crop_brain_contour(ex_img)
        new_img = cv2.resize(new_img, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
        new_img = new_img / 255.
        X = np.array([new_img])
        print(new_img.shape)
        print(model_path, image)
        print(model.predict(X))
        if model.predict(X)[0][0] > 0.5:
            predict = 1
        else:
            predict = 0 
        return Response({"predict" :  predict}, status=status.HTTP_200_OK)




def crop_brain_contour(image, plot=False):
    
    #import imutils
    #import cv2
    #from matplotlib import pyplot as plt
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.title('Original Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')
        
        plt.show()
    
    return new_image


