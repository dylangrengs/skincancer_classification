import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import random
import numpy    
import cv2

from skimage.filters import prewitt_h, prewitt_v, sobel, roberts
from skimage.feature import canny



## file with functions to be used in pynotebooks

# get array determining what label to put for an image --> requires number not str
def getLabelArray(dx):
    if dx == 'mel':
        return numpy.array([1,0,0,0,0,0,0])
    elif dx == 'nv':
        return numpy.array([0,1,0,0,0,0,0])
    elif dx == 'bcc':
        return numpy.array([0,0,1,0,0,0,0])
    elif dx == 'akiec':
        return numpy.array([0,0,0,1,0,0,0])
    elif dx == 'bkl':
        return numpy.array([0,0,0,0,1,0,0])
    elif dx == 'df':
        return numpy.array([0,0,0,0,0,1,0])
    else:
        return numpy.array([0,0,0,0,0,0,1])

## takes original data and preforms neeed data processing
def processData(origData):

    ## drop lession id
    origData.drop('lesion_id', axis=1, inplace=True)

    ## set unknown to null
    origData.loc[origData['sex'] == "unknown"]
    origData.loc[(origData.sex == 'unknown'),'sex']= None   

    origData.loc[origData['localization'] == "unknown"]
    origData.loc[(origData.localization == 'unknown'),'localization']= None

    ## impute data
    imputedData = origData

    imputedData['age'] = origData['age'].interpolate()
    imputedData['sex'] = origData['sex'].interpolate()
    imputedData['localization'] = origData['localization'].interpolate()

    ## change column types
    imputedData["sex"] = imputedData["sex"].astype("category")
    imputedData["localization"] = imputedData["localization"].astype("category")
    imputedData["dx"] = imputedData["dx"].astype("category")     
    imputedData["dx_type"] = imputedData["dx_type"].astype("category")



## takes path to image
## method values can be sobel, roberts, or canny, grey, or meanValue
def processImg(imgPath, method):
    if method == 'roberts':
        img = cv2.imread(imgPath, flags= cv2.IMREAD_GRAYSCALE)
        dim = (300, 225)
        # resize image
        img = cv2.resize(img, dim)

        img  = roberts(img)
        return img

    if method == 'sobel':
        img = cv2.imread(imgPath, flags= cv2.IMREAD_GRAYSCALE)
        dim = (300, 225)
        # resize image
        img = cv2.resize(img, dim)

        img = sobel(img)
        return img
    
    if method == 'canny':
        img = cv2.imread(imgPath, flags= cv2.IMREAD_GRAYSCALE)
        dim = (300, 225)
        # resize image
        img = cv2.resize(img, dim)

        img = canny(img, .7)
        return img
    
    if method == 'grey':
        img = cv2.imread(imgPath, flags= cv2.IMREAD_GRAYSCALE)
        dim = (300, 225)
        # resize image
        img = cv2.resize(img, dim)

        return img
    
    if method == 'meanValue':
        img = cv2.imread(imgPath, flags= cv2.IMREAD_COLOR)
        dim = (300, 225)
        # resize image
        img = cv2.resize(img, dim)

        feature_matrix = numpy.zeros((225,300)) 
        for i in range(0,img.shape[0]):
            for j in range(0, img.shape[1]):
                ave = ((int(img[i,j,0]) + int(img[i,j,1]) + int(img[i,j,2]) )/3 ) # average color channels
                feature_matrix[i][j] = ave

        return feature_matrix



