import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sklearn
import pickle
import os

# load all models
path = os.path.realpath(__file__)
  
# gives the directory where demo.py 
# exists
dir = os.path.dirname(path)
dir = dir.replace('app', 'model')
print(dir)
os.chdir(dir)
  

haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_svm = pickle.load(open('model_svm.pickle',mode='rb'))
pca_models = pickle.load(open('pca_dict.pickle','rb'))

model_pca = pca_models['pca']
mean_face_arr = pca_models['mean_face']


def facerecognitionpipeline(filename):
    # step-1 Read image
    img = cv2.imread(filename)
    # step-2 convret to gray scale
    gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # step-3 crop the face using haar cascade classifier
    faces = haar.detectMultiScale(gray,1.5,3)
    predictions = []
    for x,y,w,h in faces:
        roi = gray[y:y+h,x:x+h]
    #     plt.imshow(roi,cmap='gray')
        # step-4 normalization 0-1
        roi = roi/255.0
        # step-5 resize image 100x100
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        # step-6 flattern 1x10000
        roi_reshape = roi_resize.reshape(1,10000)
        # step-7 substarct with mean
        roi_mean = roi_reshape - mean_face_arr
        # step-8 get eigen image
        eigen_image = model_pca.transform(roi_mean)
        # step-9 Eigne image for visualization
        eig_img = model_pca.inverse_transform(eigen_image)
        # step-10 pass to ml model (SVM) and get predictions
        result = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        # step-11 generate report
        text = "%s : %d"%("female" if result[0]!='male' else 'male',prob_score_max*100)
        print(text)
        # defining colors based on results
        if result[0]=='male':
            color=(255,255,0)
        else:
            color = (0,255,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
        output = {
            'roi':roi,
            'eig_img':eig_img,
            'prediction_name':"female" if result[0]!='male' else 'male',
            'score':prob_score_max
        }
        predictions.append(output)

    return img,predictions
