from flask import render_template, request
import os
import cv2
from PIL import Image 
import PIL 
from app.face_recognitions import facerecognitionpipeline
import matplotlib.image as matim 
import glob

UPLOAD_FOLDER = 'static/upload'



def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    if request.method=='POST':
        f = request.files["image_name"]
        filename = f.filename
        # basedir = os.path.abspath(os.path.dirname(__file__))
        from os.path import join, dirname, realpath

        UPLOADS_PATH = join(dirname(realpath(__file__)), '..\\static\\upload',filename)
        # save or image to upload folder
        # path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(UPLOADS_PATH) #save image to upload folder
        # get pridictions
        pred_image, predictions = facerecognitionpipeline(UPLOADS_PATH)
        pred_filename = "predicted_image.jpg"
        path = os.path.realpath(__file__)
        dir = os.path.dirname(path)
        dir = dir.replace('app','static\predict')
        os.chdir(dir)
        # print(dir)
        cv2.imwrite(pred_filename,pred_image)
        # print(predictions)

        report = []
        for i, obj in enumerate(predictions):
            gray_image = obj['roi'] # gray scale image
            eigen_image = obj['eig_img'].reshape(100,100) # eigen_image_array
            gender_name = obj["prediction_name"]
            score = round(obj["score"]*100,2) #probability score

            #saving gray and eigen image in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eigan_image_name = f'eigan_{i}.jpg'
            matim.imsave(gray_image_name,gray_image,cmap='gray')
            matim.imsave(eigan_image_name,eigen_image,cmap='gray')

            report.append([gray_image_name,eigan_image_name,gender_name,score])
        
        # removing files form upload folder
        
        files = glob.glob(dir.replace('predict','upload')+"/*")
        for f in files:
            os.remove(f)
        return render_template('gender.html',fileupload=True,report=report) # POST request

            # print("ML model predicted Successfullt")



    return render_template('gender.html',fileupload=False,report=report) # GET request