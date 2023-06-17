from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
from sklearn import preprocessing
from keras.models import load_model
from skimage.color import rgb2gray, label2rgb, gray2rgb
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.io import imread,imsave
import cv2
import nibabel as nib
import imageio
import os
import cv2
import numpy as np
from skimage.io import imread, imshow
import random
import pickle
# create an app object using the flask class 
app = Flask(__name__)

# configure upload location
UPLOAD_FOLDER = "static/uploads/"
MODEL_FOLDER = "static/models/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Load the trained model(.pkl, .h5 ...)
modelUNET_necrosis = load_model("models/modelUNET_label_1.h5")
modelUNET_edema = load_model("models/modelUNET_label_2.h5")
modelUNET_enhancing = load_model("models/modelUNET_label_4.h5")
modelUNET_whole = load_model("models/modelUNET_whole.h5")

def area(img):
    area = 0 
    width = img.shape[0]
    height = img.shape[1]
    for i in range(width):
        for j in range(height):
            if img[i][j] > 0:
                area += 1
    return area

def add_contour(img,img_mask): 
    SIZE = 128
    img_mask = img_mask.reshape(SIZE,SIZE)
    
    edges = canny(img_mask,sigma = 1)
    edges = edges.astype(np.uint8)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured = cv2.drawContours(img.copy(), contours, -1, (255,0,0), 1)
    return contoured



img = cv2.imread("C:\\Users\\SonuMonu\\Desktop\\FYP\\contour_test\\test_images\\img.jpg")
mask = cv2.imread("C:\\Users\\SonuMonu\\Desktop\\FYP\\contour_test\\test_images\\img_mask.jpg",0)


def add_overlay(img,mask,color):
    # color --> [0,255,0]
    SIZE = 128

    mask = mask > 100

    for i in range(SIZE):
        for j in range(SIZE):
            if mask[i][j]:
                img[i][j] = color
    return img

def normalise(img):
    np.squeeze(img)
    if (np.max(img) - np.min(img)) != 0:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return img


seg_file = ""
#use the route decorator to tell Flask what URL should trigger the function
img_list = []
survival_days = -1

@app.route('/')
def home(): 
    global img_list,seg_file
    return render_template('page.html',img_list = img_list, seg_file = seg_file)

@app.route('/predict_nifti',methods = ['POST'])
def predict_nifti():
    global img_list,seg_file,survival_days
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        ext = filename.split(".")[-1]
        filename = "upload_nifti." + ext
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        nifti_file = nib.load(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        nifti_file_data = nifti_file.get_fdata()

        # Display Image slices 

        img_list = []
        for i in range(50,120,2):
            img = cv2.rotate(nifti_file_data[:,:,i], cv2.ROTATE_90_CLOCKWISE)
            # Normalization : Min-Max Scaling
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img_list.append(img)
        
        imageio.mimsave(os.path.join(app.config['UPLOAD_FOLDER'],"generated\\all_slices.gif"),img_list,duration = 0.2)

        num1 = random.randint(70,90)

        img1 = cv2.rotate(nifti_file_data[:,:,num1], cv2.ROTATE_90_CLOCKWISE)
        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1)) * 255
        SIZE = 128
        img1 = gray2rgb(img1)
        img1 = resize(img1, (SIZE,SIZE,3), anti_aliasing=True)

        slices = np.array([img1])
        preds_whole = modelUNET_whole.predict(slices, verbose=1)
        
        slice_masks_whole = np.array((preds_whole > 0.5).astype(np.uint8))

        slice_masks_whole = [normalise(img) for img in slice_masks_whole]

        img_path = os.path.join(app.config['UPLOAD_FOLDER'],"generated")
        cv2.imwrite(img_path + "\\slice_1.jpg",img1)
        cv2.imwrite(img_path + "\\slice_1_mask_whole.jpg",slice_masks_whole[0])
        

        whole_tumor = []
        for i in range(60,120): 
            img = cv2.rotate(nifti_file_data[:,:,i], cv2.ROTATE_90_CLOCKWISE)
            # Normalization : Min-Max Scaling
            if (np.max(img) - np.min(img)) != 0:
                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                SIZE = 128
                img = gray2rgb(img)
                img = resize(img, (SIZE,SIZE,3), anti_aliasing=True)
                whole_tumor.append(img) 
                
        whole_tumor = np.array(whole_tumor)

        preds_whole = modelUNET_whole.predict(whole_tumor, verbose=1)
        slice_masks_whole = np.array((preds_whole > 0.5).astype(np.uint8))
        slice_masks_whole = [normalise(img) for img in slice_masks_whole]

        preds_enhancing = modelUNET_enhancing.predict(whole_tumor, verbose=1)
        slice_masks_enhancing = np.array((preds_enhancing > 0.5).astype(np.uint8))
        slice_masks_enhancing = [normalise(img) for img in slice_masks_enhancing]

        preds_edema = modelUNET_edema.predict(whole_tumor, verbose=1)
        slice_masks_edema = np.array((preds_edema > 0.5).astype(np.uint8))
        slice_masks_edema = [normalise(img) for img in slice_masks_edema]

        preds_necrosis = modelUNET_necrosis.predict(whole_tumor, verbose=1)
        slice_masks_necrosis = np.array((preds_necrosis > 0.5).astype(np.uint8))
        slice_masks_necrosis = [normalise(img) for img in slice_masks_necrosis]

        overlay_whole = []
        for i,m in zip(whole_tumor,slice_masks_whole):
            overlay_whole.append(add_overlay(i.copy(),m,[255,215,0]))
        overlay_enhancing = []
        for i,m in zip(overlay_whole,slice_masks_enhancing):
            overlay_enhancing.append(add_overlay(i.copy(),m,[154,205,50]))
        overlay_edema = []
        for i,m in zip(overlay_enhancing,slice_masks_edema):
            overlay_edema.append(add_overlay(i.copy(),m,[176,224,230]))
        overlay_necrosis = []
        for i,m in zip(overlay_edema,slice_masks_necrosis):
            overlay_necrosis.append(add_overlay(i.copy(),m,[255,69,0]))

        imageio.mimsave(os.path.join(app.config['UPLOAD_FOLDER'],"generated\\img_slices.gif"),whole_tumor,duration = 0.4)    
        imageio.mimsave(os.path.join(app.config['UPLOAD_FOLDER'],"generated\\overlay_whole.gif"),overlay_necrosis,duration = 0.4)
        
        vol_whole,vol_necrosis,vol_edema,vol_enhancing = 0,0,0,0
        for i,j,k,l in zip(slice_masks_whole,slice_masks_necrosis,slice_masks_edema,slice_masks_enhancing):
            vol_whole +=  area(i)
            vol_necrosis += area(j)
            vol_edema += area(k)
            vol_enhancing += area(l) 

        vol_necrosis = round(vol_necrosis/vol_whole,3)
        vol_edema = round(vol_edema/vol_whole,3)
        vol_enhancing = round(vol_enhancing/vol_whole,3)

        vol_data = [[vol_necrosis,vol_edema,vol_enhancing]]
        df = pd.DataFrame(vol_data,columns=['necrosis','edema','enhancing'])

        filename = "models/survival_model_svr.pkl"
        survival_model = pickle.load(open(filename,'rb'))
        survival_days = round(survival_model.predict(df)[0])

    return redirect(url_for('results'))
@app.route('/results')
def results():
    return render_template('results.html',prediction = survival_days)
if __name__ == "__main__":
    app.run()
