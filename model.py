from numpy import loadtxt
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# load model
model1 = load_model('final_model.h5')



def predict(image_path):
    dim=128
    category={'0':'Anger','1':'Disgust','2':'Fear','3':'Happiness','4':'Sadness','5':'Surprise'}
    model1 = load_model('final_model.h5')
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(image,(128,128))
    img_arr=[]
    img_arr.append(img)
    # print(image)
    image_arr=np.array(img_arr)/255
    print(image_arr.shape)
    reshap_img=image_arr.reshape(1,128,128,1)
    ans= model1.predict(reshap_img)
    ans=np.argmax(ans)
    return category[str(ans)]