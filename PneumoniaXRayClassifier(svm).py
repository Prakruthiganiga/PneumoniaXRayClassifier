#we are using images as the data
#1.download manually images from google
#2.download dataset from kaggle.com
#3.Build a image web crawler
#4.use python libraries to scape the images( using this/bingimagedownloader)

!pip install ipython_autotime
%load_ext autotime

!pip install bing-image-downloader

!mkdir images #make a directory named images

downloader.download("Normal chest X ray",limit=60,output_dir="images",adult_filter_off=True)

#preprocessing the data
#1,resizing
#2.flattening
import os #usng lot of folders we need this
import matplotlib.pyplot as plt #visualizing
import numpy as np #for numerical computing
from skimage.io import imread #to read the images we can use openvc also # skimage means skikit images
from skimage.transform import resize #resizing the images

#images consists of lot of pixels pixels can be in the form of height or width we need to flatten that pixels
target=[]
images=[]
flat_data=[] #images should be flattened

datadir='/content/images' #path of image dir
categories=['Pneumonia X ray' , 'Normal chest X ray']

for category in categories:
  class_num=categories.index(category) #label encoding the values
  path=os.path.join(datadir,category) #creating path to use all the images
  for img in os.listdir(path):
    img_array=imread(os.path.join(path,img))
    #print(img_array.shape)#height,width,rgb distribution
    #plt.imshow(img_array)
    #break (displays only last image )
    img_resized=resize(img_array,(150,150,3))#normalises the value from 0 to 1
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)

flat_data=np.array(flat_data)
target=np.array(target)
images=np.array(images)



flat_data[0]
len(flat_data[0])
target #0 60 times and 1 60 times

unique,count=np.unique(target,return_counts=True)
plt.bar(categories,count)

#splitiing data into Training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(flat_data,target,test_size=0.3,random_state=109)

#we use SVM(support vector machine)model from classification under supervised learning
from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid=[
    {'C':[1,10,100,1000],'kernel':['linaer'] },
    {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf'] }#gets the best particular variend among these two
]
svc=svm.SVC(probability=True)
clf=GridSearchCV(svc,param_grid)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
y_pred
y_test

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)

#saving the model using picle library
import pickle
pickle.dump(clf,open('img_model.p','wb'))#helpful when using for next time
model=pickle.load(open('img_model.p','rb'))

#testing a new image eg:from google
flat_data=[]
url=input('Enter your URL')
img=imread(url)
img_resized=resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data=np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out=model.predict(flat_data)
y_out=categories[y_out[0]]
print(f'PREDICTED OUTPUT:{y_out}')

#copy the url of any normal or pneumonia chest xray from web browser and check the model
