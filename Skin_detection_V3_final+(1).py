
# coding: utf-8

# In[121]:


# part - 1
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import scipy as sp
from os import listdir
import time


# In[168]:


scale_X = 50
scale_Y = 41
# path for original image 
PATH='/home/alegehar/Desktop/mahi/original_training'
# path for skin_trainig image 
PATH_SKIN='/home/alegehar/Desktop/mahi/skin_training'
# path for training data set for Y 
PATH_TRAINING_Y='/home/alegehar/Desktop/mahi/training_Y'
# path for training data set for X 
PATH_TRAINING_X='/home/alegehar/Desktop/mahi/training_X'
# path for test image 
PATH_TEST='/home/alegehar/Desktop/mahi/originals_test'
# path for test data set for X 
PATH_TEST_X='/home/alegehar/Desktop/mahi/test_X'


# In[133]:


# list all image in original test directory and store as an array  
im_list=listdir(PATH)
print (len(im_list))
# to access each image and parse 
for i in im_list:
    # read original image data 
    im_original = Image.open('/home/alegehar/Desktop/mahi/original_training/'+i)
    x,y = im_original.size
    # scale the image by 5
    im_original = im_original.resize((x//scale_X, y//scale_Y))
    # list all image pixel data 
    pixels_original = list(im_original.getdata())
    # to store the height and width 
    width, height = im_original.size 
    # print out the width and height
    print("Width", width )
    print("height", height)
    # save the pixels data into csv file 
    np.savetxt("./training_X/training_data_X_"+i.split('.')[0]+".csv", pixels_original, delimiter=",") 


# In[134]:


# list all skin image file from skin training directory 
im_list_skin=listdir(PATH_SKIN)
print len(im_list_skin)
# access each skin image and parse 
for j in im_list_skin:
    # read original image data 
    print j
    im_original_skin = Image.open('/home/alegehar/Desktop/mahi/skin_training/'+j)
    x,y = im_original_skin.size
    im_original_skin = im_original_skin.resize((x//scale_X, y//scale_Y))
    # list all image pixel data 
    pixels_skin = list(im_original_skin.getdata())
    # to store the height and width 
    w, h = im_original_skin.size 
    # print out the width and height
    print("Width", width )
    print("height", height)
    # assign array data
    Y=[]
    for i in pixels_skin:
        # append for non-skin 
        if [255,255,255] in np.array([i[0], i[1], i[2]]):
            # append value 1 for non-skin
            Y.append(1)
        else:
             # append value 2 for skin
            Y.append(2)
    # save the pixels data into csv file 
    #np.savetxt("./skin_data/test_data_X_"+j+".csv", pixels_skin, delimiter=",") 
    np.savetxt("./training_Y/training_data_Y_"+j.split('.')[0]+".csv", Y, delimiter=",")


# In[238]:


im_test=listdir(PATH_TEST)
print (len(im_test))
# to access each image and parse 
for t in im_test:
    # read original image data 
    im_original = Image.open('/home/alegehar/Desktop/mahi/originals_test/'+t)
    x,y = im_original.size
    # scale the image by 5
    #im_original = im_original.resize((x//scale_X, y//scale_Y))
    # list all image pixel data 
    pixels_original = list(im_original.getdata())
    # to store the height and width 
    width, height = im_original.size 
    # print out the width and height
    print("Width", width )
    print("height", height)
    # save the pixels data into csv file 
    np.savetxt("./test_X/test_X_"+t.split('.')[0]+".csv", pixels_original, delimiter=",") 


# In[240]:


# list all training Y files 
training_list_Y= listdir(PATH_TRAINING_Y)
print len(training_list_Y)
print training_list_Y[1]
# list all training X files 
training_list_X= listdir(PATH_TRAINING_X)
print len(training_list_X) 
print training_list_X[1]
test_list_X= listdir(PATH_TEST_X)
print len(test_list_X)


# In[242]:


training_X_all=[]
training_Y_all=[]
test_X_all=[]

for b in range(0,len(training_list_X)):
    
    
    data_Y= np.genfromtxt('./training_Y/'+training_list_X[b].split('.')[0].replace('X', 'Y')+'_s.csv', delimiter=',')
    Y_SIZE= len(data_Y) * 1
    Y=np.resize(data_Y, (Y_SIZE,))
    training_Y_all.append(Y ) 
    # read training data from csv file 
    data_X= np.genfromtxt('./training_X/'+training_list_X[b], delimiter=',')

    X_SIZE= len(data_X) * 1
    X=np.resize(data_X,(X_SIZE,3))
    training_X_all.append(X)
print("Y training size", len(training_Y_all))
print ("X training size", len(training_X_all))
    #read test data from csv file 
    #data_test_X= np.genfromtxt('test_data_X.csv', delimiter=',')

    #X_Test_SIZE= len(data_test_X) * 1

    #test_X = np.resize(data_test_X,(X_Test_SIZE,3)) 
    #print("X test size", len(test_X))
    
    


# In[243]:


N_train_max = 1500
training_X_new = np.empty((N_train_max, 3))
training_Y_new = np.empty((N_train_max, 1))

for c in range(N_train_max):
    g = np.random.randint(1500)
    trainingx_random = training_X_all[g]
    trainingy_random = training_Y_all[g]
    r = np.random.randint(trainingx_random.shape[0])
    training_X_new[c] = trainingx_random[r]
    training_Y_new[c] = trainingy_random[r]


# In[248]:


test_X_all=[]
test_list_X= listdir(PATH_TEST_X)
print len(test_list_X)
test_X_all=[]
for d in range(300,len(test_list_X)):
    data_X_TEST= np.genfromtxt('./test_X/'+test_list_X[d], delimiter=',')
    X_SIZE_TEST= len(data_X_TEST) * 1
    X_TEST=np.resize(data_X_TEST,(X_SIZE_TEST,3))
    test_X_all.append(X_TEST) 



# In[249]:


N_test_max = 500
test_X_new = np.empty((N_test_max, 3))

for c in range(N_test_max):
    g = np.random.randint(100)
    testx_random = test_X_all[g]
    r = np.random.randint(testx_random.shape[0])
    test_X_new[c] = testx_random[r]


# In[250]:


# KNN classifier for two neigbors 
score_ind =  {}

start_time = time.time()
for n in xrange (1,401):
    
    neigh = KNeighborsClassifier(n_neighbors=n,algorithm='brute', metric='euclidean')

    # mode fit for pixels dataset
    neigh.fit( training_X_new,training_Y_new)
    score = neigh.score(training_X_new, training_Y_new) 
    score_ind[n]=score
    predicted =neigh.predict(test_X_all[0])
    print predicted 
end_time = time.time()
print("Time for Processing for neighor {0}".format((end_time - start_time)))
execution_time = end_time - start_time

    


# In[163]:


plt.plot(score_ind.keys(), score_ind.values())
plt.suptitle('Number of neighbors vs performance')
plt.xlabel('Number or neighbors')
plt.ylabel('Performance')
plt.savefig('Performance.jpg')
plt.show()


# In[ ]:


for p in xrange(len(test_X_all)):
    predicted =neigh.predict(test_X_all[p])
    print predicted


# In[236]:


predicted =neigh.predict(test_X_all[0])
print predicted 


# In[89]:


from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='euclidean').fit(training_X_new)


# In[91]:


training_X_new.shape


# In[340]:


new_img = np.ones(shape = (w*h, 3), dtype=np.uint8)
print("new image is of : " + str(new_img.shape))
print("predicted is of : " + str(predicted.shape))
for i in range(0, len(training_X_new)):
    if predicted[i] != 2.0:
        new_img[i] = [0, 0, 0]

new_img = new_img.reshape((h, w, 3))
import matplotlib.pyplot as plt
plt.imshow(new_img)
plt.savefig('predicted.png')
plt.show()
plt.close()


# In[341]:


im_original = Image.open('/home/alegehar/Desktop/mahi/original_training/im00001.jpg')

x_orig, y_orig = im_original.size
skin_prediction = Image.fromarray(new_img)
skin_prediction = skin_prediction.resize((x_orig, y_orig))


skin_mask = np.array(im_original) * np.array(skin_prediction)
skin_mask[skin_mask == 0] = 255
skin_mask = Image.fromarray(skin_mask)

skin_mask

