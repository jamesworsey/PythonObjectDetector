import cv2 as cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from sklearn.metrics import accuracy_score


#Load an image
def load_image(image_path):
    img = cv2.imread(image_path)
    return img

#Use SIFT to find keypoints and descriptors from image
def SIFT(img):
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(grey,None)
	return kp, des

#Return list of descriptors for each image
def image_set_descriptors(test_train):
	descriptor_list = []
	for i in range(20):
		img = load_image('{}/{}.jpg'.format(test_train,i))
		kp,des = SIFT(img)
		descriptor_list.append(des)
	return descriptor_list

#Find Histogram for each image
def image_histograms(test_train):
	image_histograms = []
	classes=[]
	for i in range(20):
		img = load_image('{}/{}.jpg'.format(test_train,i))
		kp,des = SIFT(img)
		predict_kmeans=kmeans.predict(des)
		hist, bin_edges=np.histogram(predict_kmeans)
		image_histograms.append(hist)
		if i < 10:
			class_sample=1
		else:
			class_sample=0
		classes.append(class_sample)

	return classes, image_histograms


#Find descriptors for training images
descriptors = image_set_descriptors('train')
descriptors=np.asarray(descriptors)
descriptors=np.concatenate(descriptors, axis=0)

 #Train k-means classifier with training images
kmeans = MiniBatchKMeans(n_clusters=2000, random_state=0).fit(descriptors)

#Build histograms from training images
[train_class, train_features] = image_histograms('train')

#Train SVM with histograms from training images
clf = svm.SVC(gamma='auto')
clf.fit(train_features,train_class)

#Test SVM 
#Find descriptors from test images
descriptors = image_set_descriptors('test')
descriptors=np.asarray(descriptors)
descriptors=np.concatenate(descriptors, axis=0)

#Find histograms of test images
[test_class, test_features] = image_histograms('test')

#Use SVM to predict class of images
predict=clf.predict(test_features)

#Calculate accuracy of predictions
score=accuracy_score(np.asarray(test_class), predict)


#Output results
classes = []
for i in range(20):
	if(predict[i] == 1):
		classes.append('Bird')
	else:
		classes.append('Not a bird')

for i in range(20):
	print('{}.jpg: {}'.format(i,classes[i]))

score = score*100
print("Accuracy:" +str(score) + "%")




















