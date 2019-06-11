from first_app import rootsift
import cv2, glob, time, random
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

def readData(path):
    files = glob.glob(path + "/*")
    imagePath = []
    for i, name in enumerate(files):
        imagePath.append(name)
    return imagePath

def getRootSIFT(gray):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    # extract RootSIFT descriptors
    rs = rootsift.RootSIFT()
    kp, des = rs.compute(gray, kp)
    return kp, des

def getDescriptors(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = getRootSIFT(gray)
    return des

# def createDesc(des_list):
#     # descriptors = des_list[0]
#     # for descriptor in des_list[1:]:
#     #     descriptors = np.vstack((descriptors, descriptor))
#     # return descriptors
#     descrip = np.vstack([des for des in des_list])
#     return descrip

def getAllDescriptors(path, desFile):
    imgpaths = readData(path)
    descsList = []
    for i in imgpaths:
        descsList.append(getDescriptors(i))
    descriptors = np.vstack(descsList)
    # descriptors = np.vstack([des for des in descsList])
    joblib.dump((imgpaths, descsList, descriptors), desFile, compress=3)
    return desFile
def getFeatures(desFile, numWord, trainFile):
    imgpaths, descsList, descriptors = joblib.load(desFile)
    print("num {}".format(len(descriptors)))
    # voc, variance = kmeans(descriptors, numWord, 1)
    model = MiniBatchKMeans(n_clusters=numWord, init_size=numWord*3, batch_size=1000,
            random_state=0).fit(descriptors)
    voc = model.cluster_centers_
   
    print(voc.shape)
    # Calculate the histogram of features
    im_features = np.zeros((len(imgpaths), numWord))
    ww, dd = vq(descsList[1],voc)
    print(ww.shape)
    for i in range(len(imgpaths)):
        words, distance = vq(descsList[i],voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(imgpaths)+1) / (1.0*nbr_occurences + 1)))

    # Perform L2 normalization
    im_features = im_features*idf
    im_features = preprocessing.normalize(im_features, norm='l2')
    joblib.dump((im_features, imgpaths, idf, numWord, voc), trainFile, compress=3)
    return trainFile
def exeSearch(path):
    print("RootSIFT Image input...")
    mydict = {}
    mydict['query'] = path
    image = cv2.imread(path[1:])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = getRootSIFT(gray)
    print("Alo, RootSIFT done!!")
    im_features, image_paths, idf, numWords, voc = joblib.load("first_app/offord1000.pkl")

    test_features = np.zeros((1, numWords))
    words, distance = vq(des,voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features*idf
    test_features = preprocessing.normalize(test_features, norm='l2')
    score = np.dot(test_features, im_features.T)
    rank_ID = np.argsort(-score)

    resultsPath = []
    for ind, i in enumerate(rank_ID[0][:12]):
        temp = image_paths[i].split("/")
        print(temp[1])
        key = "r{}".format(ind)
        mydict[key] = "static/oxford/" + temp[1]
    return mydict