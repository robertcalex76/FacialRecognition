
#Robert Alex CS584 Rangwala Final Project G# G01287617

import string
import random
import math
import os, shutil
import imageio
import cv2
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta
from scipy import spatial
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle


dim = 512 #this is the 512x512 for the size of the saved face, but it must be smaller for large folder 

def setup(folder):

    global dim

    print("Pulling Pics from raw")
    
    #go through picture files in directory
    if folder == 1:
        path = './all/'
        dim = 200      #reduces resolution for big folder to ensure PCA fits within RAM limitations  
    elif folder == 2:
        path = './big_test/'
    else:
        path = './Friends_Family/'
        
    outPath = './heads/'

    count = 0

    for image_path in os.listdir(path):

        if (count % 200 ) == 0:
            print()
            print(count)
            print()
            
        temp = image_path.split('0')
        name = temp[0]
        print(name)

        #get pic for detection
        input_path = os.path.join(path, image_path)
        raw_pic = cv2.imread(input_path)
        
        #pull faces from pictures
        detected = detectFace(raw_pic, dim)

        if len(detected) != 0:
            #save new face pic into directory for that person
            #if person does not exist in database, make new directory for person
            if ((os.path.exists(os.path.join(outPath,name))) == False):
                dirpath = os.path.join(outPath, name)
                os.mkdir(dirpath)
                savepath = os.path.join(dirpath, ("0-"+image_path))
                cv2.imwrite(savepath, detected)
            #if person exists already, add photo to directory with photo num
            else:
                dirpath = os.path.join(outPath, name)
                pic_num = len(os.listdir(dirpath))
                savepath = os.path.join(dirpath, (str(pic_num) + "-" +image_path))
                cv2.imwrite(savepath, detected)

        count += 1
        
    print()

#uses opencv to detect and extract face
def detectFace(raw_pic, dim):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(raw_pic, cv2.COLOR_BGR2GRAY)

    
    #detect faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 7, minSize=(30,30)) #change these values to alter the facial detection algorithm
    

    if len(faces) == 0:
        return []
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        crop_img = gray[y:y+h, x:x+w]

    resized = cv2.resize(crop_img, (dim, dim), interpolation = cv2.INTER_AREA)
    equal = cv2.equalizeHist(resized)
    blur = cv2.GaussianBlur(equal,(5,5),sigmaX=0)
    #blur = cv2.bilateralFilter(equal ,15,80,80)
    # Display the output
    #cv2.imshow('img', raw_pic)
    #cv2.waitKey()
    return blur

#Do validation on images in directory. test against themselves
def validation(folder, classifier, shuffle):

    train_list = []
    train_labels = []

    if folder == 1: #(all)
        components = 20
    if folder == 2: #(big_test)
        components = 20
    if folder == 3: #(Friends_Family)
        components = 100

    #retrieves training data from folder
    directory = './heads/'
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".jpg") or filepath.endswith(".png") or filepath.endswith(".jpeg"):
                pic_split = filepath.split('-')
                name_split = pic_split[1].split('0')
                name = name_split[0]
                train_labels.append(name)

                img = cv2.imread(filepath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                numpyarray = np.array(gray)
                train_list.append(numpyarray.flatten())

    train_list = np.array(train_list)
    train_labels = np.array(train_labels)

    #kf = KFold(n_splits = int(folds), shuffle = True)

    print("PCA Starting")
    pca = PCA(n_components = components, whiten = True).fit(train_list)

    x_train_pca = pca.transform(train_list)
    #x_test_pca = pca.transform(X_test.reshape(1, -1))

    #MLP Classifier
    if classifier == 1:
        print("Starting MLP Classifier")
        clf = MLPClassifier( max_iter = 100, batch_size = 256, alpha = 1e-5, hidden_layer_sizes= (1024,), random_state = 1, verbose = True).fit(x_train_pca, train_labels)

    #SVM Classifier
    elif classifier == 2:
        print("Starting SVM Classifier")
        clf = make_pipeline(StandardScaler(), SVC(gamma = 'auto', C= 5, probability = True)).fit(x_train_pca, train_labels)

    #KNN Classifier
    elif classifier == 3:
        print("Starting KNN Classifier")
        clf = KNeighborsClassifier(n_neighbors = 9).fit(x_train_pca, train_labels)

    print("Predicting")
    total_total = 0
    total_correct = 0
    for indx, test in enumerate(train_list):
            total_total += 1
            x_test_pca = pca.transform(test.reshape(1, -1))
            correct = "wrong"
            pred = clf.predict(x_test_pca)
            if (pred == train_labels[indx]):
                correct = "right"
                total_correct += 1
            print("Prediction: " + str(pred) + " Actual: " + str(train_labels[indx]) + " " + str(correct))
    print()
    print("Final results - Total tested: " + str(total_total) + " Percent Correct: " + str(total_correct / total_total))

#train neural network, test one example in the TEST_IMAGE folder
def train(folder, classifier, shuffle, iter_count):

    print("Start Train")

    #Change these to alter the number of pca components
    if folder == 1: #(all)
        components = 50
    if folder == 2: #(big_test)
        components = 50
    if folder == 3: #(Friends_Family)
        components = 50
    iterations = iter_count

    final_correct = 0
    final_tested = 0
    final_top_5 = 0
    #This loop is for the shuffle test to get an average accuracy per person, only 1 run is done when testing in the test folder
    for iterate in range(iterations):
   
        train_list = []
        train_labels = []

        #retrieves training data from folder
        directory = './heads/'
        for subdir, dirs, files in os.walk(directory):
            for filename in files:
                filepath = subdir + os.sep + filename

                if filepath.endswith(".jpg") or filepath.endswith(".png") or filepath.endswith(".jpeg"):
                    pic_split = filepath.split('-')
                    name_split = pic_split[1].split('0')
                    name = name_split[0]
                    train_labels.append(name)

                    img = cv2.imread(filepath)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    numpyarray = np.array(gray)
                    train_list.append(numpyarray.flatten())


        train_list = np.array(train_list)
        train_labels = np.array(train_labels)

        #get testing images
        test_imgs, test_labels = getTestImages()

        #if shuffle is turned on, then the test pics get shuffled with the training pics
        if shuffle == 1:
            train_list, train_labels, test_imgs, test_labels  = shuffle_func(train_list, train_labels, test_imgs, test_labels)
            
        #MLP CLASSIFIER NO PCA
        no_pca = False #change this to do brute force MLP, will most likely run out of rescources 
        if no_pca == True:
            clf = MLPClassifier(solver='lbfgs', max_iter = 100, alpha = 1e-5, hidden_layer_sizes=(1024,), random_state = 1)
            clf.fit(train_list, train_labels)
            prediction = clf.predict(test_imgs.flatten().reshape(1, -1))
            print(prediction)

        print()
        print("PCA Starting")
        pca = PCA(n_components = components, whiten = True).fit(train_list) 
        x_train_pca = pca.transform(train_list)
        x_test_pca = pca.transform(test_imgs)

        print()
        
        #MLP Classifier
        if classifier == 1:
            print("Starting MLP Classifier")
            clf = MLPClassifier( max_iter = 1000, batch_size = 256, alpha = 1e-5, hidden_layer_sizes= (1024,), random_state = 1, verbose = True).fit(x_train_pca, train_labels) #Change MLP parameters <---

        #SVM Classifier
        elif classifier == 2:
            print("Starting SVM Classifier")
            clf = make_pipeline(StandardScaler(), SVC(gamma = .01 , C=5, probability = True)).fit(x_train_pca, train_labels) #Change SVM Parameters <---

        #KNN Classifier
        elif classifier == 3:
            print("Starting KNN Classifier")
            clf = KNeighborsClassifier(n_neighbors = 9).fit(x_train_pca, train_labels) #Change KNN parameters <---
            
        print("Predicting")
        n = 5 #change this to change how many highest probabilities are printed
        total_tested = 0
        total_correct = 0
        total_top_5 = 0
        for indx, test in enumerate(x_test_pca):
            total_tested += 1
            new_pred = clf.predict(test.reshape(1, -1))
            pred_probs = clf.predict_proba(test.reshape(1, -1))
            correct = "WRONG"
            if new_pred == test_labels[indx]:
                correct = "CORRECT"
                total_correct += 1
                
            print("Test Image Number " + str(indx + 1) + "  " + str(correct) + ": Prediction - " + str(new_pred) + "  Actual  - " + str(test_labels[indx]))
            
            probs_indxs = (-pred_probs[0]).argsort()[:n]
            for prob_indx in probs_indxs:
                if clf.classes_[prob_indx] == test_labels[indx]:
                    total_top_5 += 1
                print(str(clf.classes_[prob_indx]) + " " + str(round(pred_probs[0, prob_indx] * 100, 5)) + "%")

            print()

        print()
        print("Run Summary: "  + "Accuracy = " + str(round((total_correct / total_tested) * 100, 2)) + "%" + "|| Total Tested = " + str(total_tested) + " || Total Correct = " + str(total_correct) + " || Total Near Miss = " + str(total_top_5) + " || Run Near Miss Percentage = " + str(round((total_top_5 / total_tested) * 100, 2)) + "%")
        final_tested += total_tested
        final_correct += total_correct
        final_top_5 += total_top_5

    print()
    print()
    print("-------------------------------------")
    print("Final Summary: " + "Accuracy = " + str(round((final_correct / final_tested) * 100, 2)) + "%" + " || Total Tested = " + str(final_tested) + " Final Correct = " + str(final_correct) + " || Final Near Miss Predictions = " + str(final_top_5) + " || Final Near Miss Percentage = " + str(round((final_top_5 / final_tested) * 100, 2)) + "%")
    print("-------------------------------------")
    print()  

#Function to get the images being tested from the TEST_IMAGES folder
def getTestImages():

    global dim
    
    test_images = []
    test_labels = []
    path = './TEST_IMAGES/'
    outpath = './Test_output/'
    for image_path in os.listdir(path):

        temp = image_path.split('0')
        test_labels.append(temp[0])
        test_path = os.path.join(path, image_path)
        raw_test = cv2.imread(test_path)
        detected_test = detectFace(raw_test, dim)
        test_array = np.array(detected_test)
        out = os.path.join(outpath, image_path)
        cv2.imwrite(out, detected_test)
        test_images.append(test_array.flatten())

    test_array_final = np.array(test_images)
    return test_array_final, test_labels

#used to shuffle around pictures of test subjects (randomly select photos of those in the test_imgs pool, not only use the photo in the test folder)
def shuffle_func(train_list, train_lables, test_list, test_lables):

    total_list = np.append(train_list, test_list, axis = 0)
    total_labels = np.append(train_lables, test_lables)

    total_list, total_labels = shuffle(total_list, total_labels)

    n_test_list = []
    n_test_labels = []
    
    for test_label in test_lables:
        found = 0
        for indx, train_label in enumerate(total_labels):
            if (train_label == test_label and found != 1):
                n_test_list.append(total_list[indx])
                n_test_labels.append(train_label)
                total_list = np.delete(total_list, indx, 0)
                total_labels = np.delete(total_labels, indx)
                found = 1

    n_test_list = np.array(n_test_list)
    n_test_labels = np.array(n_test_labels)

    return total_list, total_labels, n_test_list, n_test_labels
    

#used to clean "heads" directory before next use
def clean_func():
    folder = './heads/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#Main Function
def main():

    #Changes utility to clean heads directory before execution
    clean = True  
    if clean == True:
        clean_func()
        
    start = timer()

    validate = input("Self-validation or test folder? (1: test folder, 2: validation) ")
    
    folder = input("Which data set? (1: all, 2: big_test, 3: Friends and Family) ")

    classifier = input("Select Classifier: (1: MLP, 2: SVM, 3: KNN) ")

    shuffle = input("Shuffle test pictures?: (1: shuffle, 2: no shuffle) ")

    iter_count = 1
    if int(shuffle) == 1:
        iter_count = input("Input how many test runs to average accuracy: ")

    if int(validate) == 2:
        
        setup(int(folder))
        validation(int(folder), int(classifier), int(shuffle))

    else:

        setup(int(folder))
        train(int(folder), int(classifier), int(shuffle), int(iter_count))

    
    print("Done")

    end = timer()

    print(timedelta(seconds = end - start))
    

if __name__ == "__main__":
    main()
