# -*- coding: utf-8 -*-
"""
Created on Sat Aug 7 19:50:27 2016

@author: tmiura
"""

#%%
# settings
#
path_img = "./img/"
path_lbl = "./lbl/"
# list of (color image, label image)
# training dataset
img_lbl_filenames_train  = [
                           ("06Apr03Face.jpg", "06Apr03Face.png"), 
                           ("chenhao0017me9.jpg", "chenhao0017me9.png"),
                           ("our-familly.jpg", "our-familly.png"), 
                           ("family_bible_study.jpg", "family_bible_study.png"), 
                           ("royal_family_fs_108e_2.jpg", "royal_family_fs_108e_2.png"), 
                           ("Kishani_-__n_-_5.jpeg", "Kishani_-__n_-_5.png"),
                           ("Srilankan-Actress-Yamuna-Erandathi-001.jpg","Srilankan-Actress-Yamuna-Erandathi-001.png"),
                           ("normalc67bdab.jpg","normalc67bdab.png"),
                           ("toddler_mollie_aug07_400.jpg","toddler_mollie_aug07_400.png"),
                           ("Salma-Hayek-face-wi-new-lg.jpg","Salma-Hayek-face-wi-new-lg.png"),
                           ("Matthew_narrowweb__300x381,0.jpg","Matthew_narrowweb__300x381,0.png"),
                           ("infohiding.jpg","infohiding.png"),
                           ("w_sexy.jpg","w_sexy.png"),
                           ("m_unsexy_gr.jpg","m_unsexy_gr.png"),
                           ("josh-hartnett-Poster-thumb.jpg","josh-hartnett-Poster-thumb.png"),
                           ("Family_Bryce.jpg","Family_Bryce.png"),
                           ("buck_family.jpg","buck_family.png"),
                           ("920480_f520.jpg","920480_f520.png"),
                           ("friends.jpg","friends.png"),
                           ("familySri.jpg","familySri.png"),
                           ("Family-Cell-C.jpg","Family-Cell-C.png"),
                           ("family-photo-2005-10.jpg","family-photo-2005-10.png"),
                           ("MyIndianFamily-2007-1.jpg","MyIndianFamily-2007-1.png"),
                           ("0520962400.jpg","0520962400.png"),
                           ("abbasprize.jpg","abbasprize.png"),
                           ("tang-wei-1.jpg","tang-wei-1.png"),                          
                           ]
# test dataset  !!do not change this test dataset!!
img_lbl_filenames_test = [
                           ("m(01-32)_gr.jpg", "m(01-32)_gr.png"),
                           ("toddler_mollie_aug07_400.jpg", "toddler_mollie_aug07_400.png"),
                           ("family4.jpg", "family4.png"), 
                           ("obama.jpg", "obama.png"), 
                           ]


#%%
# function definitions to read/write image pixels and pixel labels
#
import numpy as np
from time import time
from PIL import Image

def read_image_as_3d_points(f):
    img = Image.open(f)
    img = np.asarray(img)
    height, width, colors = img.shape
    assert colors == 3
    features = np.reshape(img, (width * height, colors))
    return features

def read_image_as_labels(f):
    lbl = Image.open(f).convert("L")
    width, height = lbl.size
    labels = np.asarray(lbl).ravel()/255
    return labels

def fetch_data(image_label_filenames):
    features = np.empty((0,3), int)
    labels = np.empty((0), int)
    for f in image_label_filenames:
        f_img, f_lbl = f
        f_img = path_img + f_img
        f_lbl = path_lbl + f_lbl
        features = np.append(features, read_image_as_3d_points(f_img), axis = 0)
        labels = np.append(labels, read_image_as_labels(f_lbl), axis = 0)
    return features, labels

def save_labels_as_image(label_pred, f):
    img = Image.open(f)
    width, height = img.size
    lbl = np.reshape(label_pred, (height, width)).astype(np.uint8)
    lbl = Image.fromarray(lbl*255)
    import os
    basename = os.path.splitext(os.path.basename(f))[0]
    lbl.save(path_lbl + basename + "_pred_label.png")
    Image.composite(img, Image.new(img.mode, img.size), lbl).save(path_img + basename + "_pred_skin.png")


#%%
# define a classifier 
#
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=15)


#%%
# load training data
#
print("Loading training images and labels ..")
t0 = time()
features, labels = fetch_data(img_lbl_filenames_train)

# reduce training data for saving computation time
from sklearn.cross_validation import train_test_split
features, features_NA, labels, labels_NA = train_test_split(features, labels, 
                                                            test_size=0.90, random_state=2016)
print('done in %.2fs.' % (time() - t0))


t0_all = time()

#%%
# train the classifier
#
print("Training ..")
t0 = time()
clf.fit(features, labels)
print('done in %.2fs.' % (time() - t0))


#%%
# evaluate training errors and test errors
#
print("Evaluating ..")

from sklearn.metrics import precision_score, recall_score, f1_score

labels_pred = clf.predict(features)
print("  train: P={0:f}, R={1:f}, F ={2:f}".format(precision_score(labels,labels_pred), recall_score(labels,labels_pred),f1_score(labels,labels_pred)))

features_test, labels_test = fetch_data(img_lbl_filenames_test)
labels_pred = clf.predict(features_test)
print("  test : P={0:f}, R={1:f}, F ={2:f}".format(precision_score(labels_test,labels_pred), recall_score(labels_test,labels_pred),f1_score(labels_test,labels_pred)))

print('done in %.2fs (should be < 180 seconds).' % (time() - t0_all))


#%%
# save predicted labels for training images
#
print("Saving predicted skin regions for ..")
t0 = time()
for f in img_lbl_filenames_train:
    f_img, f_lbl = f
    f_img = path_img + f_img
    f_lbl = path_lbl + f_lbl
    X = read_image_as_3d_points(f_img)
    y_true = read_image_as_labels(f_lbl)
    y_pred = clf.predict(X)
    save_labels_as_image(y_pred, f_img)
    print("  training image {0:s}:  P={1:f}, R={2:f}, , F ={3:f}".format(f_img, precision_score(y_true, y_pred), recall_score(y_true,y_pred),f1_score(y_true,y_pred)))
print('done in %.2fs.' % (time() - t0))


#%%
# save predicted labels for test images
#
print("Saving predicted skin regions for ..")
t0 = time()
for f in img_lbl_filenames_test:
    f_img, f_lbl = f
    f_img = path_img + f_img
    f_lbl = path_lbl + f_lbl
    X = read_image_as_3d_points(f_img)
    y_true = read_image_as_labels(f_lbl)
    y_pred = clf.predict(X)
    save_labels_as_image(y_pred, f_img)
    print("  test image {0:s}:  P = {1:f}, R = {2:f}, F = {3:f}".format(f_img, precision_score(y_true, y_pred), recall_score(y_true, y_pred),f1_score(y_true, y_pred)))
print('done in %.2fs.' % (time() - t0))
