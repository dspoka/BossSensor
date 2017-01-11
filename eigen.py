"""
http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib

import os
import sys

from boss_input import read_face_scrub_csv, IMAGE_SIZE

class Eigen(object):
  def __init__(self):
      self.h = None
      self.w = None
      self.n_samples = None
      self.n_classes = None
      self.n_features = None
      self.images = None
      self.labels = None
      self.dict_actor_id = None
      self.dict_id_actor = None
      self.target_names = None

  def get_data(self):
    images, labels, dict_actor_id, dict_id_actor = read_face_scrub_csv()
    h = IMAGE_SIZE
    w = IMAGE_SIZE
    n_features = h*w*3
    n_samples = len(images)
    n_classes = len(dict_id_actor.keys())
    target_names = list(dict_id_actor.values())

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)


    self.h = h
    self.w = w
    self.n_samples = n_samples
    self.n_classes = n_classes
    self.n_features = n_features
    self.images = images
    self.labels = labels
    self.dict_actor_id = dict_actor_id
    self.dict_id_actor = dict_id_actor
    self.target_names = target_names



def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w, 3)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def get_pca(x, n_components):
  t0 = time()
  pca = PCA(n_components=n_components, svd_solver='randomized',
            whiten=True).fit(x)
  print("done in %0.3fs" % (time() - t0))
  print(pca)
  print(dir(pca))
  print(pca.get_params(deep=True))

  return pca



def eigen_main():
  print(__doc__)
  # Display progress logs on stdout
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

  eigen = Eigen()
  eigen.get_data()
  flat_images = eigen.images.reshape(len(eigen.images),-1)

  # Split into a training set and a test set using a stratified k fold
  X_train, X_test, y_train, y_test = train_test_split(
      flat_images, eigen.labels, test_size=0.25, random_state=42)


  # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
  # dataset): unsupervised feature extraction / dimensionality reduction

  n_components = 150
  pca = get_pca(X_train, n_components)

  eigenfaces = pca.components_.reshape((n_components, eigen.h, eigen.w, 3))

  # print(eigenfaces)
  # os.exit()
  # print("Extracting the top %d eigenfaces from %d faces"
  #       % (n_components, X_train.shape[0]))
  # t0 = time()
  # pca = PCA(n_components=n_components, svd_solver='randomized',
  #           whiten=True).fit(X_train)
  # print("done in %0.3fs" % (time() - t0))
  # eigenfaces = pca.components_.reshape((n_components, h, w, 3))

  print("Projecting the input data on the eigenfaces orthonormal basis")
  t0 = time()
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  print("done in %0.3fs" % (time() - t0))


  ###############################################################################
  # Train a SVM classification model



  print("Fitting the classifier to the training set")
  t0 = time()
  param_grid = {'C': [1e3],
                'gamma': [0.0005], }

# SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.005, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)

  clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
  clf = clf.fit(X_train_pca, y_train)
  print("done in %0.3fs" % (time() - t0))
  print("Best estimator found by grid search:")
  print(clf.best_estimator_)


  ###############################################################################
  # Quantitative evaluation of the model quality on the test set

  print("Predicting people's names on the test set")
  t0 = time()
  y_pred = clf.predict(X_test_pca)
  print("done in %0.3fs" % (time() - t0))

  print(classification_report(y_test, y_pred, target_names=eigen.target_names))
  print(confusion_matrix(y_test, y_pred, labels=range(eigen.n_classes)))


  ###############################################################################
  # Qualitative evaluation of the predictions using matplotlib



  prediction_titles = [title(y_pred, y_test, eigen.target_names, i)
                       for i in range(y_pred.shape[0])]

  # plot_gallery(X_test, prediction_titles, h, w)

  # plot the gallery of the most significative eigenfaces

  # eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
  # plot_gallery(eigenfaces, eigenface_titles, h, w)

  # plt.show()

  joblib.dump(clf, 'eigen.pkl')
  return pca

# clf = joblib.load('filename.pkl')
