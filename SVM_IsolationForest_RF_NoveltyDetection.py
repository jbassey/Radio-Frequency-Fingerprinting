#import matplotlib
##matplotlib.use('Agg')
import os,random
# os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
import numpy as np
from numpy import *
import tensorflow as tf
from sklearn.utils import shuffle
from random import seed
import tensorflow as tf
import pandas 
import time
from sklearn.cluster import MeanShift, estimate_bandwidth
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import gc
from keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import seaborn as sns

from pylab import rcParams

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#%matplotlib inline
import seaborn as sns
#from utils import plot_confusion_matrix

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', choices=['Adam','sgd','adagrad'], default='Adam')
parser.add_argument('--loss', choices=['mse','mae' 'binary_crossentropy', 'categorical_crossentropy'], default='mse')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_samples', type=int, default= 50)
parser.add_argument('--result', default=os.path.join(curdir, 'result.png'))
parser.add_argument('--comp_vector', type=bool, default= False)
parser.add_argument('--cartesian', type=bool, default= True)
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--window', type=int, default= 64)
parser.add_argument('--trainProp', type=float, default= 0.7)
parser.add_argument('--size', type=int, default= 16)
parser.add_argument('--model_name', choices=['autoencoder', 'deep_autoencoder','convolutional_autoencoder'], default='convolutional_autoencoder')


rcParams['figure.figsize'] = 14, 8.7 # Golden Mean
LABELS = ["Normal","Fraud"]
col_list = ["cerulean","scarlet"]# https://xkcd.com/color/rgb/
##sns.set(style='white', font_scale=1.75, palette=sns.xkcd_palette(col_list))

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


def adapt_y(x_concat, y_concat,shuff,intr):
    #y_concat = np.ravel(y_concat)
    unique_elements, counts_elements = np.unique(y_concat, return_counts=True)
    print(np.asarray((unique_elements, counts_elements)))
    #y_concat[np.where(y_concat != np.array(intr).any())] = 10000
    #y_concat[np.where(y_concat == np.array(intr).any())] = 20000
    y_concat[np.where(~np.in1d(y_concat, np.array(intr)))[0]] = 10000 #auth
    y_concat[np.where(np.in1d(y_concat, np.array(intr)))[0]] = 20000 #intr
    
    lb_make = LabelEncoder()
    y_concat = lb_make.fit_transform(y_concat)
    unique_elements, counts_elements = np.unique(y_concat, return_counts=True)
    #print(np.asarray((unique_elements, counts_elements)))
    if shuff:
        x_concat, y_concat = shuffle(x_concat, y_concat)
        #print (x_concat.shape, y_concat.shape)
    return x_concat, y_concat

def load_image(path, size,comp_vector, cartesian, window,trainProp):
    # data augmentation logic such as random rotations can be added here
    if args.model_name == 'autoencoder' or args.model_name =='deep_autoencoder':
        window = int(window/2)
        dim = 2
    elif args.model_name == 'convolutional_autoencoder':
        dim = 2
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(("AllDev.txt")):
                X,y=get_data(root+'/'+name, 6, [window],comp_vector ,cartesian,dim)
    return X,y#img_to_array(X),y

def main(args):
    path =  '/media/joshua/Data/python_codes/fingerprinting/internship_experiments/AnomalyDetectionUsingAutoencoder-master/data/'##single/all'
    #path =  '/media/joshua/Data/python_codes/fingerprinting/internship_experiments/AnomalyDetectionUsingAutoencoder-master/ID_data/single/'
    intruder = [0]#4#
    intr=[0]
    snr = '0_1db'
    X0,y0 = load_image(path +snr,args.size,args.comp_vector , args.cartesian, args.window,args.trainProp)

    #X0,y0 = shuffle(X0,y0)
    #unique_elements, counts_elements = np.unique(y0, return_counts=True)
    #print(np.asarray((unique_elements, counts_elements)))

    [X_train, y_train], [X_val, y_val], [X_tes, y_tes] = data_split(X0, y0,args.trainProp, intruder)


    X_test = np.concatenate((X_tes[np.where(np.in1d(y_tes, np.array(0)))[0]],X_tes[np.where(np.in1d(y_tes, np.array(1)))[0]],
                                     X_tes[np.where(np.in1d(y_tes, np.array(2)))[0]],X_tes[np.where(np.in1d(y_tes, np.array(3)))[0]],
                                     X_tes[np.where(np.in1d(y_tes, np.array(4)))[0]],X_tes[np.where(np.in1d(y_tes, np.array(5)))[0]]),axis=0)
    
    y_test = np.concatenate((y_tes[np.where(np.in1d(y_tes, np.array(0)))[0]],y_tes[np.where(np.in1d(y_tes, np.array(1)))[0]],
                                     y_tes[np.where(np.in1d(y_tes, np.array(2)))[0]],y_tes[np.where(np.in1d(y_tes, np.array(3)))[0]],
                                     y_tes[np.where(np.in1d(y_tes, np.array(4)))[0]],y_tes[np.where(np.in1d(y_tes, np.array(5)))[0]]),axis=0)                                
    
    scaler = StandardScaler()

    if args.dim == 2:#
        s0, s1, s2 = X_train.shape[0], X_train.shape[1], X_train.shape[2]
        X_train = X_train.reshape(s0 * s1, s2)
        X_train = scaler.fit_transform(X_train)
        X_train = X_train.reshape(s0, s1, s2)

        s0, s1, s2 = X_test.shape[0], X_test.shape[1], X_test.shape[2]
        X_test = X_test.reshape(s0 * s1, s2)
        X_test = scaler.transform(X_test)
        X_test = X_test.reshape(s0, s1, s2)

        s0, s1, s2 = X_val.shape[0], X_val.shape[1], X_val.shape[2]
        X_val = X_val.reshape(s0 * s1, s2)
        X_val = scaler.fit_transform(X_val)
        X_val = X_val.reshape(s0, s1, s2)
    elif args.dim == 1:#

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.fit_transform(X_val)
        X_test = scaler.transform(X_test)
        print (X_train.min(), X_train.max(), X_test.min(), X_test.max())
            

    ### Take PCA to reduce feature space dimensionality
    ##pca = PCA(n_components=3, whiten=True)
    ##pca = pca.fit(X_train)
    ##print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    ##X_train = pca.transform(X_train)
    ##X_test = pca.transform(X_test)
    ###xval = pca.transform(xval)



    ## Train classifier and obtain predictions for OC-SVM
    oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
    if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=0.4, n_estimators=40)  # Obtained using grid search

    oc_svm_clf.fit(X_train)
    if_clf.fit(X_train)

    oc_svm_preds = oc_svm_clf.predict(X_test)
    if_preds = if_clf.predict(X_test)

    #calculate accuracy metrics
    print ("SVM OOC accuracy: ", accuracy_score(y_test, oc_svm_preds))
    print ("IF accuracy: ", accuracy_score(y_test, if_preds))

    df = pd.DataFrame({'Labels': np.ravel(y_test), 'Clusters': np.ravel(oc_svm_preds)})
    df2 = pd.DataFrame({'Labels': np.ravel(y_test), 'Clusters': np.ravel(if_preds)})

    ct = pd.crosstab(df['Labels'], df['Clusters'])
    ct2 = pd.crosstab(df2['Labels'], df2['Clusters'])
    print(ct)
    print(ct2)
    print (classification_report(df['Clusters'], df['Labels'], target_names=['anomaly','normal']))
    print (classification_report(df2['Clusters'], df2['Labels'], target_names=['anomaly','normal']))

    conf_matrix = confusion_matrix(df.Clusters, df.Labels)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("One class SVM Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

    conf_matrix = confusion_matrix(df2.Clusters, df2.Labels)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Isolation forest Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
                    
    ##
    ##
    ##### Further compute accuracy, precision and recall for the two predictions sets obtained
    ##from sklearn.mixture import GaussianMixture
    ##from sklearn.isotonic import IsotonicRegression
    ##gmm_clf = GaussianMixture(covariance_type='spherical', n_components=18, max_iter=int(1e7))  # Obtained via grid search
    ##gmm_clf.fit(X_train)
    ##log_probs_val = gmm_clf.score_samples(X_test)
    ##isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    ##isotonic_regressor.fit(log_probs_val, y_test)  # y_val is for labels 0 - not food 1 - food (validation set)
    ##
    ### Obtaining results on the test set
    ##log_probs_test = gmm_clf.score_samples(X_test)
    ##test_probabilities = isotonic_regressor.predict(log_probs_test)
    ##test_predictions = [1 if prob >= 0.5 else -1 for prob in test_probabilities]
    ##
    ### Calculate accuracy metrics
    ##print ("IF accuracy: ", accuracy_score(y_test, test_predictions))
    ##
    ##df3 = pd.DataFrame({'Labels': np.ravel(y_test), 'Clusters': np.ravel(test_predictions)})
    ##
    ##ct = pd.crosstab(df3['Labels'], df3['Clusters'])
    ##
    ##print(ct3)
    ##
    ##print (classification_report(df3['Clusters'], df3['Labels'], target_names=['anomaly','normal']))
    ##

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
