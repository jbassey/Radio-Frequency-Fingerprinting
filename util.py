import os,random
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import scipy.ndimage
from utilities import *
#from models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pandas as pd

from scipy.fftpack import fft
from scipy.signal import welch
import keras
##fft_size = 2048 # window size for the FFT
##step_size = fft_size/16 # distance to slide along the window (in time)
##spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
##lowcut = 500 # Hz # Low cut for our butter bandpass filter
##highcut = 15000 # Hz # High cut for our butter bandpass filter
### For mels
##n_mel_freq_components = 64 # number of mel frequency channels
##shorten_factor = 10 # how much should we compress the x-axis (time)
##start_freq = 300 # Hz # What frequency to start sampling our melS from 
##end_freq = 8000 # Hz # What frequency to stop sampling our melS from
from keras.preprocessing.image import ImageDataGenerator

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def get_data(filename, numberOfTX,windowSize,complexVector,cartesian,dim):
    stInd=0
    t0=time.time()
    fp = open(filename)
    lines = fp.readlines()
    RFTrace = np.array([[float(v) for v in line.split()] for line in lines])
    print (RFTrace.shape)
    for ix in windowSize:
        print ("/////////////////////////////newWindowSize/////////////////////////")
        print ("ix_windowSize = ", ix)
        #numberOfTX=6
        stC=stInd
        enC = len(RFTrace)
        columnone = np.array(RFTrace[:,0])  
        columntwo = np.array(RFTrace[:,1])
##        print (min(columnone),max(columnone), min(columntwo), max(columntwo))
##        xx

        if complexVector:
            miniBatchWindow=ix
            numberofminibatch=int(enC/miniBatchWindow)
            complexData = np.abs((columnone + 1j*columntwo))#.reshape(numberofminibatch,miniBatchWindow)
            #print (complexData[0], complexData.shape)
            X = complexData.reshape(numberofminibatch,miniBatchWindow)#complexData#
            del(complexData)
              
        else:            
            miniBatchWindow=ix
            numberofminibatch=int(enC/miniBatchWindow)
            if dim==1:
                aX = columnone.reshape(numberofminibatch,miniBatchWindow)
                bX = columntwo.reshape(numberofminibatch,miniBatchWindow)
                if cartesian:
                    X = np.array(np.concatenate((aX, bX), axis=1))
                else:
                    abso = np.abs(aX + 1j*bX)
                    angl = np.angle(aX + 1j*bX)
                    X = np.array(np.concatenate((abso, angl), axis=1))
                    #X = X.reshape((len(X), 2*ix))
            else:                
                aX = columnone.reshape(numberofminibatch,1,miniBatchWindow)
                bX = columntwo.reshape(numberofminibatch,1,miniBatchWindow)
                if cartesian:
                    X = np.array(np.concatenate((aX, bX), axis=1))
                    #X = X.reshape((len(X), 2, 1, ix)).transpose(0, 2, 3, 1)
                    X = X.reshape((len(X), ix,2))
                    print("yes")
                else:
                    abso = np.abs(aX + 1j*bX)
                    angl = np.angle(aX + 1j*bX)
                    X = np.array(np.concatenate((abso, angl), axis=1))
                    del(abso,angl,aX,bX)
                
                    #X = X.reshape((len(X), 2, ix)).transpose(0, 2, 1)#X.reshape((len(X), 2, 1, ix)).transpose(0, 2, 3, 1)
                    #X = X.reshape((len(X), 2, 1, ix)).transpose(0, 2, 3, 1)
                    #print (X.shape)
                

        #print(X.shape)
        labeleSize=int(numberofminibatch/numberOfTX)
        alabel=[]
        for i in range(numberOfTX):
            elabel = [int(i)]*labeleSize
            alabel = alabel + elabel
        label =np.array((alabel)).T
##    unique_elements, counts_elements = np.unique(label, return_counts=True)
##    print(np.asarray((unique_elements, counts_elements)), label.shape)
##    xx
    del(columnone,columntwo,elabel,alabel)
    return X,label



def data_split(X, y,trainProp, label):
    # Partition the data into training and test sets of the form we can train/test on 
    for zx in [1000]:#, 1100, 1300, 1600, 2000, 2500, 3500, 5000, 7500, 10000]:   
        #print ("WindowSize, batchSize, seed=", ix, jx, zx
        #print ("t_seed_"+str(zx)+"=",time.time())
        #np.random.seed(zx)
        n_examples = X.shape[0]
        print ("n_examples", n_examples)
        #X, y = shuffle(X,y)
        authorizedData = X[np.where(~np.in1d(y, np.array(label)))[0]]
        intruderData = X[np.where(np.in1d(y, np.array(label)))[0]]
        authorizedLabels = y[np.where(~np.in1d(y, np.array(label)))[0]]
        intruderLabels = y[np.where(np.in1d(y, np.array(label)))[0]]
        
        testProp = 1-trainProp
        X_trainAuth, X_valAuth, y_trainAuth, y_valAuth = train_test_split(authorizedData, authorizedLabels, test_size=testProp)#, random_state=42)
        X_valAuth, X_testAuth, y_valAuth, y_testAuth = train_test_split(X_valAuth, y_valAuth, test_size=0.5)#, random_state=42)


        lestlen = int(testProp*len(intruderData))
        X_train, y_train = (X_trainAuth, y_trainAuth)
        X_val, y_val = (X_valAuth, y_valAuth)
        X_test, y_test = (np.concatenate((X_testAuth,intruderData[0:lestlen]),axis=0), \
                         np.concatenate((y_testAuth,intruderLabels[0:lestlen]),axis=0))

        print (np.unique(y_train))
        print (np.unique(y_val))
        print (np.unique(y_test))
        

    return [X_train, y_train], [X_val, y_val], [X_test, y_test]

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


 
def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values
    


#Plot Spectrogram
def graph_spectrogram(data, rate, nfft, noverlap):
    #findDuration(data)#(wav_file)
    #rate, data = wavfile.read(wav_file)
    #print("")
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    #fig.savefig('sp_xyz.png', dpi=300, frameon='false')
    fig.canvas.draw()
    size_inches  = fig.get_size_inches()
    dpi          = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()

    #print(size_inches, dpi, width, height)
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #print("MPLImage Shape: ", np.shape(mplimage))
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    #plt.close(fig)
    #plt.show()
    return imarray

#Convert color image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#Normalize Gray colored image
def normalize_gray(array):
    return (array - array.min())/(array.max() - array.min())

# Function to find the duration of the wave file in seconds
def findDuration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        sw   = f.getsampwidth()
        chan = f.getnchannels()
        duration = frames / float(rate)
        #print("File:", fname, "--->",frames, rate, sw, chan)
        return duration
    
#Save created model
def save_model_to_disk(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

#Load saved model
def load_model_from_disk():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
def get_spectrograms(X,y):
    data_spectrograms = np.empty((0,256))
    spec_label = []
    for i in range(6):
        deviceData = X[np.where(np.in1d(y, i))[0]]#X[np.where(y ==i),:]#np.ravel()
        labelData = y[np.where(np.in1d(y, i))[0]]
        f, t, Sxx = signal.spectrogram(np.ravel(deviceData), 24000000000, nfft=256,return_onesided=False)
        spec_label = spec_label+[i]*Sxx.shape[1]
        data_spectrograms = np.concatenate((data_spectrograms,Sxx.T))
    spec_label = np.array(spec_label)
    print (data_spectrograms.shape)
    print (len(spec_label))

##    data_spectrograms, spec_label = shuffle(data_spectrograms, spec_label)
##    data_spectrograms = data_spectrograms.reshape((len(data_spectrograms),16,16,1))
##    ##x_train,x_test,y_train,y_test = train_test_split(data_spectrograms, spec_label, test_size=0.2, random_state=42)
    return data_spectrograms, spec_label

#spec = get_spectrograms(X,y)
def get_psd():
    data_psd = np.empty((0,256))
    psd_label = []
    for i in range(6):
        deviceData = X[np.where(np.in1d(y, i))[0]]#X[np.where(y ==i),:]#np.ravel()
        labelData = y[np.where(np.in1d(y, i))[0]]
        f, Sxx = signal.spectrogram(np.ravel(deviceData), 1000000, nfft=256,return_onesided=False)
        psd_label = psd_label+[i]*Sxx.shape[1]
        data_psd = np.concatenate((data_psd,Sxx.T))
    psd_label = np.array(psd_label)
    print (data_psd.shape)
    print (len(psd_label))

    data_psd, psd_label = shuffle(data_psd, psd_label)
    data_psd = data_psd.reshape((len(data_psd),16,16,1))
    ##x_train,x_test,y_train,y_test = train_test_split(data_spectrograms, spec_label, test_size=0.2, random_state=42)
    return data_psd, psd_label
    


def plot_specgram():
    f, t, Sxx = signal.spectrogram((columnone + 1j*columntwo), 1000000, nfft=1024)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.show()

    Pxx, freqs, bins, im = plt.specgram(columnone + 1j*columntwo, NFFT=1024, Fs=1000000)
    plt.title("PSD of 'signal' loaded from file")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    #plt.show()  # if you've done this right, you should see a fun surprise here!
    print (Pxx.shape)
    pass

def plot_psd():
    f, Pxx_den = signal.periodogram((columnone + 1j*columntwo), 1000000, nfft=1024)
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    #plt.show()
    print (f.shape)
    # Let's try a PSD plot of the same data
    Pxx, im = plt.psd(columnone + 1j*columntwo, NFFT=1024, Fs=1000000)
    plt.title("PSD of 'signal' loaded from file")
    #plt.show()
    pass # return []



##print('x_train shape:', x_train.shape)
##print(x_train.shape, 'train samples')
##print(x_test.shape, 'test samples')
##
##
##batch_size = 64
##num_classes = 6
##epochs = 20
##data_augmentation = True
##num_predictions = 20
##save_dir = os.path.join(os.getcwd(), 'saved_models')
##model_name = 'keras_cifar10_trained_model.h5'
##
### Convert class vectors to binary class matrices.
##y_train = keras.utils.to_categorical(y_train, num_classes)
##y_test = keras.utils.to_categorical(y_test, num_classes)
##
##model = get_model()
##x_train = x_train.astype('float32')
##x_test = x_test.astype('float32')
##
###if not data_augmentation:
##print('Not using data augmentation.')
##model.fit(x_train, y_train,
##          batch_size=batch_size,
##          epochs=epochs,
##          validation_data=(x_test, y_test),
##          shuffle=True)
##
##
### Score trained model.
##scores = model.evaluate(x_test, y_test, verbose=1)
##print('Test loss:', scores[0])
##print('Test accuracy:', scores[1])


    




