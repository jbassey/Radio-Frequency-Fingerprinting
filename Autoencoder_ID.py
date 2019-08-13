import os
from util import *
from models2 import *
import argparse
from keras import backend as K
from keras.utils import to_categorical#processing labels
from keras.models import Sequential, load_model #model building
from keras.layers import Dense, Conv2D, Flatten #model buiding
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.optimizers import adam,sgd,adagrad,adadelta
from mpl_toolkits import mplot3d


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

def trainAE(path,snr,x_train, x_val,intr):#, i):
    timer = []
    inp_sz = int(np.sqrt(args.window))
    if args.model_name == 'autoencoder' or args.model_name =='deep_autoencoder':
        x_train = x_train.reshape(-1,args.window)
        x_val = x_val.reshape(-1,args.window)
    elif args.model_name == 'convolutional_autoencoder':
        x_train = x_train.reshape(-1,inp_sz,inp_sz,2)#
        x_val = x_val.reshape(-1,inp_sz,inp_sz,2)#
        print (x_train.shape, x_val.shape)
        
    # train each model and test their capabilities of anomaly deteciton
    model = load_models(args.model_name,x_train.shape)
    print(model.summary())

    # compile model
    model.compile(optimizer=adam(lr=0.001), loss=args.loss )
    #model.compile(optimizer=args.optimizer, loss=args.loss )
    start = time.time()
    print (x_train.shape, x_val.shape)

    # train on only normal training data
    history = model.fit(x=x_train,y=x_train,epochs=args.epochs,batch_size=args.batch_size)
    end = time.time()
    timer.append(end-start)
    model.save(snr+'/'+str(args.model_name)+'_w'+str(args.window)+'_'+snr+ 'intruder_device_'+str(intr)+'_Model.h5')
    scores = model.evaluate(x_val, x_val, verbose=0)
    print("scores: ",scores)#"%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model


def predictAE(window,snr,X_val, y_val,intr):
    inp_sz = int(np.sqrt(args.window))
    unique_elements, counts_elements = np.unique(y_val, return_counts=True)
    print(np.asarray((unique_elements, counts_elements)))
    if args.model_name == 'autoencoder' or args.model_name =='deep_autoencoder':
        X_val = X_val.reshape(len(X_val),args.window)
    elif args.model_name == 'convolutional_autoencoder':
        X_val = X_val.reshape(-1,inp_sz,inp_sz,2)

    loaded_model= load_model(snr+'/'+str(args.model_name)+'_w'+str(args.window)+'_'+snr+ 'intruder_device_'+str(intr)+'_Model.h5')
    print("Loaded model from disk")

    loaded_model.compile(optimizer=args.optimizer, loss=args.loss)
    losses = []
    for x in X_val:
        # compule loss for each test sample
        x = np.expand_dims(x, axis=0)
        loss = loaded_model.test_on_batch(x, x)
        losses.append(loss)
    
    # plot
    plt.plot(range(len(losses)), losses, linestyle='-', linewidth=1, label=str(args.model_name)+'_'+str(args.window)+'window_'+snr+'intruder_device_'+str(intr)+'_model')
    # delete model for saving memory
    #del loaded_model
    # create graph
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('sample index')
    plt.ylabel('loss')
    #plt.savefig(snr+'/'+str(args.model_name)+'_w'+str(args.window)+'_'+snr+'intruder_device_'+str(intr)+'_loss.png')
##    plt.clf()
    plt.show()
    return loaded_model.evaluate(X_val,X_val)

    
def AEtester(model_name, window,X,y,intr,snr):#threshold_fixed,intr):
    inp_sz = int(np.sqrt(args.window))
    #(X,y) = shuffle(X,y)
    model = load_model(snr+'/'+str(args.model_name)+'_w'+str(args.window)+'_'+snr+ 'intruder_device_'+str(intr)+'_Model.h5')
    if model_name == 'autoencoder' or model_name =='deep_autoencoder':
        #str(path)+'/'+str(args.model_name)+'Model.h5')
        X = X.reshape(len(X),args.window)
        test_x_predictions0 = model.predict(X)
        mse = np.mean(np.power(X - test_x_predictions0, 2), axis=1)
    elif model_name == 'convolutional_autoencoder':
        #model = load_model(str(path)+'/'+str(args.model_name)+'Model.h5')
        X = X.reshape(-1,inp_sz,inp_sz,2)
        test_x_predictions0 = model.predict(X).reshape(len(X),args.window*2)
        X = X.reshape(-1,args.window*2)
        mse = np.mean(np.power(X - test_x_predictions0, 2), axis=1)

    #mse = np.mean(np.power(X - test_x_predictions0, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,'True_class': np.ravel(y)})
    #print (error_df.describe())

    threshold_fixed = np.mean(error_df)[0]#thresh0,thresh1,thresh5,thresh10,thresh15 = 0.235,0.015,0.005,0.0020,0.0012
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    
    #Reconstruction error vs threshold check
    #threshold_fixed = np.mean(error_df)[0]#2.4
    groups = error_df.groupby('True_class')
    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                label= "Fraud" if name == 1 else "Normal")
    #ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for different classes ("+ str(args.model_name)+'_'+str(args.window)+'window_'+snr+ 'intruder_device_'+str(intr)+'_model)')
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    #plt.savefig(snr+'/'+str(args.model_name)+'_w'+str(args.window)+'_'+snr+'intruder_device_'+str(intr)+'_predictions.png')
    plt.show()  
    

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
    
##    ff = int((len(y_test)/6)*1)
##    gg = int((len(y_test)/6)*2)
##    print (np.unique(y_test[ff:gg]))
##    xx
    del(X0,y0)
    
    scaler = StandardScaler()

    if args.model_name == 'convolutional_autoencoder':
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
    elif args.model_name =='autoencoder'or args.model_name == 'deep_autoencoder':

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.fit_transform(X_val)
        X_test = scaler.transform(X_test)
        print (X_train.min(), X_train.max(), X_test.min(), X_test.max())
        

    
##    shuff = False#True
##    X_test, y_test = adapt_y(X_test, y_test,shuff,intr)
    
    AE_Train_accuracy = trainAE(path,snr,X_train,X_val,intr)
    AE_test_acc = predictAE(path,snr, X_test, y_test,intr)
    AEtester(args.model_name,args.window, X_test, y_test, intr,snr)
    #plot_output(args.model_name,args.window, X_test, y_test, intr,'00db')
    exit()
##    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
##    print(np.asarray((unique_elements, counts_elements)))
##    unique_elements, counts_elements = np.unique(y_val, return_counts=True)
##    print(np.asarray((unique_elements, counts_elements)))
##    unique_elements, counts_elements = np.unique(y_test, return_counts=True)
##    print(np.asarray((unique_elements, counts_elements)))    

##    thresholds = []
##    for label in labels:
##        ind = labels.index(label)
##        print (ind)model_name, window,X,y,threshold_fixed,intr
##        X_tra = X_train[np.where(np.in1d(single_y_train, np.array(ind)))[0]]
##        X_va = X_val[np.where(np.in1d(single_y_val, np.array(ind)))[0]]
##        X_tes = X_test[np.where(np.in1d(single_y_test, np.array(ind)))[0]]
##
##        y_tra = y_train[np.where(np.in1d(single_y_train, np.array(ind)))[0]]
##        y_va = y_val[np.where(np.in1d(single_y_val, np.array(ind)))[0]]
##        y_tes = y_test[np.where(np.in1d(single_y_test, np.array(ind)))[0]]
##        
##        print(X_tra.shape, np.unique(y_tra))#, np.unique(single_y_train))
##        print (X_va.shape, np.unique(y_va))#, np.unique(single_y_val))
##        print (X_tes.shape, np.unique(y_tes))#, np.unique(single_y_test))
##        scaler = MinMaxScaler()
##        X_tra = scaler.fit_transform(X_tra)
##        X_tes = scaler.fit_transform(X_tes)
##        X_va = scaler.fit_transform(X_va)
##        
##        AE_Train_accuracy = trainAE(label,X_tra,X_va)
##        AE_test_acc = predictAE(label, X_va, y_va)
##        thresholds.append(AE_test_acc)
##        AEtester(path,X_tes, y_tes)
##    print (thresholds)
##    xx
##    exit()              
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


