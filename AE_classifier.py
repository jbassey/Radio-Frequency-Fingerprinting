import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
##%matplotlib inline
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


from util import *
from models2 import *
import argparse

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
parser.add_argument('--window', type=int, default=64)
parser.add_argument('--trainProp', type=float, default= 0.7)
parser.add_argument('--size', type=int, default= 16)
parser.add_argument('--model_name', choices=['autoencoder', 'deep_autoencoder','convolutional_autoencoder'], default='convolutional_autoencoder')
args = parser.parse_args()

        
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
    


scaler = MinMaxScaler()

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

train_data = X_train.reshape(len(X_train),8,8,2).astype('float32')
val_data = X_val.reshape(len(X_val),8,8,2).astype('float32')
test_data = X_test.reshape(len(X_test),8,8,2).astype('float32')
print(np.max(train_data), np.max(test_data))

batch_size = 64
epochs = 20#200
inChannel = 2
x, y = 8, 8
input_img = Input(shape = (x, y, inChannel))
num_classes = 6

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):    
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder_train = autoencoder.fit(train_data, train_data, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_data, val_data))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(20)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
##plt.show()

autoencoder.save_weights('autoencoder.h5')

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_val)

# Display the change for category label using one-hot encoding
print('Original label:', y_train[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())

autoencoder.get_weights()[0][1]
full_model.get_weights()[0][1]

for layer in full_model.layers[0:19]:
    layer.trainable = False

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

classify_train = full_model.fit(train_data, train_Y_one_hot, batch_size=64,epochs=20,verbose=1,validation_data=(val_data, test_Y_one_hot))

full_model.save_weights('autoencoder_classification.h5')

for layer in full_model.layers[0:19]:
    layer.trainable = True

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

classify_train = full_model.fit(train_data, train_Y_one_hot, batch_size=64,epochs=20,verbose=1,validation_data=(val_data, test_Y_one_hot))

accuracy = classify_train.history['acc']
val_accuracy = classify_train.history['val_acc']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

####test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)
####print('Test loss:', test_eval[0])
####print('Test accuracy:', test_eval[1])
####predicted_classes = full_model.predict(test_data)
####predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
####predicted_classes.shape, test_labels.shape
####correct = np.where(predicted_classes==test_labels)[0]
####print "Found %d correct labels" % len(correct)
####for i, correct in enumerate(correct[:9]):
####    plt.subplot(3,3,i+1)
####    plt.imshow(test_data[correct].reshape(28,28), cmap='gray', interpolation='none')
####    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct]))
####    plt.tight_layout()
####incorrect = np.where(predicted_classes!=test_labels)[0]
####print "Found %d incorrect labels" % len(incorrect)
####for i, incorrect in enumerate(incorrect[:9]):
####    plt.subplot(3,3,i+1)
####    plt.imshow(test_data[incorrect].reshape(28,28), cmap='gray', interpolation='none')
####    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
####    plt.tight_layout()
####
####    from sklearn.metrics import classification_report
####target_names = ["Class {}".format(i) for i in range(num_classes)]
####print(classification_report(test_labels, predicted_classes, target_names=target_names))
##
