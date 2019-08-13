from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, LeakyReLU,Input,AveragePooling2D
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform

def autoencoder(inp_shape):
    input_shape=(inp_shape[-1],)
    model = Sequential()
    model.add(Dense(2,  activation='relu', input_shape=input_shape,kernel_initializer=glorot_uniform((input_shape))))#,use_bias=False))
    #model.add(Dropout(0.2))
    #kernel_regularizer=l2(0.00001)
    #model.add(layers.BatchNormalization())
    model.add(Dense(inp_shape[-1], activation='tanh'))
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
    return model

def deep_autoencoder(inp_shape):
    input_shape=(inp_shape,)
    model = Sequential()
    
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    kernel_regularizer=l2(0.00001)

    model.add(Dense(32, activation='relu'))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    kernel_regularizer=l2(0.00001)
    
    model.add(Dense(8, activation='relu'))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    kernel_regularizer=l2(0.00001)

    model.add(Dense(32, activation='relu'))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    kernel_regularizer=l2(0.00001)
    
    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    kernel_regularizer=l2(0.00001)
    
    model.add(Dense(inp_shape[-1], activation='tanh'))
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)

##    # Layer by layer pretraining Models
##    # Layer 1
##    input_img = Input(shape = (256, ))
##    distorted_input1 = Dropout(.1)(input_img)
##    encoded1 = Dense(128, activation = 'relu')(distorted_input1)
##    encoded1_bn = BatchNormalization()(encoded1)
##    decoded1 = Dense(256, activation = 'tanh')(encoded1_bn)
##
##    autoencoder1 = Model(input = input_img, output = decoded1)
##    encoder1 = Model(input = input_img, output = encoded1_bn)
##
##    # Layer 2
##    encoded1_input = Input(shape = (128,))
##    distorted_input2 = Dropout(.2)(encoded1_input)
##    encoded2 = Dense(64, activation = 'relu')(distorted_input2)
##    encoded2_bn = BatchNormalization()(encoded2)
##    decoded2 = Dense(128, activation = 'tanh')(encoded2_bn)
##
##    autoencoder2 = Model(input = encoded1_input, output = decoded2)
##    encoder2 = Model(input = encoded1_input, output = encoded2_bn)
##
##    # Layer 3 - which we won't end up fitting in the interest of time
##    encoded2_input = Input(shape = (64,))
##    distorted_input3 = Dropout(.3)(encoded2_input)
##    encoded3 = Dense(32, activation = 'relu')(distorted_input3)
##    encoded3_bn = BatchNormalization()(encoded3)
##    decoded3 = Dense(64, activation = 'tanh')(encoded3_bn)
##
##    autoencoder3 = Model(input = encoded2_input, output = decoded3)
##    encoder3 = Model(input = encoded2_input, output = encoded3_bn)
##
##    # Deep Autoencoder
##    encoded1_da = Dense(128, activation = 'relu')(input_img)
##    encoded1_da_bn = BatchNormalization()(encoded1_da)
##    encoded2_da = Dense(64, activation = 'relu')(encoded1_da_bn)
##    encoded2_da_bn = BatchNormalization()(encoded2_da)
##    encoded3_da = Dense(32, activation = 'relu')(encoded2_da_bn)
##    encoded3_da_bn = BatchNormalization()(encoded3_da)
##    decoded3_da = Dense(64, activation = 'relu')(encoded3_da_bn)
##    decoded2_da = Dense(128, activation = 'relu')(decoded3_da)
##    decoded1_da = Dense(256, activation = 'relu')(decoded2_da)
##
##    deep_autoencoder = Model(input = input_img, output = decoded1_da)
##
##    autoencoder1.compile(loss='mse', optimizer = 'adam')
##    autoencoder2.compile(loss='mse', optimizer = 'adam')
##    autoencoder3.compile(loss='mse', optimizer = 'adam')
##
##    encoder1.compile(loss='mse', optimizer = 'adam')
##    encoder2.compile(loss='mse', optimizer = 'adam')
##    encoder3.compile(loss='mse', optimizer = 'adam')

    return model

def convolutional_autoencoder(inp_shape):
    
    input_shape=inp_shape[1:]#(16,16,2)
    n_channels = input_shape[-1]
    model = Sequential()
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(AveragePooling2D(padding='same'))#MaxPool2D(padding='same'))
    #model.add(Dropout(0.3))
    #kernel_regularizer=l1(0.00001)
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(AveragePooling2D(padding='same'))#MaxPool2D(padding='same'))
    model.add(Dropout(0.3))
    #kernel_regularizer=l1(0.00001)
    model.add(Conv2D(2, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    #model.add(Dropout(0.3))
    #kernel_regularizer=l1(0.00001)
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    #model.add(Dropout(0.3))
    #kernel_regularizer=l1(0.00001)
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(n_channels, (3,3), activation='tanh', padding='same'))
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
    return model

def load_models(name, inp_shape):
    if name=='autoencoder':
        return autoencoder(inp_shape)
    elif name=='deep_autoencoder':
        return deep_autoencoder(inp_shape)
    elif name=='convolutional_autoencoder':
        return convolutional_autoencoder(inp_shape)
    else:
        raise ValueError('Unknown model name %s was given' % name)
