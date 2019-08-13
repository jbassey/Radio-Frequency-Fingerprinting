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

rcParams['figure.figsize'] = 14, 8.7 # Golden Mean
LABELS = ["0","1", "2", "3", "4", "5"]
col_list = ["cerulean","scarlet"]# https://xkcd.com/color/rgb/
sns.set(style='white', font_scale=1.75, palette=sns.xkcd_palette(col_list))


class clust():
    def load_data(self, filename):
        data = filename
        self.X = np.array(data)#[:,0:-1]#pd.DataFrame()
        self.target = label                
        
    def __init__(self, filename):
        self.filename = filename#self._load_data(sklearn_load_ds)
        
    def detect(self, output='replace'):
        #clustering = AffinityPropagation(preference=-50).fit(self.X)#af
        #clustering = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True).fit(self.X)
        clustering = DBSCAN(eps=3, min_samples=2).fit(self.X)#self.X_train)
        #clustering = AgglomerativeClustering().fit(self.X_train)
        #clustering = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        AMI = adjusted_mutual_info_score(np.ravel(self.target), np.ravel(clustering.labels_))
        RAND = adjusted_rand_score(np.ravel(self.target), np.ravel(clustering.labels_))
        print('AMI: {}'.format(adjusted_mutual_info_score(np.ravel(self.y_train), np.ravel(clustering.labels_))))
        print('Rand: {}'.format(adjusted_rand_score(np.ravel(self.y_train), np.ravel(clustering.labels_))))    
        return AMI, RAND#self
    
def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 5))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    
    return encoded

def y2indicator(y,k):
    N = len(y)
    ind = np.zeros((N, k))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p, t):
    return np.mean(p != t)


def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x
def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


def conv_net(x, keep_prob):
    #Define CNN operations
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[1, 3, 2, 256], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[1, 3, 256,128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[1, 3, 128, 64], mean=0, stddev=0.08))
    ##conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    
    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.relu(conv1)
    #conv1 = tf.nn.dropout(conv1, keep_prob)
    #conv1_pool = conv1#tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')#
    #conv1_bn = conv1_pool#tf.layers.batch_normalization(conv1_pool)
    #print conv1.shape
    
    # 3, 4
    conv2 = tf.nn.conv2d(conv1, conv2_filter, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.relu(conv2)
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    #conv2_pool = conv1#tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')#conv1#    
    #conv2_bn = conv2_pool#tf.layers.batch_normalization(conv2_pool)
    #print conv2.shape

    ### 5, 6
    conv3 = tf.nn.conv2d(conv2, conv3_filter, strides=[1,1,1,1], padding='VALID')
    conv3 = tf.nn.relu(conv3)
    #conv3 = tf.nn.dropout(conv3, keep_prob)
    #conv3_pool = conv3#tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    ##conv3_bn = tf.layers.batch_normalization(conv3_pool)
    print conv3.shape
    ### 7, 8
    ##conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    ##conv4 = tf.nn.relu(conv4)
    ##conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    ##conv4_bn = tf.layers.batch_normalization(conv4_pool)
    
    # 9
    flat = tf.contrib.layers.flatten(conv3)  
    print flat.shape
    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=64, activation_fn=tf.nn.relu)
    #full1 = tf.nn.dropout(full1, keep_prob)
    #full1 = tf.layers.batch_normalization(full1)
    full1 = tf.nn.dropout(full1, keep_prob)
    print full1.shape
    # 11
    #full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    #full2 = tf.nn.dropout(full2, keep_prob)
    #full2 = tf.layers.batch_normalization(full2)

    ### 12
    #full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    ##full3 = tf.nn.dropout(full3, keep_prob)
    ##full3 = tf.layers.batch_normalization(full3)    

    # 13
    #full4 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=1024, activation_fn=tf.nn.relu)
    #full4 = tf.nn.dropout(full4, keep_prob)
    #full4 = tf.layers.batch_normalization(full4)        

    # 14
    out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=5, activation_fn=None)
    return out,full1,flat

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

    Ytrain_ind = y2indicator(y_train,numberOfTX)
    Yval_ind = y2indicator(y_val,numberOfTX)
    Ytest_ind = y2indicator(y_test,numberOfTX)

    epoch = 20
    print_period = 4
    N = X_train.shape[0]
    BATCH_SIZE = 500
    n_batches = N // BATCH_SIZE
    keep_prob = 1#0.5

    x = tf.placeholder(tf.float32, shape=(None, 1, ix, 2), name='input_x')
    y =  tf.placeholder(tf.float32, shape=(None, 5), name='output_y')
    #keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    logits,full,flat = conv_net(x, keep_prob)
    

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=y
        )
    )

    #training = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(loss)
    training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # we'll use this to calculate the error rate
    #predict_op = tf.argmax(logits, 1)#tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    #t0 = datetime.now()
    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()
        for i in range(epoch):
            total_cost = 0
            for j in range(n_batches):
                batch_x = get_batch(X_train, j, BATCH_SIZE)#X_train[j*BATCH_SIZE:(j*BATCH_SIZE + BATCH_SIZE),]
                batch_y = get_batch(Ytrain_ind, j, BATCH_SIZE)#Ytrain_ind[j*BATCH_SIZE:(j*BATCH_SIZE + BATCH_SIZE),]

                _, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y})
                total_cost += batch_cost
                
            print("Epoch:", i, "\tCost:", total_cost)
            # predict validation accuracy after every epoch
            if i % print_period==0 or i==epoch-1:
                y_predicted = tf.nn.softmax(logits)
                correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
                accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
                accuracy_validation = accuracy_function.eval({x:X_val, y:Yval_ind})
                print("Validation Accuracy for window size:", ix,  "in Epoch ", i, ":", accuracy_validation)
##                        print("Epoch:", i, "\tCost:", total_cost)
##                        # predict validation accuracy after every epoch
##                        y_predicted = tf.nn.softmax(logits)
##                        correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
##                        accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
##                        accuracy_validation = accuracy_function.eval({x:xval, y:yvalind})
##                        print("Validation Accuracy in Epoch ", i, ":", accuracy_validation)

        end_time = time.time()
        print ("Training time ", end_time - start_time)

        lo = logits##tf.nn.softmax(logits)#
        ful = full
        feat = flat#
        
        logi = lo.eval({x:tf.cast(xval, 'float32').eval()})
        feats = feat.eval({x:tf.cast(xval, 'float32').eval()})
        fully = ful.eval({x:tf.cast(xval, 'float32').eval()})

                               
        df = pd.DataFrame(logi)
        df2 = pd.DataFrame(feats)
        df3 = pd.DataFrame(fully)
        df4 = pd.DataFrame(yval)
        
        df.to_csv("data/single/0/preSoftmaxID_0db_win128.csv")
        df2.to_csv("data/single/0/featuresID_0db_win128.csv")
        df3.to_csv("data/single/0/fullID_0db_win128.csv")
        df4.to_csv("data/single/0/labelID_0db_win128.csv")

    data = df#pd.read_csv('data/single/0/preSoftmaxID_0db_win128.csv', header=0, index_col=0)#_6classVal
    lab = df4#pd.read_csv('data/single/0/labelID_0db_win128.csv', header=0,index_col=0)#_6classVal

    data = np.array(data)#[:,0:-1]#.ravel()
    lab = np.array(lab)#[:,-1]
    AMI = []
    RAND = []
    timer = []
    ##stInd=0
    #t0=time.time()

    #for i in range(1):#zx in [1000]:#, 1100, 1300, 1600, 2000,
    samples , label = shuffle(data,lab)
    samples= samples[0:4000,:]
    label = np.ravel(label)[0:4000]
    start_time = time.time()

    ## Defining TSNE Model
    model = TSNE(learning_rate=100)
    #model = TSNE(n_components=3,perplexity=40,early_exaggeration=50, learning_rate=100)
    transformed = model.fit_transform(samples)

    #Cluster TSNE- selected features
    filename = samples#transformed
    c = clust(filename)
    c.load_data(filename)
    ami, rand = c.detect()

    end_time = time.time()
    full_time = end_time - start_time
    print ("Training time ", full_time)
    timer.append(full_time)
    AMI.append(ami)
    RAND.append(rand)



    total_time = sum(timer)/5
    print ("Time: ", total_time)
    print ("AMI: ", sum(AMI)/5)
    print ("RAND: ", sum(RAND)/5)


    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]
    plt.scatter(x_axis, y_axis, c=label)
    #plt.savefig('data/0_10/t_SNE_0_10db_win256_{}.png'.format(i))
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
