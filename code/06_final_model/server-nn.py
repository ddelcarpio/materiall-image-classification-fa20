# preprocessing
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
# modeling 
from skimage import io
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def prep_pixels(train, test):
    '''Normalizes training and test pixels
    Input: TRAIN and TEST image arrays
    Returns: Normalized TRAIN and TEST sets 
    '''
    train_norm = train.astype('float32') / 255.0
    test_norm = test.astype('float32') / 255.0
    return train_norm, test_norm
 


# plot diagnostic learning curves
def summarize_diagnostics(history, testX, testY):
    '''Plots diganostic learning curves of the model 
    Inputs: - HISTORY object generated during model fitting that maintains loss values and relevant metrics
            - TESTX and TESTY image arrays used for validating accuracy
    Returns: - Two graphs (cross entropy loss and validation accuracy) plotted over the number of epochs. 
               Results are saved in a diagnostics folder under the same directory. 
    '''
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    # save plot to file
    plt.savefig('diagnostics/vgg16_diagnostic_plot_vl.png')
    plt.close()



def load_binary_dataset():
    '''Loads in data from custom binary dataset.
    Note: The classes and probability percentage are pre-specified in this function. 
    Please change as necessary.
    '''
    df = pd.read_csv('combined_binary_dataset.csv')
    df = df[df['probability'] > 0.90]
    classes = ['inside', 'outside']
    print('Classes: ', classes)
    return df, classes



def load_one_dataset(folder_name):
    '''Loads in data from a specific folder.
    Note: The classes and probability percentage are pre-specified in this function. 
    Please change as necessary.
    '''
    # read in filelist and rename columns
    df = pd.read_csv(f'{folder_name}/filelist', sep=" ", header=None)
    df.columns = ['link', 'filepath', 'class', 'probability']
    # filter
    classes = ['living_room', 'house_view', 'kitchen', 'yard', 'garage']
    df = df[df['probability'] > 0.95]
    df = df[df['class'].isin(classes)]
    print('Classes: ', classes)
    return df, classes



def load_all_dataset():
    '''Loads in data from all three datasets instead of for a specific folder.
    Note: The classes and probability percentage are pre-specified in this function. 
    Please change as necessary.
    '''
    folders = ['ny_dataset', 'fremont_dataset', 'sa_dataset']
    all_df = pd.DataFrame()
    for folder_name in folders: 
        df = pd.read_csv(f'{folder_name}/filelist', sep=" ", header=None)
        df.columns = ['link', 'filepath', 'class', 'probability']
        df = df[df['probability'] > 0.90]
        all_df = all_df.append(df)
    # change the label of all the classes with less than 20 images into `unknown`
    all_classes = all_df['class'].value_counts()
    all_df.loc[all_df['class'].apply(lambda x: x in all_classes.index[all_classes < 20]).values, 'class'] = 'unknown'
    classes = all_df['class'].unique()
    print('Classes: ', classes)
    return all_df, classes




def load_server_dataset(folder_name, test_size=0.1, random_state=100, n=-1, shape=(100, 100, 3)):
    '''
    Given a folder name on Materiall's shared server:
    (1) reads in data files, 
    (2) convert them to pixels, 
    (3) separate them into train and testing data 
        (default test_size and random_state for reproducibility)
    Returns: X_train, X_test, y_train, y_test, shape of each image, num_of_classes
    '''
    if folder_name == 'all':
        df, classes = load_all_dataset()
    elif folder_name == 'binary':
        df, classes = load_binary_dataset()
    else: 
        df, classes = load_one_dataset(folder_name)

    # change dataset size if specified
    if n > 0:
        df = df.sample(n)
    print("Number of observations: " + str(df.shape[0]))

    # add pixels column, resize images (not proportional)
    df['pixels'] = df['link'].apply(lambda x: resize(io.imread(x), shape)) #might take a while

    #split and return training and test sets
    X = np.array([x for x in df['pixels']])
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # integer encode target labels so that keras can one hot
    label_encoder = LabelEncoder()
    vec_train = label_encoder.fit_transform(y_train)
    vec_test = label_encoder.fit_transform(y_test)

    # one hot encode target values
    y_train = to_categorical(vec_train)
    y_test = to_categorical(vec_test)

    #trainX, trainY, testX, testY 
    #(make sure ordering is correct to match harness function)
    return X_train, y_train, X_test, y_test, shape, classes



# define cnn model
def define_model(classes, shape=(32,32,3), lr=0.01, momentum=0.9, verbose=1):
    '''Defines the CNN model used to fit to the training data'''

    # grab the pre-trained VGG16 model, removing the top layers and changing the input shape
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=shape)
        
    # freeze pre-trained layers
    for layer in vgg_model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(vgg_model.layers[-1].output)
    class1 = Dense(512, activation='relu')(flat1)
    class2 = Dense(512, activation='relu')(class1)
    output = Dense(len(classes), activation='softmax')(class2)

    # define new model with top layers
    model = Model(inputs=vgg_model.inputs, outputs=output)
    opt = SGD(lr=lr, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # for output while training
    if verbose > 0:
        model.summary()

    return model


    
# MAIN() - runs the test harness for evaluating a model
def run_test_harness(folder_name, epochs=100, batch_size=64, 
                    verbose=2, test_size=0.1, random_state=100, 
                    n=-1, shape = (100, 100, 3), diagnostics = True, lr=0.05):
    '''
    The main() function used to load, fit, and evalute the model given the following params: 
    (1) FOLDER_NAME which specifies the type of model/data used (i.e. all, binary, specific folder) 
    (2) Optional hyperparamters used to fit the model
    Returns the MODEL created - progress and/or status of the model created/fitted will be printed/logged
    '''

    # load dataset
    trainX, trainY, testX, testY, shape, classes = load_server_dataset(folder_name, test_size, random_state, n, shape)
    print('Dataset Loaded!')
    # any pixel preprocessing will occur here

    # define model
    print('Defining Model...')
    model = define_model(classes, shape, lr=lr)

    # fit model
    print('Fitting Model...')
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=verbose)
    print('Model fitted! Epochs=%d, Batch Size=%d' % (epochs, batch_size))

    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=verbose)
    print('Model Evalution:')
    print('> %.3f' % (acc * 100.0))

    # learning curves
    if diagnostics:
        summarize_diagnostics(history, testX, testY)
    print('----FINISHED----')
    return model


