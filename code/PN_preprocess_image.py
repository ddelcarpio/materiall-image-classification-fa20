
try:
    import pandas as pd
    import numpy as np
    from bs4 import BeautifulSoup

    import json
    import re
    import os
    import requests
    import ipynb

    from skimage import io
    import skimage
    import cv2
    from keras.utils import np_utils
except:
    print("If you're reading this, some required packages are not installed. Here is a list of the packages required:")
    
    print('opencv-python')
    print('keras')
    print('tensorflow')
    print('tensorflow-io')
    print('ipynb')
    print('pandas')
    print('numpy')
    
    raise ValueError()
    

def preprocess_image(image_urls_df, IMG_SIZE = 25):
    # Given a dataframe with two columns['image_urls', 'label'], reads image urls into pixels, 
    # converts them to grayscale, standardizes the size and shape and returns a numpy array
    # of the converted images ready to be plugged into a keras neural net
    
    # Converted images are of dimensions (num_images, grayscale_x, grayscale_y)
    # where num_images --> number of images in image_urls
    # grayscale_x, grayscale_y --> 2d grid where postion indicates the pixel position
    #                              and the number in that position is the standardized 
    #                              grayscale integer from 0 to 1
    
    # The IMG_SIZE constant is the length of a side in the square that is the 
    # converted image
    # ie the converted size of the images is 25X25 pixels using the default
    # value above
    
    df = image_urls_df.copy()
    df['image_url'] = df['image_url'].apply(lambda x: io.imread(x)).to_numpy()
    
    # Separating data into X and y while maintaining the right shape for the NN
    X = df.drop("label", axis = 1).to_numpy()
    
    # Thinking about including y in the output, but that might mess up the pipeline
    # y = df['label'].to_numpy()
    
    def convert(data):
        # Converts data to be constant size/shape/color 
        # Also put it in the right shape to be put into a neural net
        def change_img(image):
            image = skimage.color.rgb2gray(image)
            return cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
        return np.asarray([change_img(i) for i in data])
    
    
    X = np.asarray([convert(i) for i in X])
    
    # Normalizing data to be in range [0,1] as opposed to [0,255]
    X = X / 255
    
    
    return X
