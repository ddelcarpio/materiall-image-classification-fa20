# Imagine Classification Models for Materiall


This is the repository for the Fall 2020 IEOR 135: Data Science with Venture Applications (also known as [Data X](https://datax.berkeley.edu/)) class at UC Berkeley. It is a semester long project to solve a data science related problem for a company or research related interests. We worked with [Materiall](https://materiall.com/) this semester to try and understand the driving features in the home buying market so that Materiall can improve its recommendation engine. The repository is setup as follows:
 - Code
 - Data
 - Documentation
 - Images

## Problem Statement: 


The specific image classification models to be implemented can be understood in steps. As a first pass, students will implement an image classification model that can predict the room in which the image was taken. Students will have the ability to train their models first on a set of labeled images and then deployed and tested. An important component of the first step is the identification of a suitable training set within the catalogue of San Francisco Bay Area homes. The next problem will focus on object detection per room type. The idea here is to detect important attributes of a home which may already be catalogued (such as number of bedrooms, number of bathrooms, etc.,) as well as other attributes which may not be listed but inferred from the objects detected (for example lighting in a room may be a function of the number of windows detected or kitchen space could be a function of the number of distinct objects present in the image). Finally, as a last step we will aim to build an image classifier for the extracted attributes. 


## Learning Path:

We started by scraping [realtor.com](https://www.realtor.com/) for home images with the intent to eventually determine the features within the images, such as an island within a kitchen. To solve the first step of room classification, we created a pipeline that took in the scraped image urls, preprocessed the data, defined a convolutional neural net, and trained and tested the CNN to classify the images. We divided this first step into two passes. The first pass classified whether the image was indoors or outdoors. Then, the second pass would classify indoor rooms into specific rooms, e.g. kitchen, bathroom. Due to the limitation of time, our team only completed up until the room classifier and did not get to the object detection within those images. Now that we have successfully classified the rooms, the next step is work on specific image detection. One possible path would be to use the ADE20K Scene parser to segment features in the images, and perform object detection on the segmented images. This way, we can find specific things like islands, cabinets and other features in a home. The end goal would be to use these models for Materiall’s home recommendation system where a homebuyer could input specific home features and search for homes with those specified features.

 
 

