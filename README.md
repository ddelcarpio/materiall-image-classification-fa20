# materiall-image-classification-fa20
BRINGING SCIENCE TO THE ART OF HOME BUYING

This is the repository for the Fall 2020 IEOR 135: Data Science with Venture Applications (AKA [Data X](https://datax.berkeley.edu/) class at UC Berkeley. It is a semester long project to solve a Data Science related problem for a company or research related interests. We worked with [Materiall](https://materiall.com/) this semester to try and understand the driving features in the home buying market so that Materiall can improve its reccomendation engine. We started by scraping [Realtor.com] for home images with the intent to enventually determine the features within the images, such as an island within a kitchen. Once the webscraping was completed, we designed a few layers of models to get down to the objects within the images. The first iteration was whether the image was indoors or outdoors, then which room if it was indoors, and finally which features in the room given the room type. Due to the limitations of time and data, our team only completed up until the room classifier and did not get to the object detection within those images. Also, the basic decision tree also needs to be set up. 

There are README docs in each of the folders to describe their contents and each jupyter notebook is throughly documented.

