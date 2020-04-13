# Scam detection of Craigslist post

Objective:
The main objective of this project is to help product managers at Craigslist achieve the goal of user satisfaction. 
To achieve this goal, it is necessary to flag a suspicious ad listing on the website and take necessary action against it. Our approach constitutes a mix of primary and secondary research for feature selection model formation for detecting scam posts.  


Framework:
We started by collecting data for training and test datasets from Craigslist.com for a list of exhaustive features derived from the secondary research. Post that, we eyeballed the training dataset to classify a listing as spam/ not spam based on the patterns concluded in various papers published studying the spam listings. The numerical features were then normalized to make a consolidated training dataset. This dataset was then used to build a spam classification model for the rental listings.

Data collection and analysis:

For the purpose of modeling, a training dataset was formed as a combination of content-based and numerical features.
Text analytics. A sample of 3000 titles and descriptions for various locations were eyeballed to identify spam patterns. As per the taxonomy of webspam, the following parameters were considered to flag a suspicious listing:
1)Asking for personal information
2)Fishy-looking e-mail address or domain
3)Beautiful unit for pennies on the dollar
4)No security deposit, a month’s free rent
5)Irrelevant or unrelated information
6)Poorly constructed sentences, excessive capitalization


Following NLP techniques are followed:
1.	Tokenization, lemmatization, removal of punctuations and stop-words
2.	TFIDF vectorization
3.	Apply the Naïve Bayes and Neural Networks model for text analytics
4. Image analytics:
Listings having images carrying text, phone numbers or email addresses are usually suspicious. This model uses TensorFlow to identify such images and flags them if any alphanumeric characters are spotted.
TensorFlow provides a collection of detection models pre-trained on the COCO dataset, the Kitti dataset, the Open Images dataset, the AVA v2.1 dataset, and the iNaturalist Species Detection Dataset. These models can be useful for out-of-the-box inference if some categories already have been in those datasets. They are also useful for initializing our models when training on novel datasets. In our case, we choose faster_rcnn_inception_v2_coco.
5. For the final text analytics model, the Neural networks model was selected as it resulted in higher validation accuracy.

Process:
1.	Setup: Set up all required software about TensorFlow and set Anaconda virtual environment.
2.	Gather and label pictures: We download over 300 images from craigslist which have text in their images. Through LabelImg package, draw a box around each object in each image and name the label.
3.	Generate training data: Generate the TFRecords that serve as input data to the TensorFlow training model. Our model uses the xml_to_csv.py and generate_tfrecord.py scripts from Dat Tran’s Raccoon Detector dataset, with some slight modifications to work with our directory structure.
4.	Training: Create a label map and configure training.
5.	Run the training: This process took about 10496 steps to get the loss lower than 0.05.
6.	Threshold: If the model recognition probability is higher than 0.6, we flag the image as spam.


Other features:
Apart from content-based features, we also considered various numerical and binary features for the final model based on secondary research.

Numerical:
1.	Price of the property 
2.	Number of bedrooms
3.	Number of bathrooms
4.	Word count of the title
5.	Word count of the description
6.	Number of images in the listing

Binary:
1.	Location: 1 if the neighborhood (location of the property) exists, else 0
2.	Phone number: 1 if there is a phone number in the description text, else 0
3.	Image analytics: 1 if the image is suspicious, else 0
Preparation of consolidated training dataset:
After building models for text and image analytics, they were applied to the final training dataset to get the numeric interpretation of the unstructured data. All the other numerical variables were normalized to remove the skewness in the data.
Normalization process:
To be mentioned
 
Data modeling:
Once the training dataset was developed, it was trained on various models. 
1.	Train-test split: The training dataset was split in a ratio of 70:30 for training and validation purposes.
2.	Models: The classification models such as Logistic regression, Support Vector Machine, Random Forest and Deep Learning were applied to the training dataset.


Model validation
The statistical scam online rental advertisement detection model developed needs to be validated to estimate the accuracy with which it will predict the test data. A cross-validation method was used on the training and validation datasets to evaluate the prediction of the classification of an independent sample.  We evaluated the models based on the accuracy and confusion matrix to decide upon the best model.
