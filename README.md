# README #


This project generates captions of images using CNN and LSTMs.

The image features will be extracted from Xception model which is a CNN model pre trained on the ImageNet dataset and then we feed the features into a LSTM model which will be responsible for generating the image captions.

Dataset:  Flickr8k
Consists of 8000 images. (1GB)


## Libraries Used ##
1. Python: Most popular language that suits for ML
2. TensorFlow and Keras: For preprocessing input, loading images, tokenizer, embedder, dropout, load models - LSTM
3. Xception:To use CNN Pre trained on ImageNet.
4. Pillow: For basic image processing functions.
5. NumPy:  Storing pixels as integers in an array, expanding dimensions.
6. Pandas: Used for data manipulation and storing and retrieving files.
7. Pickle: Storing, saving and loading model.
8. Matplotlib: Plot graphs.
9. Google Colab: Easy to configure interactive environment to share and run code, with free GPU access.
10. Google Drive: To make large storage dataset available amongst team.



## Link to dataset ##
https://drive.google.com/drive/folders/1mTAqh80sZGBnl3b9KBrDev62fO5BR8ux?usp=sharing


## Project File Structure ##
Downloaded from dataset:
1. <b> Flicker8k_Dataset </b> – Dataset folder which contains 8091 images.
2. <b>  Flickr_8k_text </b> – Dataset folder which contains text files and captions of images.
3. <b> Flickr_8k_text folder/Flickr8k.token.txt </b>- the raw captions of the Flickr8k Dataset . The first column is the ID of the caption which is "image address # caption number"
4. <b> Flickr_8k.testImages.txt </b> contains the filenames of the test images.
5. <b> Flickr_8k.trainImages.txt</b>  contains the filenames of the train images.
 
The below files have been created by us while making the project (Can see in  https://drive.google.com/drive/folders/1mTAqh80sZGBnl3b9KBrDev62fO5BR8ux?usp=share_link ).
1. Models – It will contain our trained models.
2. Descriptions.txt – This text file contains all image names and their captions after preprocessing.
3. Features.p – Pickle object that contains an image and their feature vector extracted from the Xception pre-trained CNN model.
4. Tokenizer.p – Contains tokens mapped with an index value. <br>

Step - wise explanation
1. First, we import all the necessary packages - string, numpy, pandas, pillow,os, pickle, keras, and tensorflow package
2. Getting and performing data cleaning
Description of the functions used
load_doc( filename ) – For loading the document file and reading the contents inside the file into a string.
all_img_captions( filename ) – This function will create a description dictionary that maps images with a list of 5 captions. 
cleaning_text( descriptions) – This function takes all descriptions and performs data cleaning. This is an important step when we work with textual data. According to our goal, we decide what type of cleaning we want to perform on the text. In our case, we removed punctuations, converted all text to lowercase and removed words that contain numbers.
So, a caption like “A man riding on a three-wheeled wheelchair” will be transformed into “man riding on three wheeled wheelchair”
text_vocabulary( descriptions ) – This is a simple function that will separate all the unique words and create the vocabulary from all the descriptions.
save_descriptions( descriptions, filename ) – This function will create a list of all the descriptions that have been preprocessed and store them into a file. We will create a descriptions.txt file to store all the captions.
3. Extracting the feature vector from all images 
We used the Xception model which has been trained on the imagenet dataset that had 1000 different classes to classify. We imported this model from the keras.applications. Xception model took 299*299*3 image size as input. We removed the last classification layer and got the 2048 feature vector.
model = Xception( include_top=False, pooling=’avg’ )
The function extract_features() will extract features for all images and we will map image names with their respective feature array. Then we dumped the features dictionary into a “features.p” pickle file.
4. Loading dataset for Training the model
In our Flickr_8k_test folder in the drive, we have the Flickr_8k.trainImages.txt file that contains a list of 6000 image names that we used for training.
For loading the training dataset, we used the below functions:
load_photos( filename ) – This will load the text file in a string and will return the list of image names.
load_clean_descriptions( filename, photos ) – This function will create a dictionary that contains captions for each photo from the list of photos. We also appended the <start> and <end> identifier for each caption. We did this so that our LSTM model could identify the starting and ending of the caption.
load_features(photos) – This function will give us the dictionary for image names and their feature vector which we have previously extracted from the Xception model.
5. Tokenizing the vocabulary 
We mapped each word of the vocabulary with a unique index value. Keras library provided us with the tokenizer function that we used to create tokens from our vocabulary and saved them to a “tokenizer.p” pickle file.
Our vocabulary contains 7577 words. We calculated the maximum length of the descriptions. This was important for deciding the model structure parameters. Max_length of description is 32.
 
6. Create Data generator
We trained our model on 6000 images and each image contained a 2048 length feature vector and the caption was also represented as numbers. This amount of data for 6000 images was not possible to be held into memory so we used a generator method that yields batches.
For example:
The input to our model is [x1, x2] and the output will be y, where x1 is the 2048 feature vector of that image, x2 is the input text sequence and y is the output text sequence that the model has to predict.
7. CNN-LSTM model
CNN-LSTM model consists of the following:
Feature Extractor – The feature extracted from the image has a size of 2048, with a dense layer, reducing the dimensions to 256 nodes.
Sequence Processor – An embedding layer that handles the textual input, followed by the LSTM layer.
Decoder – By merging the output from the above two layers, we processed using the dense layer to make the final prediction. The final layer contains the number of nodes equal to vocabulary size.
The output sequence in batches is fitted in the model using model.fit_generator() method. Model is saved in the following folder: (https://drive.google.com/drive/u/0/folders/1kRWfNi8xpTbSMXa0SGjmNsjX2jXLxp1z)
9. Testing
All test images(1000) were tested and their bleu score was calculated.
First the image was read, it’s features were extracted and caption was generated using it. There are 5 actual descriptions for the corresponding image and they were passed as reference and the model generated a caption as a candidate to the sentence_bleu() function.
BLEU-1: 0.367548 for 1-gram with the following weights(1.0, 0, 0, 0)
BLEU-2: 0.191113 for 2 gram with the following weights(0.5, 0.5, 0, 0)
BLEU-3: 0.121849 for 3 gram with the following weights(0.3, 0.3, 0.3, 0)
BLEU-4: 0.049862 for 4 gram with the following weights (0.25, 0.25, 0.25, 0.25)
