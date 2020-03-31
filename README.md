# 10DaysofMLChallenge
# This repository is for maintaining colab noteboooks for the #10DaysofMLChallenge and also to help beginners towards Deep Learning using Tensorflow
Notes: My implementations are just samples for you guys to work on
# Day 1:
Machine Learning Tools - Numpy, Pandas, Matplotlib 
- Numpy Documentation: https://numpy.org/devdocs/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/contents.html

Dataset link: https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_19-covid-Confirmed.csv&filename=time_series_2019-ncov-Confirmed.csv

Task: 
- Processed the data: On a particular date, If 70% number of confirmed case is zero, then Delete the column. i.e. whole February will be deleted and few more.
- Plot the graph
    - Country Wise
    - Date Wise
    - Continent Wise
 #10daysofMLChallenge, [23.03.20 10:06]
Day1:
Machine Learning Tools - Numpy, Pandas, Matplotlib 
- Numpy Documentation: https://numpy.org/devdocs/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/contents.html

Dataset link: https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_19-covid-Confirmed.csv&filename=time_series_2019-ncov-Confirmed.csv

Task: 
- Processed the data: On a particular date, If 70% number of confirmed case is zero, then Delete the column. i.e. whole February will be deleted and few more.
- Plot the graph
    - Country Wise
    - Date Wise
    - Continent Wise
    
Plot the data like:
1. Country-wise like (select any one date)
   x: country
   y: number of case
2. Date wise (select any country)
  x: Date
  y: number of case
3. Continent wise (select any date)
  x: Set of the country (Asia, Europe, America etc)
  y: number of case
  
 # Day 2:

Day 2: Real-world data are not simple integers and float; Hence we need to do some pre-processing and Feature Engineering! 

Topic: Feature Engineering & Pre-processing.

Resources:

1. "What Is Feature Engineering for Machine Learning?" Amit Shekhar https://medium.com/mindorks/what-is-feature-engineering-for-machine-learning-d8ba3158d97a
2. "ML Crash Course by Google" Google Developers https://developers.google.com/machine-learning/crash-course/representation/video-lecture
3. “Categorical Data” by Dipanjan (DJ) Sarkar https://link.medium.com/aI7JKJSY54
4. “Continuous Numeric Data” by Dipanjan (DJ) Sarkar https://link.medium.com/KAuqugRY54
5. “Ways to Detect and Remove the Outliers” by Natasha Sharma https://link.medium.com/Y89PjvVY54
6. "Introduction to Feature Engineering" Ali Mustufa https://colab.research.google.com/drive/1xMEqb5n1zfXJO8_lWQsovz6ubfXDmPdU#scrollTo=2TwkLcmqX-lW

Task:

Titanic dataset cleaning+Feature Engineering and visualization only (Apply what you learned in Day1) 
Data link: https://www.kaggle.com/c/titanic/data

Starter pack for beginners: https://colab.research.google.com/drive/18j97Ia-xlEKa9IWqEFt613CF1y4qgxVx

# Day 3:

We have already Learned about Data Preprocessing and Feature Engineering. Let's take a Step Ahead and Learn How to actually Build ML Models and Train them.

To start with here are some basic types of ML problems and some Resources:
1. ML | Types of Learning by GeeksforGeeks https://www.geeksforgeeks.org/ml-types-learning-supervised-learning/
2. Which Machine Learning model to use?: https://towardsdatascience.com/which-machine-learning-model-to-use-db5fdf37f3dd
3. Classification problem: https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
4. Regression Problem: https://towardsdatascience.com/solving-regression-problems-by-combining-statistical-learning-with-machine-learning-82949f7ac18a
5: Official Documentation: https://scikit-learn.org/stable/supervised_learning.html

Task:
Do the modeling for these Datasets:
a) Predict Loan_Status (http://iali.in/datasets/loan_status_train.csv)
b) Predict rating (http://iali.in/datasets/cereal.csv) 


For Advanced Users:
Analyze the Toxicity of comment's (Data
! wget https://www.dropbox.com/s/ggl9krhh6dcwhhz/train.csv
! wget https://www.dropbox.com/s/tst2y6mzwzbhxo3/test.csv)

# Day 4:

Unsupervised Learning

Introduction to Unsupervised Learning
https://algorithmia.com/blog/introduction-to-unsupervised-learning

“Unsupervised Learning and Data Clustering” by Sanatan Mishra https://link.medium.com/RUlFBArf94
“Unsupervised Learning: Dimensionality Reduction” by Victor Roman https://link.medium.com/eZStdrvf94

Dataset link: https://www.kaggle.com/c/expedia-personalized-sort/data

Task: Prepare a model on the above Dataset

# Day 5:

Introduction to CNN 
Resources:
1. Introduction to CNN- 
https://towardsdatascience.com/introduction-to-convolutional-neural-networks-cnn-with-tensorflow-57e2f4837e18

2. https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

3. https://machinelearningmastery.com/image-augmentation-deep-learning-keras/

4. https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/

Task: To classify dogs and cats.

Note: If you encounter overfitting, please use dropouts and data augmentation.

Dataset:
https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

# Day 6:
Topic: Imaging Problem 

Resources: 
1. Fashion MNIST with Keras and Deep Learning https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/

2. How to develop a CNN from scratch for fashion MNIST http://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/

3. Fashion MNIST: https://www.tensorflow.org/datasets/catalog/fashion_mnist

4. An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare: https://www.hindawi.com/journals/jhe/2019/4180949/

5. Deep Learning for Detecting Pneumonia from X-ray Images: https://towardsdatascience.com/deep-learning-for-detecting-pneumonia-from-x-ray-images-fc9a3d9fdba8


Data set:
For beginners: https://www.kaggle.com/zalando-research/fashionmnist

For advanced users: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# Day 7:

Topic: Natural Language Processing 

Concepts: 
1. Introduction to NLP: https://towardsdatascience.com/gentle-start-to-natural-language-processing-using-python-6e46c07addf3
2. NLTK 3.5b1 documentation: https://www.nltk.org/
3. NLP Tutorial by Sentdex- https://www.youtube.com/watch?v=FLZvOKSCkxY

Task:
For Beginners: Movie Review Sentiment Analysis
Dataset: https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

Resources
Simple Sentiment Analysis : https://dzone.com/articles/simple-sentiment-analysis-with-nlp
Sentiment Analysis of Movie Reviews: https://towardsdatascience.com/sentiment-analysis-a-how-to-guide-with-movie-reviews-9ae335e6bcb2



Task: NLP Advanced-Twitter Sentiment Analysis

Detect hate/racist speech in tweets

Plotting WorldCloud for hate words

Plot the graph for hate/racist tweets and non-racist/hate tweets

Using extracting features from cleaned tweets- Bag-of-Words/TF-IDF
     
How can you proceed:

-Clean the Data 
-Remove Twitter Handles @ from the data
-Remove Punctuations, Numbers, and Special Characters
-Remove Short Words
-Perform tokenization and stemming
-Understand common words  by plotting WorldcCloud


Dataset: 
https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech#train.csv

Resources: 
https://monkeylearn.com/blog/sentiment-analysis-of-twitter/

Generating WordCloud: https://www.geeksforgeeks.org/generating-word-cloud-python

Twitter Sentiment Analysis tutorial: https://pythonprogramming.net/twitter-sentiment-analysis-nltk-tutorial

Bag of words model: https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/?ref=rp

Tokenizing words and sentences- https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial

Introduction to Stemming- https://www.geeksforgeeks.org/introduction-to-stemming
 
# Day8:
Topic: Clustering Documents
What will you learn? 

Document clustering, also called text clustering, is a cluster analysis on textual documents. One of the typical usages would be document management.

Task: Clustering or grouping the documents based on the patterns and similarities.

Dataset: https://www.kaggle.com/dushyantv/consumer_complaints

How can you implement it?
1. Tokenization 
2. Stemming and lemmatization 
3. Removing stop words and punctuation 
4. Computing term frequencies or TF-IDF 
5. Clustering: K-means/Hierarchical; we can then use any of the clustering algorithms to cluster different documents based on the features we have generated
6. Evaluation and visualization: 
The clustering results can be visualized by plotting the clusters into a two-dimensional space (Plot clusters in 2D Graphs)

Resources: 
Basic visualization and clustering in Python:
https://www.kaggle.com/dhanyajothimani/basic-visualization-and-clustering-in-python

K-means clustering in Python: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

A Beginner’s Guide to Hierarchical Clustering: https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/



Topic: Implementing Multiclass Classification

What will you learn?
To understand how to do multiclass classification for text data in Python through solving Consumer complaint classifications for the finance industry.

Task: 
Classify Consumer Financial Protection Bureau complaints into the product category it belongs to using the description of the complaint

Goal of the task: 
The goal of the project is to classify the complaint into a specific product category. 
Since it has multiple categories, it becomes a multiclass classification that can be solved through many of the machine learning algorithms. 
Once the algorithm is in place, whenever there is a new complaint, we can easily categorize it and can then be redirected to the concerned person. 
This will save a lot of time because we are minimizing the human intervention to decide whom this complaint should go to. 
Use a visualization matrix to prepare actual vs predicted.

Dataset: https://www.kaggle.com/subhassing/exploring-consumer-complaint-data/data

Resources: 

Multiclass and MultiLabel Algorithm: https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format

Multi-Class Text Classification with Scikit-Learn:
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

Plot a confusion Matrix: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

# Day 9 and Day 10:

Topic: Machine Learning on the Fly using TF.js

Resources:

Check these links for some fun code that is easy to use:

Glitch: https://glitch.com/@TensorFlowJS

CodePen: https://codepen.io/topic/tensorflow/templates

Disappearing People: https://github.com/jasonmayes/Real-Time-Person-Removal

For learning more check out:

Deep Learning with JavaScript (covers everything):  https://livebook.manning.com/book/deep-learning-with-javascript/about-this-book/

Gant Laborde's (GDE possibly on this thread) online class is also a fun intro to lower level usage of Tensors:  https://academy.infinite.red/p/beginning-machine-learning-with-tensorflow-js/

Laurence Moroney's Coursera Course - goes more into training models and such with TF.js: https://www.coursera.org/learn/browser-based-models-tensorflow
by Jason Mayes.

Codelabs:
https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#0

Project: Feel free to make any, but the project is important!
