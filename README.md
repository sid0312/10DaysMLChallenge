# 10DaysofMLChallenge
# This repository is for maintaining colab noteboooks for the #10DaysofMLChallenge 
Notes: My implementations are just samples for you guys to work on
# Day1:
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

Day 5: Introduction to CNN 
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

 
 
