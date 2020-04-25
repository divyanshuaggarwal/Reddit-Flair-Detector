# Reddit Flair Detector

A Reddit Flair Detector web application to detect flairs of India subreddit posts using Machine Learning algorithms. The application can be found live at [Reddit Flair Detector](https://flair-detector.herokuapp.com/).

### Structure

The directory largely contains 2 ipython notebooks and 1 web directory:

1. [Data-Acquisition Notebook](https://github.com/divyanshuaggarwal/Reddit-Flair-Detector/blob/master/Reddit_Data_Collector.ipynb) contains all the code that was used to fetch data using PRAW API from reddit and adding it to the mongodb database using pymongo. It was the fetched back from the data base and pre-processed to add it into a CSV file to do the data analysis and build the machine learning model.
   
2. [Exploratory Data Analysis Notebook](https://github.com/divyanshuaggarwal/Reddit-Flair-Detector/blob/master/Exploratory_Data_Analysis.ipynb) contains all the code that was used to analyse and visualize the data.
    
3. [Flair-Detector Notebook](https://github.com/divyanshuaggarwal/Reddit-Flair-Detector/blob/master/Flair_Detection.ipynb) contains the code used to train various machine learning models and check the accuracy on different features.
 
4. [Website Directory](https://github.com/divyanshuaggarwal/Reddit-Flair-Detector/tree/master/website) the directory contains the flask implementation of the app,the requirements and procfile for heroku deployment. The detail of each file in the directory can be found in the readme.md file of the directory.
  
### Codebase

The entire code has been developed using Python programming language, utilizing it's powerful text processing and machine learning modules. The application has been developed using Flask web framework and hosted on Heroku web server.

### How to run the project:

  1. Open the `Terminal`.
  2. Clone the repository by entering `git clone https://github.com/divyanshuaggarwal/Reddit-Flair-Detector.git` and navigate into `website` directory by entering `cd website` in the terminal.
  3. Ensure that `Python3` and `pip` are installed on the system.
  4. Create a `virtualenv` by executing the following command: `virtualenv venv`.
  5. Activate the `venv` virtual environment by executing the follwing command: `source venv/bin/activate`.
  6. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  7. Now, execute the following command: `flask run` and it will point to the `localhost` server with the port `5000`.
  8. Enter the `IP Address: http://localhost:5000` on a web browser and use the application.
  
### Dependencies

The following dependencies can be found in [requirements.txt](https://github.com/divyanshuaggarwal/Reddit-Flair-Detector/blob/master/website/requirements.txt):

  1. [praw](https://praw.readthedocs.io/en/latest/)
  2. [scikit-learn](https://scikit-learn.org/)
  3. [nltk](https://www.nltk.org/)
  4. [Flask](https://palletsprojects.com/p/flask/)
  5. [Gensim](https://radimrehurek.com/gensim/)
  6. [pandas](https://pandas.pydata.org/)
  7. [numpy](http://www.numpy.org/)
  8. [scikit-learn](https://scikit-learn.org/stable/index.html)
  9. [gunicorn](https://gunicorn.org/)
  10. [XGBoost](https://xgboost.readthedocs.io/en/latest/)
  
### Approach

Going through various literatures available for text processing and suitable machine learning algorithms for text classification, I based my approach using [[2]](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568) which described various machine learning models like Naive-Bayes, Linear SVM and Logistic Regression for text classification with code snippets. Along with this, I tried other models like Random Forest and Multi-Layer Perceptron for the task. I have obtained test accuracies on various scenarios which can be found in the next section.

The approach taken for the task is as follows:

  1. Collect 100 India subreddit data for each of the 12 flairs using `praw` module [[1]](http://www.storybench.org/how-to-scrape-reddit-with-python/) and [2](http://machineloveus.com/mining-reddit-data-or-links-to-33-python-cheat-sheets/).
  2. The data includes *title, comments, body, url, author, score, id, time-created* and *number of comments*.
  3. For **comments**, only top level comments (top 10) are considered in dataset and no sub-comments are present.
  4. The ***title, comments*** and ***body*** are cleaned by removing bad symbols and stopwords using `nltk`.
  5. Five types of features are considered for the the given task: 
    a) Title
    b) Comments
    c) Urls
    d) Body
    e) Combining Title, Comments and Urls as one feature.
  6. The dataset is split into **80% train** and **20% test** data using `train-test-split` of `scikit-learn`.
  7. The dataset is then converted into a `Vector` and `TF-IDF` form.
  8. Then, the following ML algorithms (using `scikit-learn` libraries) are applied on the dataset:
    
    a) Naive-Bayes
    b) Linear Support Vector Machine
    c) Logistic Regression
    d) Random Forest
    e) MLP
    f) XGBoost
   9. Training and Testing on the dataset showed the **XGBoost** showed the best testing accuracy of **94.4%** when trained on the combination of **Title + Comments + URL** feature.
  
### Results

#### Title as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.8571428571      |
| Linear SVM                 | 0.8995535714      |
| Logistic Regression        | 0.8973214285      |
| Random Forest              | **0.904017**      |
| MLP                        | 0.8772321428      |
| XGBoost                    | 0.8080357142      |

#### Body as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.2678571428      |
| Linear SVM                 | 0.3906250000      |
| Logistic Regression        | **0.415178**      |
| Random Forest              | 0.4084821428      |
| MLP                        | 0.4107142857      |
|XGBoost                     | 0.4218750000      |

#### URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.6919642857      |
| Linear SVM                 | 0.810267857100    |
| Logistic Regression        | 0.82142857142     |
| Random Forest              | 0.8191964285      |
| MLP                        | **0.8281250000**  |
| XGBoost                    | 0.4799107142      |

#### Comments as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.7544642857      |
| Linear SVM                 | 0.8660714285      |
| Logistic Regression        | 0.8683035714      |
| Random Forest              | **0.8928571428**  |
| MLP                        | 0.8459821428      |
| XGBoost                    | 0.8147321428      |

#### Title + Comments + URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.8125000000      |
| Linear SVM                 | 0.9285714285      |
| Logistic Regression        | 0.933035714285714 |
| Random Forest              | 0.94196428571     |
| MLP                        | 0.8660714285      |
| XGBoost                    | **0.944196**      |

### Inferences 
 the tests shows that combined features i.e. Title + comments + URL shows the best accuracy while body shows the worst accuracy. Title as feature and comments as features are close runner ups, followed by URL. As machine learning models tries to detect specific words to identify the sentiment it makes sense because more the content means more information. Title as feature performing so well can be due to the fact the title consists of all the keywords to expect in the body, and comments can show a pattern on what topic the discussion is going on.


### References

#### 1. For data collection:
1. http://machineloveus.com/mining-reddit-data-or-links-to-33-python-cheat-sheets/
2. http://www.storybench.org/how-to-scrape-reddit-with-python/
3. https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
4. https://api.mongodb.com/python/current/tutorial.html

#### 2. For Building machine learning model:
1. https://medium.com/themlblog/splitting-csv-into-train-and-test-data-1407a063dd74
2. https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
3. https://medium.com/@robert.salgado/multiclass-text-classification-from-start-to-finish-f616a8642538
4. https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

#### 3.For Building the Website and Deploying it:
1. https://medium.com/techkylabs/getting-started-with-python-flask-framework-part-1-a4931ce0ea13 (entire series)
2. https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b
3. https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/
4. https://hackernoon.com/deploy-a-machine-learning-model-using-flask-da580f84e60c
5. https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7
