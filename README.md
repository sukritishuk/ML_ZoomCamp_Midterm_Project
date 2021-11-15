## MLZoomCamp Midterm Project

# **Building & Deploying a Python ML Model for Stroke Predictions**

## (Deploying a Classification-based Model in AWS Cloud using Flask & Docker)
DataTalks.Club - [Machine Learning Zoomcamp](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html) **Midterm Project** - an initiative by Alexey Grigorev

Prepared by - [Sukriti Shukla](https://www.linkedin.com/in/sukriti-shukla-3989a819/)

![pexels-karolina-grabowska-4386467](https://user-images.githubusercontent.com/50409210/141792019-f6d135a3-3c15-43ae-9161-190785567d9c.jpg)

Photo by Karolina Grabowska from Pexels

## Introduction:

Stroke as a medical condition can cause severe health implications and even lead to the death of a person. If identified and treated within time, it can save one's life. There are a number of factors that can lead to strokes. The purpose of this project was to understand what are the factors that influence the incidents of strokes in patients. It was interesting to study a publicly available dataset in Kaggle and make use of Python Machine Learning techniques to build a model for predicting strokes incidents using Supervised Machine learning techniques.

Firstly, I prepared my data by cleaning and formatting the dataset. Then, I used Exploratory Data Analysis techniques in Python to visualize relationships between different factors in the dataset like BMI, Hypertension, Heart disease or Smoking etc. and their influence on the risk of strokes. I also analyzed the importance of features influencing stroke in patients.   Once my data was prepared, I started pre-processing it for use in different Supervised Machine Learning models. This involved splitting the data into subsets, identifying & segregating the feature matrix from the target variable and one-hot encoding of categorical variables in datasets. 

Since my dataset was used to predict strokes, I used Binary Classification models from Python to train on my datasets. After this, I trained multiple Binary Classification models both Regression-based & Tree-based. Each of the model was evaluated using classification metrics AUC score and their performances compared. Thereafter the models were tuned using different parameters to find the most optimal parameters. Lastly, multiple final models with optimal parameters were run to select the Best Model for making stroke predictions on this dataset.

Once the best model for my dataset was selected its code was exported from a Python Notebook into a Python script. Thereafter, this model was put into a web service using Flask. A Python virtual environment was created using Pipenv containing all the necessary dependencies and packages for running the model. Then the model was deployed locally using Docker as a containerized service. Then this locally deployed model was used to predict the risk of stroke for a new 'sample person' with unseen data as input. The model gave as output, details regarding the risk of stroke (as True or False) and probability of stroke for a 'sample person' as inputs to the model. 

Lastly, I also tried deploying the stroke prediction service to the cloud using AWS Elastic Beanstalk. This cloud environment created for the stroke prediction service was then used to make predictions for a new 'sample person' as input.



## Data Sources and Preparation: 

### Sources of Data -

For this project, I retrieved data from Kaggle Public Dataset - [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) Data was shared on Kaggle as a csv file (*healthcare-dataset-stroke-data.csv*).

### Data Processing Steps - 

Following steps were used in preparing the data for this project:-

* I processed the dataset using Python in-built functions and libraries such as **Pandas** for manipulation and structuring them as DataFrames.**Numpy** functions along with statistical analysis techniques like scaling were used to align the ranges of different factors. 
* Once the data was cleaned & formatted it was used to perform Exploratory Data Analysis (EDA) and create a series of visualizations for drawing insights about factors.
* Categorical features in the dataset like gender, work_type, Residence_type etc. were converted into dictionaries using one-hot encoder **DictVectorizer**.
* **Correlation** of numerical and categorical features was computed to understand their relation with stroke variable and inter-relationship among themselves.
* **Feature Importance** techniques were implemented for different models to understand which factors affected stroke more and influenced stroke predictions. 
* **Binary Classification Supervised Machine Learning algorithms** both regression-based (like Logistic Regression) and tree-based (like Decision Trees) were used from **scikit-learn** to train data and make predictions. **Ensemble methods** using bagging like Random Forest and boosting like eXtreme Gradient Boosting or XGBoost were also used. 
* Models were compared and their performance evaluated using metrics like - roc_auc_score, f1_score, classification_report etc. 
* Parameter of the models were tuned to improve their performance and select the most optimal parameters for each model.
* **Cross-Validation** techniques like **K-Fold** was used which split the dataset into ‘k’ number of subsets, then used k-1 subsets to train the model and used the last subset as a validation set to test the model. Then the score of the model on each fold was averaged to evaluate the performance of the model.
* To address the problem of Imbalanced classes in my dataset I oversampled the minority class using **Synthetic Minority Oversampling Technique (SMOTE)**. Data augmentation was done to oversample the minority class label (stroke=1 ie., those having a risk of stroke). Although, this technique was only used additionally on my dataset just to assess its impact. Different models were re-evaluated on this balanced dataset using AUC score to see any improvements or decline in their performance.
* Then after evaluating all the different models and their performances the best model was chosen as **XGBoost for Classification**. This model with its optimal parameter was used hereafter (exported from Python notebook into a Python script) in a Web service using **Flask** and deployed locally using **Docker**.
* Lastly, the Stroke Prediction service was also deployed in cloud using **AWS Elastic Beanstalk**. For this the Elastic Beanstalk command-line interface (CLI) was added as a development dependency only for the project. Then an environment for the stroke prediction service was created in AWS which successfully launched the environment and provided a public url to reach the application. This url was finally used to make stroke-related predictions about the new 'sample person' details provided as input. 



## Exploratory Data Analysis (EDA), Feature Correlation & Importance: 

Name of the Python Notebook - ***ML ZoomCamp - Midterm Project.ipynb***

Below are the important components from the Python Notebook used to load data, perform EDA, train multiple models and evaluate them to select the best model:-
* **Data Loading** - Firstly, I loaded all the basic libraries used in the project. Then, I imported the data from the .csv file into a Pandas DataFrame and got a snapshot of data (like exploring the unique values in each column of dataset, getting a statistical summary of dataset).

![image](https://user-images.githubusercontent.com/50409210/139657709-7c6ded3c-9b3a-4c35-86db-c4e1a33a67cb.png)

*  **Data Cleaning and Formatting** - I cleaned the dataset by formatting the column to lower case, imputing missing values with mean or mode of values and dropping irrelevant columns (like 'id'). I also separated numerical and categorical variable columns. 

* **Exploratory Data Analysis (EDA)** - Then, I performed EDA using Python libraries like Matplotlib, Pandas and Seaborn to analyze data and visualize key components to answer important questions like the following:-

      ***Who are at risk of a stroke? or What are the most highly correlated numerical or categorical features with stroke?***

![image](https://user-images.githubusercontent.com/50409210/139699132-65783fac-6657-42b5-a0b5-cbbaa38d6654.png)

A number of visualizations were made using Matplotlib and Seaborn plots like factorplot, bar chart, countplot, boxplot and violinplot to understand the relation of different factors like BMI, age, gender, residence_type, work_type with the stroke variable. 

The following **insights** were drawn from them - 
* People with a higher risk of having stroke, were self-employed, in private or government jobs. While those with no work experience and children had lower risk of a stroke. Also, there was a high association of work_type feature with age.
* People with a higher risk of stroke were more likely to be smokers or smoked formerly. They also tend to have higher average glucose levels. Also, this pattern can be seen in both the residential settings - Rural as well as Urban.
* People with a higher risk of stroke, irrespective of being male or female had slightly higher bmi levels. Although, there were large set of outliers in each case.
* People that ever married had a higher risk of having stroke. Also, people with stroke had higher mean age.
 
Correlation between numerical factors ('age','avg_glucose_level' and 'bmi') and the stroke variable was computed and a Heatmap was made to visualize it.  
![image](https://user-images.githubusercontent.com/50409210/139699291-e5d7de53-3819-44db-b817-05708bd8a367.png)

Correlation between categorical factors ('gender','ever_married', 'work_type', 'residence_type','smoking_status') and the stroke variable was also computed using Chi-square test. 

The following **insights** were drawn from them -
* Correlation between stroke and numerical columns showed that 'age' column had the highest correlation with stroke.
* Chi-square computations for categorical factors showed that 'gender' was not correlated to risk of stroke. For all other features, they were correlated with risk of stroke variable.

Regardless of patient’s gender, and where they stayed (Rural or Urban), they had the same likelihood to experience stroke.      
 
* **One-hot Encoding of Categorical Data** - As the stroke prediction dataset contained some categorical factors like work_type residence_type, gender etc. These were encoded using DictVectorizer to be used further in training different models and making predictions. DictVectorizer helped in transforming lists of feature-value mappings to vectors i.e., feature matrix into dictionaries for training and predicting on subsets. When feature values were strings, this transformer would do a binary one-hot coding. 

* **Feature Importance** - Here, I computed the Mutual information score for all feature columns. It was found that the avg_glucose_level was the most important, while gender was least important feature. I also computed the important features in the dataset for each of the models Decision tree, Random Forest and XGBoost so as to identify how they differed among models.



## Model Selection and Tuning & Evaluation:


* **Setting up the validation framework** - Firstly, I split the dataset into training, validation and testing subsets in the ratio of 60:20:20. Then, I defined the feature matrix containing all the factor columns and defined the 'stroke' column as the target variable. I also ensured that the target column was removed from each of the 3 data subsets.

* **Model Selection & Evaluation** - Once the data was split and pre-processed for machine learning algorithms I implemented different models by training them on the full_train set and made predictions on the validation set. The models were then evaluated using Classification metrics like roc_auc_score, f1_score etc. to compare their performances. 

Following were the different modelling algorithms I used in this project:

      * Logistic Regression
      * Decision Trees for Classification
      * Random Forest for Classification (using Ensemble learning, bagging)
      * XGBoost for Classification (using Gradient boosting)
      
For each of the model feature importance was done to identify which features contributed most to the predictions. AUC scores were also computed on the validation set.

![image](https://user-images.githubusercontent.com/50409210/139699504-2deae157-1877-4df0-b41b-887ac8923e79.png)

* **Parameter Tuning of Models** - The parameters for each of the above models were also tuned to find the most optimal parameters. 
Following were the parameters tuned for each model.
     * Decision Trees for Classification - selecting max_depth , min_samples_leaf and max_features
     * Random Forest for Classification - selecting n_estimators, max_depth, min_samples_leaf, max_features
     * XGBoost for Classification - selecting eta, max_depth, min_child_weight
 
After tuning the models with different parameters the final models from each type were selected with most optimal parameters. These were as follows:-

    * Final Decision Tree Model - max_depth = 5; min_samples_leaf = 10; max_features = 8
    * Final Random Forest Model - max_depth = 5; min_samples_leaf = 50; max_features = 10,; n_estimators = 180
    * Final XGBoost Model -(training for 200 iterations) - eta = 0.1; max_depth = 3; min_child_weight = 20

* **Selecting the Best Model** - Once final models were built next step was choosing between final Decision tree, Random forest and XGBoost for Classification models. This was done by evaluating each of the final models on the validation set and comparing the AUC scores. By doing so, I found that, ***XGBoost for Classification*** model gave the best AUC score on validation set hence, it was selected as the **best model for Stroke Prediction dataset**. 

Thereafter, I used this best model to predict on testing set (unseen data) where also it performed fairly close to the validation set scores. Finally, this best model was then saved as a Python script for further deployment as a web service.


## Dealing with Imbalanced Classes:

* **Balancing Imbalanced Classes** - As the stroke prediction dataset contained imbalanced classes additinally I tried to balance the dataset using SMOTE technique. This was used to oversample the minority class (stroke=1 i.e., those with a risk of having stroke). 

* **Evaluating using Balanced dataset** - The decision tree, random forest and XGBoost models were trained on oversampled dataset and predictions made on validation set. AUC scores were computed on all the three models to understand what effect did balancing of data labels have on the model predictions.Here too the XGBoost for Classification model outperformed the others.
   

## Exporting Best Model to Python script:

The code for best model i.e., XGBoost for Classification was first saved as a Python Notebook (Final Model Code.ipynb) then saved as a Python script (FinalModeltrain.py) for further deployment as a web service.

Name of Python script used - ***FinalModeltrain.py***

This above file would be used for training the final model and further deployment in a server. 

**Saving and Loading the Stroke Prediction Model** - 
* It contains all the necessary libraries & Python packages for the model like pandas, numpy, scikit-learn, seaborn etc. It contains the parameters used for the training the model. It also has steps about data preparation, data cleaning and formatting like the once we used in Python Notebook for Kaggle dataset. Then it lists the steps to create a validation framework (splitting dataset into 3 subsets, identifying feature matrix and target variables etc.). Thereafter, it performs one-hot encoding using DictVectorizer on data subsets, trains on training or validation subsets and finally lists steps for making predictions. It also performs KFold Cross-Validation on subsets before making predictions.

* After training, validation and making model ready for predictions it saves the model to a binary file using the **Pickle** library. This enables, to use the model in future without training and evaluating the code. Here, I have used pickle to make a binary file named ***model.bin***. It contains the one-hot encoder (DictVectorizer) and model details as an array in it.

![image](https://user-images.githubusercontent.com/50409210/139581819-0fe4351e-f48f-4c2c-910d-4945506bf1ba.png)

With unpacking the model and the DictVectorizer here, I would be able to predict the risk of stroke for any new input values (sample_person) or unseen data without training a new model by re-running the code.



## Putting the Model into a Web Service and Local Deployment using Docker: 

**Creating a Web service for Model using Flask** - 

Name of Python script used - ***FinalModelpredict.py***

* Here I use the **Flask** library in Python to create a web service. The script used here would be implementing the functionality of prediction to our stroke web service and would be making it usable in a development environment. To be able to use the model without running the code firstly, I open and load the previously saved binary file as shown below.

![image](https://user-images.githubusercontent.com/50409210/139581873-4b6058f3-05e6-405d-b2c0-cb0f6f4aa74f.png)

* Finally a function is used for creating the web service. We can run this code now to post a new person data and see the response of our model.

![image](https://user-images.githubusercontent.com/50409210/139582818-aed76cc8-a5ef-4727-994f-1cd83c8debc4.png)

The details of a new 'sample person' are provided in JSON format. These details are sent as a POST request to the web service. The web service sends back a response in JSON format which is converted back into a Python dictionary. Finally, a response message is printed based on the stroke decision provided by the model (threshold as 0.55 for stroke decision) for the new person.

Name of Python script used - ***FinalModelpredicttest.py***

![image](https://user-images.githubusercontent.com/50409210/139594219-9776d254-b845-475a-8057-ef0ee438bef6.png)

As shown above, I made a simple web server that predicts the risk of stroke for every new person. When I ran the app I got a warning that this server is not a WGSI server, hence not suitable for production environmnets. To fix this issue for my Windows machine, I used the library **waitress** to run the waitress WSGI server. Thus, this fix helped me make a production server that predicts the risk of stroke for new customers.


**Creating a Python virtual environment using Pipenv** - 

Names of files used - ***Pipfile*** and ***Pipfile.lock***

Virtual environments can help solve library version conflictions in each machine and manage the dependencies for production environments. I used the **Pipenv** library to create a virtual environment for my Stroke Prediction project. 

This was done using the following steps:-
* Firstly, I installed pipenv library using *pip install pipenv*. Then, I installed all the necessary libraries for my project in the new virtual environment like numpy, flask, pandas (also specifying exact versions in some cases) using pipenv command like, *pipenv install numpy sklearn==0.24.1 flask*. 

Pipenv command created two files named Pipfile and Pipfile.lock. Both these files contain library-related details, version names and hash files. If I want to run the project in another machine, I can easily install the libraries using command *pipenv install*, which would look into Pipfile and Pipfile.lock to install the libraries with specified version suitable for my project.

* After installing the required libraries I can run the project in the virtual environment with *pipenv shell* command. This will go to the virtual environment's shell and then any command I run will use the virtual environment's libraries.

* Next I installed and used the libraries such as waitress like before.


**Environment Management using Containerization in Docker** - 

Name of file used - ***Dockerfile***

Docker allows to separate or isolate my project from any system machine and enables running it smoothly on any machine using a container. To use Docker after installing Docker Desktop for Windows I had to perform the following steps:- 
* Firstly, I had to build a Docker image. I used a Python Docker image from the Docker website[https://hub.docker.com/_/python]. 

This Docker image file would have dependencies for my project. 
![image](https://user-images.githubusercontent.com/50409210/139591380-10ffb4c1-263f-4509-a1b4-993c0d7c67ed.png)

* After creating the Dockerfile and writing the settings in it, I built it and specified the tag name *zoomcamp-test* for this Dockerfile, using the command below:
     *docker build -t zoomcamp-test .*
* Thereafter, running the image once built and launching waitress service using command - *docker run -it --rm --entrypoint=bash zoomcamp-test*
* Mapping the port 5000 of the Docker to 5000 port of my machine for successful run of project app using Docker container using the below command:
     *docker run -it --rm -p 5000:5000 zoomcamp-test*
 
 This finally deployed my stroke prediction app inside a Docker container.



## Instructions for Local Deployment of Stroke Prediction Project:


In this section I have summarized the steps for running the Best model  (**XGBoost for Classification**) for my project after exporting it from Notebook to a script and deploying it locally as an app using Docker.

Below are the steps in this regard:-
* Changing the directory to the desired one using the command prompt terminal.
* Running the train script (*FinalModeltrain.py*) used for training the best model using command ----> *python FinalModeltrain.py* 
* Installing flask library using command -----> *pip install flask*
* Running the predict script (*FinalModelpredict.py*) used for loading the best model using command ----> *python FinalModelpredict.py*
* Installing waitress library (for Windows) using command ------> *pip install waitress*
* Telling waitress service where the stroke predict app is using command ----> *waitress-serve --listen=127.0.0.1:5000 FinalModelpredict:app*
* Running the script (in a new cmd terminal) with new sample person details (*FinalModelpredicttest.py*) for testing the best model after deployment 
  using command ------> *python FinalModelpredicttest.py*

  This would give the stroke prediction and probability of stroke for the new person (unseen data) input detail.

* Installing the pipenv library for creating virtual environment using command -----> *pip install pipenv* 
* Installing other dependent libraries for stroke prediction model using command -----> *pipenv install numpy scikit-learn==0.24.2 flask pandas requests xgboost*
* Installing python 3.8 version to match python version in the docker image and that in Pipfile using command -----> *pipenv install --python 3.8*
  
  This would update our Pipfile and Pipfile.lock with all library details.
  
* Next, getting into the newly created virtual environment using command ----> *pipenv shell*
* Now running the script with sample person details using command ----> *python FinalModelpredicttest.py*

  This launches pipenv shell first then runs waitress service. It results with predictions and probabilities from our model.

* Changing the directory to the desired one using a new cmd terminal. Downloading a desired Docker Python image (*3.8.12-slim*) using command ----> *docker run -it --rm   python:3.8.12-slim*
* Getting into this image using command ----> *docker run -it --rm --entrypoint=bash python:3.8.12-slim*
* Building the Docker image with tag *zoomcamp-test* (while still within the virtual env shell) using command -----> *docker build -t zoomcamp-test .*
* Running the image once built using command ----> *docker run -it --rm --entrypoint=bash zoomcamp-test*
  
* This brings us inside the app terminal where we get a list of files in app terminal using command -----> *ls*
* Then launching the waitress service within app terminal using commnad -----> *waitress-serve --listen=127.0.0.1:5000 FinalModelpredict:app*
* Thereafter, mapping the port in our container (5000) to the port in our host machine (5000) using command ----> *docker run -it --rm -p 5000:5000 zoomcamp-test*
  
  Both the above 2 steps would launch our waitress service giving below output:-
  
  *INFO: waitress serving on http://0.0.0.0:5000*
  
* Then in fresh cmd terminal changing the directory to the desired one and getting into the virtual environment created earlier using command ----> *pipenv shell*
* Finally, running the script with sample person details using command ------> *python FinalModelpredicttest.py*

This results in giving us stroke predictions and probabilities from our model for the new sample person as input. 

![image](https://user-images.githubusercontent.com/50409210/139918569-2944542e-6e11-41fd-add0-dd1f21bd6f30.png)

At last our Stroke Prediction project has been deployed locally using Docker container.



## Cloud Deployment of Stroke Prediction Service using AWS Elastic Beanstalk -

Once my Stroke prediction was deloyed locally I also tried to deploy it to the cloud using AWS Elastic Beanstalk. This was done using a special utility Elastic Beanstalk command-line interface (CLI).

Following steps were undertaken in this regard:-

* Firstly, the CLI written in Python, was installed and added as a development dependency (using pipenv) only for the project using command ---> *pipenv install awsebcli --dev*
* Then, I entered into the my virtual environment using command -----> *pipenv shell* 
* After this, I checked the version for Elastic Beanstalk available using CLI using commmand -----> *eb --version*
* Next, we initialize the Docker-based stroke-serving platform using command -----> *eb init -p docker stroke-serving*
* Now we first test our application locally sby specifying the port 5000 using command -----> *eb local run --port 5000*. It will first build an image and then run the container. 

* Then we make predictions about new 'sample customer' using command as earlier -----> *python FinalModelpredicttest.py*

![image](https://user-images.githubusercontent.com/50409210/139911782-c07e2ca9-30f6-4f91-9f42-cb50216e14a5.png)

* Now, we create the stroke-serving environment in AWS which sets up EC2 instance, applying auto-scaling rules using command -----> *eb create stroke-serving-env*

It creates the application and launches the ***stroke-serving environment***. It also generates a public url which can be used to reach the application and make predictions.

![image](https://user-images.githubusercontent.com/50409210/139920909-6616f3f1-af9e-4ae2-b879-f098b48ea3a0.png)


* Finally, we can test the cloud service to make predictions about new 'sample person' as input using commnad -----> *python FinalModelpredicttest.py* 

![image](https://user-images.githubusercontent.com/50409210/139912332-cf342652-d25d-499f-a57e-dd2b9831de0a.png)

It results in giving us stroke predictions and probabilities from our model for the new sample person (as shown below).

![image](https://user-images.githubusercontent.com/50409210/139912024-7dbd12e8-f08a-4196-8193-718ebb800772.png)  

Thus, our stroke-prediction service is deployed inside a container on AWS Elastic Beanstalk (as shown below). To reach it, we use its public URL.

![image](https://user-images.githubusercontent.com/50409210/139912524-47a45d6b-15f0-4060-9c4f-6e8a6d672337.png)









