## MLZoomCamp Midterm Project

# **Building & Deploying a Python ML Model for Stroke Predictions**

Prepared by - Sukriti Shukla

## Introduction 

Stroke as a medical condition can cause severe health implications and even lead to the death of a person. If identified and treated within time, it save one's life. There are a number of factors that can lead to strokes. The purpose of this project was to understand what are the factors that influence the incidence of strokes in patients. It was interesting to study a publicly available dataset in Kaggle and make use of Python Machine Learning techniques to build a model for predicting strokes incidents using Supervised Machine learning techniques.

Firstly, I prepared my data by cleaning and formatting the dataset. Then, I used Exploratory Data Analysis techniques in Python to visualize relationships between different factors in the dataset like BMI, Hypertension, Alcoholism or Smoking etc. and their influence on risk of strokes. I also analyzed important features influencing strokes in patients.   

Once my data was prepared, I started pre-processing it for use in different Supervised Machine Learning models. This involved splitting the data into subsets, identifying & segregating the feature matrix from the target variable and one-hot encoding of categorical variables in datasets. Since my dataset was used to predict strokes, I used Binary Classification models from Python to train on my datasets. After this, I trained multiple Binary Classification models both Regression-based & Tree-based. Each of the model was evaluated using classification metrics AUC score and their performances compared. Thereafter the models were tuned using different parameters to find the most optimal parameters.
Multiple Final models with optimal parameters were run to select the Best Model for making stroke predictions.


Once the best model for my dataset was selected its code was exported from a Python Notebook into a Python script. Thereafter, this best model was put into a web service using Flask. A Python virtual environment was created using Pipenv. this virtual environment contained all the necessary dependencies and packages for running the model. Then the model was deployed locally with Docker into a containerized service.  

Finally, this locally deployed model was used to predict the risk of stroke for a 'sample person' with unseen feature data. The model gave as output details about the incidence of stroke or not and the probability of stroke for this 'sample person' based on the predictions made by the model deployed.


## Data Sources and Preparation: 

### Sources of Data -

For this project, I retrieved data from Kaggle Public Dataset - [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) Data shared on Kaggle as a csv file (*healthcare-dataset-stroke-data.csv*).


### Data Processing Steps - 

* I processed the dataset using Python in-built functions and libraries such as **Pandas** for manipulation and structuring them as DataFrames.
* **Numpy** functions along with statistical analysis techniques like scaling were used to align the ranges of different factors. 
* Once the data was cleaned & formatted it was used to perform Exploratory Data Analysis (EDA) and create a series of visualizations for drawing insights about factors.
* Categorical features in the dataset like gender, work_type, Residence_type etc. were converted one-hot encoded using **DictVectorizer**.
* Correlation of numeric features and categorical features was computed to understand which factors influenced the risk of stroke.
* **Feature Importance** techniques were implemented for different models to understand which factors affected stroke and influenced stroke predictions. 
* **Binary Classification Supervised Machine Learning algorithms** both regression-based and tree-based in **scikit-learn** were used to fit and make predictions about stroke   incidents.
* I used **Regression-based models** like Logistic Regression and **Tree-based models** like Decision Trees, Random Forests and XGBoost to make predictions. 
* Different models were compared and evaluated using AUC score metrics. Parameter of models were tuned to improve their performance using **Cross-Validation** techniques like K-Fold.
* As my dataset had **Imbalanced classes** I used **SMOTE analysis** technique to remove the class imbalance and oversample the minority class label (stroke=1 ie., those having a risk of stroke). The model was additionally evaluated on this re-balanced dataset also using AUC score just to see any improvements in performance.
* Based on the AUC scores computed from each model the best model was chosen as **XGBoost for Classification**. This model with its optimal parameter was exported from Python notebook into a Python script to be used for further use in a Web service through ** Flask** and local deployment using **Docker container service**.



### Details about Python Notebook - 
Name of the Python Notebook - ML ZoomCamp - Midterm Project.ipynb

* **Data Loading** - Firstly, I loaded all the basic libraries used in the project. Then, I imported the data from the .csv file into a Pandas DataFrame and got a snapshot of data (like exploring the unique values in each column of dataset, getting a statistical summary of dataset).
*  **Data Cleaning and Formatting** - I cleaned the dataset by formatting the column to lower case, imputing missing values with mean or mode of values and dropping irrelevant columns (like 'id'). I also separating numerical and categorical variable columns. 

*  **Exploratory Data Analysis (EDA)** - Then, I performed EDA using Python libraries like Matplotlib, Pandas and Seaborn to analyze data and visualize key components to answer important questions like the following:-
      ****Who are at risk of a stroke? 
      ****What are the most highly correlated numerical or categorical features with stroke?
    
A number of visualizations were made using Matplotlib and Seaborn like factorplot, bar chart, countplot, boxplot and violinplot to understand the relation of different factors like BMI, age, gender, residence_type, work_type with the stroke variable.  
The following **insights** were drawn - 
      * People with a higher risk of having stroke, were self-employed, in private or government jobs. While those with no work experience and children had lower risk of a stroke. Also, there was a high association of work_type feature with age.
      * People with a higher risk of stroke are more likely to be smokers or smoked formerly. They also tend to have higher average glucose levels. Also, this pattern can be seen in both the residential settings - Rural as well as Urban.
      * People who have been diagnosed with hypertension & heart disease, have a higher risk of having stroke.
      * People with a higher risk of stroke, irrespective of being male or female had slightly higher bmi levels. Although, there were large set of outliers in each case.
      * People that ever married had a higher risk of having stroke. Also, people with stroke have higher mean age.
 
Correlation between numerical factors ('age','avg_glucose_level' and 'bmi') and the stroke variable was computed and a Heatmap was made to visualize it.  
Correlation between categorical factors ('gender','ever_married', 'work_type', 'residence_type','smoking_status') and the stroke variable was also computed using Chi-square test. 
The following **insights** were drawn -
      * From the above computation of correlation between stroke and numerical columns we saw that 'age' column had the highest correlation with stroke.
      * From the Chi-square computations for categorical factors it was found that 'gender' was not correlated to risk of stroke. For all other features, they were correlated with risk of stroke variable.
      * Regardless of patient’s gender, and where they stayed (Rural or Urban), they had the same likelihood to experience stroke.
      
 
* **One-hot Encoding of Categorical Data** - As the stroke prediction dataset contained some categorical factors like work_typem residence_type, gender etc. These were encoded using DictVectorizer to be used further in training different ML models and maing predictions. DictVectorizer helped in transforming lists of feature-value mappings to vectors i.e., feature matrix into dictionaries for training and predicting on subsets. When feature values were strings, this transformer will do a binary one-hot coding. 


* **Setting up the validation framework** - Firstly, I split the dataset into training, validation and testing subsets in the ratio of 60:20:20. Then, I defined the feature matrix containing all the factor columns and defined the 'stroke' column as target column. I also ensured that the target column was removed from each of the 3 data subsets.
* **Feature Importance** - Here, I computed the Mutual information score for all feature columns. It was found that the avg_glucose_level was the most important, while gender was least important feature.

* **Modelling on our dataset** - Once the data was split and pre-processed for machine learning algorithms I implemented different models by training them on the full_train set and predicted on the validation set. The models were evaluated using Classification models and AUC score metrics was used to compare performance. Following were the models used:
      * Logistic Regression
      * Decision Trees for Classification
      * Random Forest for Classification (using Ensemble learning)
      * XGBoost for Classification (using Gradient boosting)

For each of the model feature importance was done to identify which features contributed most to the predictions. AUC scores were also computed on the validation set.

* **Parameter Tuning of Models** - The parameters for each of the above models were also tuned to find the most optimal parameters. Following were the parameters tuned for each model.
     * Decision Trees for Classification - selecting max_depth , min_samples_leaf and max_features
     * Random Forest for Classification - selecting n_estimators, max_depth, min_samples_leaf, max_features
     * XGBoost for Classification - selecting eta, max_depth, min_child_weight
 
After tuning the models with different parameters the final models from each type were selected with most optimal paramters. These were as follows:-
    * Final Decision Tree Model - max_depth = 5; min_samples_leaf = 10; max_features = 8
    * Final Random Forest Model - max_depth = 5; min_samples_leaf = 50; max_features = 10,; n_estimators = 180
    * Final XGBoost Model -(training for 200 iterations) - eta = 0.1; max_depth = 3; min_child_weight = 20


* **Selecting the Best Model** - Once final models were built next step was choosing between final Decision tree, Random forest and XGBoost for Classification models. This was done by evaluating each of the final models on the validation set and comparing the AUC scores. 

We found that, ***XGBoost for Classification model** gave us the best AUC score on validation set hence, it was the **best model for our stroke prediction dataset**.

Thereafter, I used this best model to predict on testing set (unseen data) where also it performed fairly close to validation set scores. Finally, this best model was then saving as a Python script for further deployment as a web service.


* **Balancing Imbalanced Classes** - As the stroke prediction dataset contained imbalanced classes I also tried to balance the dataset using SMOTE technique. This was used to oversample the minority class (stroke=1 i.e., those with a risk of having stroke). The decision tree, random forest and XGBoost models were trained on oversampled dataset and predictions made on validation set. AUC scores were computed on all the three models to understand what effect did balancing of data labels have on the model predictions.

Here too the XGBoost for Classification model outperformed the others.
   




## Instructions on the Running the Project:

 


Data preparation and data cleaning








## Visualizations & Insights Drawn 

