# NCSU CSC 422 - Machine Learning - Rain Prediction Final Project
## Brief Introduction to Rain Prediciton Project:
In this project, we used the dataset provided on Kaggle fetched from the offical dataset released by the government of Australia in 2020 to make predicitons on if tomorrow will rain.
To have more novelty, we selected three locations in the dataset and compared different attributes that impact their climates.
First city is Uluru, located in the middle of the Australia surrounded by gigantic desert. The second city is a costal city called Brisbane. The third city is in Pacific Ocean named Norfolk Island.
Uluru, Brisbane, and Norfolk Island locates from 24°N to 26°N, having a similar latitudes; but they have quite different longtitudes which leads to various climates.
What are the attributes that impact each city the most on rainfall? How are the most affective features different from other cities? We will learn that after features selection and make a prediciton on rainfall.

## Documents description:
### Rain_Prediction_Review.html
We ran it on JyupterHub and recorded one of the acutal results to a html file. The accuracy may differ each time by 2% because the dataset used to training and testing are random.

### Rain_Prediction_JupyterHub.ipynb
JupyterHub version of our code

### Rain_Predicition_Runnable.py
Raw file in .py extension that could open and view the code on any editor

### Rain_Prediction_LaTex.tex
LaTex version of the rain prediction Project

### weatherAUS.csv
Raw dataset download from https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
Office dataset of rainfall released by the government of Australia

### weatherAUS_modified.csv
Modified version of raw dataset for us to better understand the dataset

### Rain_Prediciton.zip
Compressed file of everything in the directory

### images folder:
  1. Uluru_DT.png: Built decision tree model for predicitons in Uluru
  2. Brisbane_DT.png: Built decision tree model for predicitons in Brisbane
  3. Norfolk_Island_DT.png: Built decision tree model for predicitons in Norfolk Island
  4. PCA.png: Running principle components analysis (PCA) and compared how the eigenvalues changed as more features input to determine how many features we should select
