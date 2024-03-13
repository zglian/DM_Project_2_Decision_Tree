# Data Mining Project 2 - Classification Analysis
This is a course project in NCKU DM class.

[project link](https://hackmd.io/jGaZxVjPRIKX1kpfcfUoRQ)

[report link](https://www.notion.so/zhi-gang/Data-Mining-Project2-d290eb6f2d6348bda8e1f314147035d2)

## Overview
This project focuses on classification analysis, aiming to assess the risk of motorcycle accidents based on various rider characteristics. It involves designing a dataset with 10 features, including categorical, Gaussian continuous, discrete numerical, and boolean types. The project includes defining rules for different risk levels, implementing classification models, analyzing their performance, and discussing the results.

## Dataset
The dataset is designed with 10 features, each contributing to the classification of motorcycle accident risk into three categories: low risk, medium risk, and high risk. Features include gender, engine displacement, motorcycle model, pillion rider frequency, helmet type, height, weight, age, purchase price, monthly mileage, traffic ticket count, and average speeding rate.

## Classification Models
- **Decision Tree**: Implemented with an accuracy of 87.70% on the test dataset.
- **Other Models**: Also implemented Naive Bayes, SVM, and KNN, achieving accuracies ranging from 54.07% to 63.53%.

## Analysis
- **Decision Trees**: Analyzed decision tree structure, feature importance, and accuracy.
- **Comparisons**: Compared decision tree results with other classification models, discussing their strengths and weaknesses.
- **Discussion**: Altered the absolutely-right rules to observe the impact on classification accuracy.

## File Structure
- **inputs**: Directory for input files containing the generated dataset.
- **main.py**: Main script for running the classification models and generating figures.
- **report.pdf**: Report file containing detailed analysis, figures, and results.
