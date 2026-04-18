# HK 2026 Challenge by UPJŠ: ML Model Training Information

## Project Overview
* **Challenge Name**: HK 2026 Challenge by UPJŠ.
* **Domain Topic**: Analysis of human tear desiccate based on data analysis and machine learning.
* **Primary Goal**: Develop a data classifier of human tear microscopy images running under the Windows operating system using data analysis and machine learning methods and algorithms.
* **Target Output**: The developed classifier should estimate the probability that a patient has a specific chronic disease.

## Scientific Background
* A human tear contains a cocktail of chemicals consisting of sodium and potassium chlorides, various proteins, glucose, enzymes, and other biomolecules.
* The core hypothesis is that the overall composition of this cocktail directly influences the crystallization patterns of salt crystals, which can act as a fingerprint for a certain type of disease.

## Data
* **Training Set**: Consists of real microscopy images of human tear desiccates. The data includes samples from both healthy individuals and people with chronic diseases, such as diabetes, sclerosis multiplex, and glaucoma.
* **Data Access Links**: The dataset can be accessed at https://temp.kotol.cloud/?c=7IDU  or http://10.0.1.8.

## Evaluation & Testing
* **Hidden Dataset**: The model will be verified using a hidden dataset. This dataset contains real microscopy images from healthy individuals and people with chronic diseases that are not supplied to the participants.
* **Winning Criteria**: The model will be evaluated on the hidden dataset, and the team achieving the highest weighted F1-score will be declared the winner of the challenge.

## Required Solution Components
Your final solution must consist of the following components:
* **Data Description and Preprocessing**: You must provide a clear description of the dataset, including its properties, structure, and any extracted or engineered features. This section should also explain preprocessing steps such as data cleaning, normalization, handling missing values, and feature selection.
* **Algorithms and Methods**: You must include an explanation of the selected algorithms, including their underlying principles and justification for their use.
* **Analytical Objectives**: The applied methods should focus on tasks most relevant to the problem, such as data comparison, correlation analysis, similarity detection, pattern recognition, or the identification of recurring structures.

## Recommended Tools
* **Gwyddion**: This is a multiplatform, modular free software that can be used for the visualization and analysis of data from scanning probe microscopy (SPM) techniques.