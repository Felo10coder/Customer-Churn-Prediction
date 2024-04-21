<a name="readme-top"></a>

<div align="center">
  <h1><b>CUSTOMER CHURN PREDICTION</b></h1>
</div>


 TABLE OF CONTENTS -->

# 📗 Table of Contents

- [📗 Table of Contents](#-table-of-contents)
- [Customer Churn Prediction ](#customer-churn-prediction-)
  - [🛠 Built With ](#-built-with-)
    - [Tech Stack ](#tech-stack-)
  - [Key Features ](#key-features-)
  - [💻 Getting Started ](#-getting-started-)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Install](#install)
    - [Usage](#usage)
  - [👥 Authors ](#-authors-)
  - [🔭 Future Features ](#-future-features-)
  - [🤝 Contributing ](#-contributing-)
  - [⭐️ Show your support ](#️-show-your-support-)
  - [🙏 Acknowledgments ](#-acknowledgments-)
  - [📝 License ](#-license-)
## Project Description
<!-- PROJECT DESCRIPTION -->

# Customer Churn Prediction <a name="about-project"></a>

**Customer Churn Prediction** This project aims to develop a machine learning model to predict customer churn in a telecommunications company. By leveraging historical customer data, including usage patterns, demographics, and service subscriptions, the model will identify customers at risk of churning. This predictive capability will enable the company to implement targeted retention strategies and improve customer retention rates.

Features
1. **customerID** :Unique identifier of different customers 
2. **gender**: sex of the customer male or female 
3. **SeniorCitizen**: 'No' to show customer is not a SeniorCitizen and 'Yes' to show a SeniorCitizen  
4. **Partner** :indicates if the customer has a partner 
5. **Dependents**: Shows if the customer has individuals depending on them 
6. **tenure** : The period the customer has been using the company services 
7. **PhoneService** : Indicates if the customer has a phone service 
8. **MultipleLines** : Indicates if the customer has  MultipleLines 
9. **InternetService** : Indicates if the customer has an InternetService 
10. **OnlineSecurity** : Indicates if the customer has OnlineSecurity 
11. **OnlineBackup**: Indicates if the customer has OnlineBackup 
12. **DeviceProtection**: Indicates if the customer has DeviceProtection 
13. **TechSupport**: Indicates if the customer has TechSupport 
14. **StreamingTV**: Shows if the customer streams his or her tv 
15. **StreamingMovies**: Shows if the customer streams his or her movies
16. **Contract**: Shows the type of contract the customer has 
17. **PaperlessBilling**: Indicates if the customers billing was done on paper
18. **PaymentMethod**: Shows the method used in buying services 
19. **MonthlyCharges**:Amount paid by the customer on a monthly basis 
20. **TotalCharges**: Amount the customer has paid throughout his or her tenure using the company services 
21. **Churn** :Shows if the customer churned i.e stopped using the company's services 



## 🛠 Built With <a name="built-with"></a>

### Tech Stack <a name="tech-stack"></a>

<details>
  <summary>GUI</summary>
  <ul>
    <li><a href="">Streamlit</a></li>
  </ul>
</details>

<details>
<summary>Database</summary>
  <ul>
    <li><a href="">Microsoft SQL Server</a></li>
  </ul>
</details>

<details>
<summary>Language</summary>
  <ul>
    <li><a href="">Python</a></li>
  </ul>
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Features -->

## Key Features <a name="key-features"></a>

- **Business understanding: Understanding the project and formualating the hypotheis to test together with business questions to answer that will help in the analysis**
- **Data collection : Gathered comprehensive data on customers churning behaviour and factors influencing them**
- **Data Integration: Combine data from multiple sources such as SQL databases, csv files and GitHub repositories to create a unified dataset for analysis**
-**Exploratory Data Analysis(EDA): Examining and understanding the structure, patterns, and relationships within the dataset. This involved summary statistic, univariate analysis , bivariate analysis, outlier detection and handling missing values**
 **Data Cleaning: preparing the unified data ready for analysis**
 - **Visualizations: Using matplotlib and seaborn to create visually appealing charts, graphs and heatmaps to help answer the business questions**
 - **Answering Business questions: Using visualizations to answer business questions that help in analysing the dataset**
 - **Recommendations: From the answered questions and analysis done , recommendations are given to will help to reduce rate at which the customers churn**
- **Hypothesis Testing: Testing formulated Hypothesis about the relationship between MonthlyCharges and whether a customer churns. Performing statistical analysis to test this hypothesis using the MannWhitneyU test method**
- **Data preparation : Transforming the dataset into a format suitable for training and evaluating ML models. Encompassing a range of tasks to improve the quality, consistency, and relevance of the data, ultimately leading to better model performance and insights. It involes checking if the data is balanced, feature engineering, data transformation, data splitting , constructing pipelines.**
- **Modelling : Creating mathematical representations (models) that learn patterns and relationships from the dataset in order to make predictions on whether a customer will churn or not.Data is trained both when balanced and unbalanced. The models trained include;DecisionTreeClassifier ,LogisticRegression , KNeighbor,RandomForestClassifier, XGBClassifier**

- **Visualizing confusion matrix : Visualizing a confusion matrix can be done using various techniques, including heatmaps.Create the Confusion Matrix : Obtain the confusion matrix using the actual and predicted labels.Plot the ConfusionMatrix:Visualize the confusion matrix using a heatmap or color-coded matrix Interpretation: Analyze the distribution of true positives, true negatives, false positives, and false negatives to evaluate the performance of the classifier**
- **Visualizing Roc_auc: Obtain predicted probabilities from the classifier.Calculate TPR, FPR, and AUC.Plot the ROC curve.Interpret the curve and AUC value to assess the model's performance**
- **Hyperparameter Tuning : Hyperparameters are parameters whose values are set before the learning process begins. They control aspects such as the complexity of the model and its capacity to learn.Hyperparameter tuning helps improve model performance by finding the optimal settings.It prevents overfitting or underfitting of the model by adjusting hyperparameters.Proper tuning can lead to better generalization on unseen data.The steps include Use cross-validation to evaluate the performance of different hyperparameter settings.Consider the computational cost when choosing a tuning technique.Start with a coarse search space and refine it based on initial results.Monitor the performance metrics closely to avoid overfitting to the validation set.**

- **Feature Importance: Feature importance refers to a technique used to determine which features have the most significant impact on the target variable in a machine learning model.gher feature importance values indicate stronger influence on the model's predictions.Negative importance values suggest that the feature has a negative impact on the target variable.**
- **Model Evaluation : Model evaluation is the process of assessing the performance of a machine learning model on unseen data. It helps determine how well the model generalizes to new data and whether it meets the desired objectives. Classification Models:Accuracy: Measures the proportion of correctly classified instances.Precision: Measures the proportion of true positive predictions among all positive predictions.Recall (Sensitivity): Measures the proportion of true positive predictions among all actual positive instances.F1-Score: Harmonic mean of precision and recall, providing a balance between the two.ROC-AUC: Area under the Receiver Operating Characteristic curve, quantifying the model's ability to discriminate between classes.**
- **saving the models: After training a machine learning model, it's important to save it for future use or deployment. Saving the model allows you to reuse it without the need to retrain from scratch.Joblib: Use libraries like Joblib or Pickle to serialize Python objects, including trained models, to disk.**


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->

## 💻 Getting Started <a name="getting-started"></a>


To get a local copy up and running, follow these steps.

### Prerequisites

In order to run this project you need:

- Python
- Jupyter lab
-  Data on Customer churning from Telcomunication companies 


### Setup

Create a new repository on github and copy the URL
Open Visual Studio Code and clone the repository by pasting the URL and selecting the repository destinatination

Create a virtual environment

```sh

python -m venv myvenvv

```

Activate the virtual environment

```sh
    myvenv/Scripts/activate
```


### Install

Here, you need to recursively install the packages in the `requirements.txt` file using the command below 

```sh
   pip install -r requirements.txt
```
## Usage
To run the project, execute the following command:
```sh
    jupyter notebook customer_churn_prediction.ipynb

```
- jupyter notebook will open the specified notebook
- users can explore the code, execute cells, and interact with the project's analysis and visualization

<!-- AUTHORS -->

## 👥 Authors <a name="authors"></a>

🕵🏽‍♀️ **Felix Kwemoi Motonyi**

- GitHub: [GitHub Profile](https://github.com/Felo10coder/git-and-github)
- Twitter: [Twitter Handle](https://x.com/Felo109?t=QQ7Gv-Lj-t6EyLIxYaJFGg&s=09)
- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/felo10)
- Medium: [Medium Profile]()
- Email: [email](felixkwemoi7@gmail.com)
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->

## 📝 License <a name="license"></a>

This project is [MIT](./LICENSE) licensed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## 🤝 Contributing <a name="contributing"></a>

Contributions, issues, and feature requests are welcomed!

Feel free to check the [issues page](../../issues/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SUPPORT -->

## ⭐️ Show your support <a name="support"></a>

If you like this project kindly show some love, give it a 🌟 **STAR** 🌟

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGEMENTS -->

## 🙏 Acknowledgments <a name="acknowledgements"></a>
I would like to thank my team members for their efforts and support in this project. Starting with my team leader Dennis Gitobu, Joy Koech , Loyce Zawadi, Davis Kazungu and Evalyne Nyawira.
I would like to also thank all the free available resource made available online

<p align="right">(<a href="#readme-top">back to top</a>)</p>