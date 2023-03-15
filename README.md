<img align="right" alt="Coding" width="300" src="https://media.giphy.com/media/gutZ5Pm6Xl62eIf5RZ/giphy.gif">
<h1 align="center">Breast-Cancer-Prediction-Using-SVM</h1> 
<h1 align="center">Python | Pandas | Matplotlib | Seaborn | Machine Learning | Support Vector  Machine</h1>
<img align="left" alt="Coding" width="300" height = "200" src="https://media.giphy.com/media/sCqnpiUFN228E/giphy.gif">

  ## Classification of Breast Cancer using ML 
  By using a best fit Machine Learning Algorithmic model (Support Vector Machine) to classify where the patient is classified with one of Cancer or not ,
  - Benign   --> for not a Cancer , denoted by  "1"
  - Malignant --> as a Cancer , denoted by  "0"


## Deployment

Depending Libraries

```bash
  import sklearn
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.model_selection import train_test_split
  from sklearn import datasets
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
  from sklearn.model_selection import GridSearchCV
```

## Loading Data-set

Load the breast cancer Dataset from sklearn!

See `from sklearn import datasets` this has n number of data's.

From there importing data `datasets.load_breast_cancer()`.


## Having look at Data-set

![App Screenshot](images/df_data.png)

