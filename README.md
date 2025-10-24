[![PyPI Downloads](https://static.pepy.tech/personalized-badge/lime-ndt?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/lime-ndt)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=aymen20002005_lime_ndt&metric=vulnerabilities)
![Bug](https://sonarcloud.io/api/project_badges/measure?project=aymen20002005_lime_ndt&metric=bugs)

# lime_ndt

lime_ndt is a Python library that introduces an enhanced version of the LIME technique for model explainability, leveraging Neural Decision Trees (NDTs) for improved local and global interpretability of machine learning models.

## Features

- Enhanced LIME explanations using Neural Decision Trees.
- Support for tabular data.
- Tools for extracting, analyzing, and comparing decision trees and forests.
- Utilities for submodular pick and discretization.
- Integration with Keras and scikit-learn.

## Example Usage

Below is a minimal example showing how to use lime_ndt to explain a regression model on the diabetes dataset:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lime_ndt.lime_tabular import LimeTabularExplainer
from lime_ndt.utils.ndt_sklearn_wrapper import NDTRegressorWrapper
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a Random Forest regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Prediction function
def predict_fn(X):
    return rf.predict(X)

# Create the LIME explainer
explainer = LimeTabularExplainer(
    X_train,
    feature_names=diabetes.feature_names,
    class_names=None,
    discretize_continuous=True,
    mode='regression',
)

# Create the local NDT model
model_regressor = NDTRegressorWrapper(D=X_train.shape[1])  # gamma=[1,100] , max_depth = 5

# Explain a test instance
exp = explainer.explain_instance(
    X_test[8],
    predict_fn,
    num_features=10,
    model_regressor=model_regressor
)

# Visualize the explanation
exp.as_pyplot_figure()
plt.show()
```
<img width="718" height="557" alt="image" src="https://github.com/user-attachments/assets/b59c98ec-2463-4a65-a465-5153d5137151" />


## Requirements

- Python 3.7+
- numpy, pandas, scikit-learn, keras, tensorflow, matplotlib

See [requirements.txt](requirements.txt) for the full list.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
