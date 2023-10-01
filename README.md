# SuperconductorRegressionAnalysis

Access the full colab notebook [here](https://colab.research.google.com/drive/1rvXt8XBbyUkSVo73d0YCBkaSmE9ebSRx?usp=sharing) for all code and in-depth explanations. 

## Table of Contents:
- [Project Objective](#project-objective)
- [Project Outcomes](#project-outcomes)
- [Background Information](#background-information)
- [Understanding the Data](#understanding-the-data)
- [Feature Selection](#feature-selection)
- [Non-Neural Network Models](#non-neural-network-models)
- Neural Network Models
- Conclusions

## Project Objective
The objectives of this project are to:
- Create a model that best predicts the critical temperature of a superconductor based on its material properties
- Compare model performance based on evaluation metrics
- Determine which features are most important in determining critical temperatures

## Project Outcomes 
- Developed a Random Forest model that predicted the critical temperature of a superconductor with root mean square error of 9.41 and R^2 value of 0.92
- Found a subset of 13 of 81 features that explained a minimum of 65% of the variability in the target variable for all linear models
- Found a subset of 25 of 81 features that explained a minimum of 65% of the variability in the target variable for all non-neural network models


## Background Information

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Superconductors are a unique class of materials that efficiently conduct electricity without electrical resistance or heat loss. When an electrical current is passed through conventional materials, some of the flowing electrons collide with the atoms in the material, creating resistance. This resistance leads to dissipation of energy as heat.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;However, when an electrical current is sent through a superconductor, no collisions or resistance occurs. The electrons flow smoothly through the material, so no energy is lost as heat. This property of zero resistance makes superconductors highly desirable in areas like power transmission and distribution.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Superconductors also display interesting relationships with magnetic fields. Type 1 superconductors repel outside magnetic fields by creating a shield that prevents exterior magnetic lines from entering (Meissner Effect). When a magnet is placed atop a Type 1 superconductor, the magnet will hover in the air because the superconductor’s force field repels the magnet’s magnetic field. This is how hoverboards and levitating (Maglev) trains work!

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Type 2 superconductors allow some magnetic fields into their shields without losing their superconducting properties. MRI machines require strong magnetic fields to create high-quality images. Since Type 2 superconductors allow the controlled entry of magnetic fields, they are ideal for creating the magnetic fields needed for MRIs.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;What’s the catch?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Superconductors require extremely low temperatures to reach their superconducting state. The temperature at which a material enters its superconductive state is the critical temperature. For some reference, the highest temperature superconductor (as of now) is a hydrogen sulfide compound which exhibits superconductivity only at -70 degrees Celsius! Since superconductors require extremely low temperatures, readily applying them in technology is challenging. Much of the current research in the field focuses on discovering superconductors at higher temperatures.

## Understanding the Data
- This data was taken from UC Irvine's machine learning repository.
- The superconductor data set contains 21263 superconducting materials.
- There are 81 features representing the material properties of the superconductors. These 81 features are variations of 9 main features: number of elements, atomic mass,  first ionization energy, atomic radius, density, electron affinity, fusion heat, thermal conductivity, and valence. Click here for a brief introduction to these features.

Here is what the first five rows of the data frame look like before feature selection: 

The target variable is the critical temperature:

<p align="center">
  <img src="Images/crit_temp.png" alt="Image Alt Text" width="500px" height="auto">
</p>

- Many materials have critical temperatures between 0 and 25 Kelvin ( -459.67°F to -414.67°F). Another group of materials have critical temperatures between 65 and 95 Kelvin (-337.67°F to -297.67°F).

## Feature Selection
When feeding data into our models, we must decide which features are relevant. There are several ways to do this. We examine four techniques of feature selection. 
1) Using all 81 features (baseline metric) 
2) Using only the features that have correlation coefficients with the target variable of over 0.5 
3) Using Principal Component Analysis 
4) For our linear models, using the RF and XGB 10 most important features (more on this later)

## Non Neural Network Models

We consider these regression models: 
- Linear Regression 
- Ridge Regression 
- Support Vector Regression 
- Decision Tree Regression 
- Random Forest Regression 
- XGB Regression

We look for a model and feature selection technique that optimizes our evaluation metrics (minimizes RMSE and maximizes R^2). For more information on both the models and the evaluation metrics, click here. 

Before we begin modeling, we create a simple pipeline that scales whatever set of input features we use for every model. Feature scaling ensures that all features have similar magnitudes, which prevents certain features (features with larger magnitudes) from dominating the training process.

```python
scaled_pipelines = {
    'Lin reg scaled': Pipeline([
        ('scaler', StandardScaler()),
        ('Linear Regression', LinearRegression())
    ]),
    'Ridge reg scaled': Pipeline([
        ('scaler', StandardScaler()),
        ('Ridge Regression', Ridge(max_iter=10000))
    ]),
    'Lasso reg scaled': Pipeline([
        ('scaler', StandardScaler()),
        ('Lasso Regression', Lasso(max_iter=10000))
    ]),
    'SVR scaled': Pipeline([
        ('scaler', StandardScaler()),
        ('Support Vector Regressor', SVR())
    ]),
    'DT scaled': Pipeline([
        ('scaler', StandardScaler()),
        ('Decision Tree', DTR())
    ]),
    'RF scaled': Pipeline([
        ('scaler', StandardScaler()),
        ('Random Forest', RFR())
    ]),
    'XGB scaled': Pipeline([
        ('scaler', StandardScaler()),
        ('XGB', XGB.XGBRegressor(random_state=42))
    ]),
}
```
- Since we are trying to optimize the evaluation metrics, we must tune our parameters to see which parameters yield the best results. We then save the best estimator for each model type and select which of the tuned models yields the lowest RMSE and the highest R^2. 
- We use GridSearchCV to exhaustively check all possible combinations of hyperparameters from the parameter grid for a particular model. The hyperparameters we check come from this parameter grid:



```python
param_grid = {
    'Lin reg': {},
    'Ridge reg': {'Ridge Regression__alpha': [0.1, 1.0]},
    'Lasso reg': {'Lasso Regression__alpha': [0.1, 1.0]},
    'SVR': {
        'Support Vector Regressor__C': [0.1, 1.0],
        'Support Vector Regressor__kernel': ['linear', 'rbf']
    },
    'DT': {'Decision Tree__max_depth': [None, 10]},
    'RF': {
        'Random Forest__n_estimators': [100],
        'Random Forest__max_depth': [None, 10]
    },
    'XGB': {
        'XGB__n_estimators': [100, 200],
        'XGB__learning_rate': [0.01, 0.1],
        'XGB__max_depth': [3, 4],
        'XGB__min_child_weight': [1, 3],
        'XGB__subsample': [0.8, 0.9],
        'XGB__colsample_bytree': [0.8, 0.9],
        'XGB__gamma': [0, 0.1]
    }
}
```
Using a for loop with GridSearch, we save the best estimators for each model type in the best estimator list. We then compare the best estimators for each of the models.

These were the results for our models when we used all 81 features: 

<p align="center">
  <img src="Images/RMSE_all_features" alt="Image Alt Text" width="500px" height="auto">
</p>

<p align="center">
  <img src="Images/R^2_all_features" alt="Image Alt Text" width="500px" height="auto">
</p>









