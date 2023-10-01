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
4) For our non-treebased models, using the most important features determined by the RF and XGB algorithm (more on this later)

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

### Using All Features 

<p align="center">
  <img src="Images/RMSE_all_features.png" alt="Image Alt Text" width="600px" height="auto">
</p>

<p align="center">
  <img src="Images/R^2_all_features.png" alt="Image Alt Text" width="600px" height="auto">
</p>

Results for non-NN models using all features: 
- As we can see, the RF algorithm performs best (RMSE= 9.41 and R^2=0.92) 
- The XGB algorithm is a close second (RMSE = 10.74 and R^2= 0.90). 
- Using all features, RF and XGB perform best: tree-based models are well equipped to handle multicollinearity, nonlinearity, and high dimensionality
- Our linear models performed the worst using all 81 variables. This was largely due to the high multicollinearity between the variables.

### Using RF and XGB Feature Selection 
- After running the RF and XGB models using all our data, we used the feature importance scores built into both algorithms to determine which features were most important in making predicting the critical temperature. Instead of inputting all 81 variables, we used only the top 10 most important features from RF and XGB. After accounting for overlap, we found 13 RFXGB features to plug into our linear models.

<p align="center">
  <img src="Images/RFXGB_FI_graphs.png" alt="Image Alt Text" width="1000px" height="auto">
</p>

- The thirteen features:
'wtd_std_Valence',
 'wtd_mean_ThermalConductivity',
 'std_atomic_mass',
 'wtd_std_ElectronAffinity',
 'std_Density',
 'wtd_entropy_ThermalConductivity',
 'range_atomic_radius',
 'gmean_ElectronAffinity',
 'range_ThermalConductivity',
 'wtd_gmean_Valence',
 'mean_Density',
 'wtd_mean_Valence',
 'wtd_gmean_ThermalConductivity'

<p align="center">
  <img src="Images/RF_XGB_Dif.png" alt="Image Alt Text" width="1000px" height="auto">
</p>



- Using only a reduced set of 13 features, the RMSE scores went up a few points in the non-tree-based models, likely due to a loss of information. However, R^2 is the more interesting metric here.
- Surprisingly, the variations in these 13 features accounted for a minimum of 65% of the variation in the critical temperature for our non-tree-based models. In comparison, using all 81 features accounted for only 70% variation in the linear models. 
- This means that variation in 60 features resulted in only 5% of the variation in our target variable!
- Using RF and XGB, we found a subset of 13 features that are most important in predicting the critical temperature.


### Using Correlation Coefficient Feature selection 
- We next tried to plug in the 25 features that had a correlation coefficient magnitude of greater than 0.5 with the target variable. 
- Using these selected features, the RMSE scores decreased only slightly in comparison to using the 13 selected features. 
- Moreover, the R^2 values are almost identical when using the 13 RFXGB features and the 25 selected correlation features. 
- This means that the RF and XGB algorithms found the least amount of features that explained the most amount of variance. Using these 13 features helps us create  a less complex, more interpretable model.


<p align="center">
  <img src="Images/RMSE_corr_values.png" alt="Image Alt Text" width="1000px" height="auto">
</p>

<p align="center">
  <img src="Images/R^2_corr_values.png" alt="Image Alt Text" width="1000px" height="auto">
</p>


### Using Principal Component Analysis 
- Principal component analysis (PCA) is a dimensionality technique used to transform a large number of correlated features into a lower-dimensional set of uncorrelated features. The goal is to reduce the number of features while retaining the most information. Click here for a more detailed explanation. 
- Note that PCA is used primarily in linear regression models to deal with multicollinearity. We do not expect this technique to perform well on models that are equipped to handle highly correlated data in a large feature space. 
- For our model, our inputs are the first 16 PCs. These 16 PCs explain 95% of the variation in the target variable.

Insert image 
- Of all our input features, PCA performed the worst. 
- Unsurprisingly, PCA increased the RMSE and decreased the R^2 value for the non-linear models. Since SVR and tree-based models are known for their capabilities to handle large feature numbers and multicollinearity relatively well, reducing the number of features could have led to information loss. 
PCA often improves scores in linear models. Why did our linear model scores become less optimal? 
- Given how well our non-linear models fit the data, we have a strong assumption that the features are not linearly related to the target variable. PCA does not change the linear assumption between features and target variables. It only transforms the original data set to a new data set of linearly uncorrelated features. However, if the relationship between the features and target variables is nonlinear,  most models that have a linear assumption will not perform well.









