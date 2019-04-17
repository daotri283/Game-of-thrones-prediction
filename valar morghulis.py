# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:10:36 2019

@author: daotr
"""

# Loading Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file = 'GOT_character_predictions.xlsx'

got = pd.read_excel(file)


# Column names
got.columns


# Displaying the first rows of the DataFrame
print(got.head())


# Dimensions of the DataFrame
got.shape


# Information about each variable
got.info()


# Descriptive statistics
got.describe().round(2)


###############################################################################
# Part 1: Imputing Missing Values
###############################################################################

print(
      got
      .isnull()
      .sum()
      )

for col in got:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)
        
        
###Imputing mising Values 
###Title
fill = 'No title'

got['title'] = got['title'].fillna(fill)   


###Culture
fill='No Culture'
got['culture']=got['culture'].fillna(fill)

###Mother
fill='Unknown Mother'
got['mother']=got['mother'].fillna(fill)

###Father
fill='Unknown Father'
got['father']=got['father'].fillna(fill)

###Heir
fill= 'Unknown Heir'
got['heir']=got['heir'].fillna(fill)

##House 
fill='No house'
got['house']=got['house'].fillna(fill)

##Spouse 
fill='No Spouse'
got['spouse']=got['spouse'].fillna(fill)

###Age 
got['age'].mean()
got['name'][got['age']<0] 

#There are 2 cases where the age is negative. Further research is required 
# Rhaego was son of Khal Drogo and Daenerys. He was never born, which mean his age is 0
# Doreah age in the book is 25 
# fixing the data in age 

got.loc[110,'age']=0
got.loc[1350,'age']=25
#checking the mean again
got['age'].mean() 
#Filling the age with 
fill=got['age'].mean()
got['age']=got['age'].fillna(fill)


##isAliveMother, Father , Heir and Spouse NA will be filled with -1 
fill= -1 
got['isAliveMother']=got['isAliveMother'].fillna(fill)
got['isAliveFather']=got['isAliveFather'].fillna(fill)
got['isAliveHeir']=got['isAliveHeir'].fillna(fill)
got['isAliveSpouse']=got['isAliveSpouse'].fillna(fill)


#Checking one more time for NA
print(
      got
      .isnull()
      .sum()
      )
#Birthdate is measured with unknown method so I am not going to use it in my model 

##Creating one more variable 
for val in enumerate(got.loc[ : , 'popularity']):
    
    if val[1] <= 0.3:
        got.loc[val[0], 'popular'] = 0



for val in enumerate(got.loc[ : , 'popularity']):
    
    if val[1] > 0.3:
        got.loc[val[0], 'popular'] = 1

###############################################################################
# Part 2: EDA
###############################################################################

###Culture: There are many different values for the same culture, so I will group them up
##Grouping culture 
cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavos': ['braavosi', 'braavos'],
    'Dorne': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westeros': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvos': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Wildling': ['wildling', 'first men', 'free folk'],
    'Qarth': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
    'Ironborn': ['ironborn', 'ironmen'],
    'Mereen': ['meereen', 'meereenese'],
    'RiverLands': ['riverlands', 'rivermen'],
    'Vale': ['vale', 'valemen', 'vale mountain clans']
}

def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()
got.loc[:, "culture"] = [get_cult(x) for x in got["culture"]]


##Title and Alive or dead 
got.boxplot(column = ['isAlive'],
                by = ['m_title'],
                vert = False,
                patch_artist = False,
                meanline = False,
                showmeans = True)

sns.countplot(x="isAlive", hue="m_title", data=got)
sns.countplot(x="isAlive", hue="male", data=got)
sns.countplot(x="isAlive", hue="m_house", data=got)
sns.countplot(x="isAlive", hue="m_culture", data=got)

########################
# Visual EDA (Histograms)
########################


plt.subplot(1, 1, 1)
sns.distplot(got['age'],
             bins = 35,
             color = 'g')

plt.xlabel('age')


###############################################################################
# Qualitative Variable Analysis (Box Plots + Violin Plots)
###############################################################################
        
# Violin Plots

sns.violinplot(x = 'popular',
               y = 'isNoble',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()

sns.violinplot(x = 'popular',
               y = 'male',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()

sns.violinplot(x = 'popular',
               y = 'isMarried',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()

sns.violinplot(x = 'popular',
               y = 'book1_A_Game_Of_Thrones',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()


sns.violinplot(x = 'popular',
               y = 'book1_A_Game_Of_Thrones',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()


sns.violinplot(x = 'popular',
               y = 'book2_A_Clash_Of_Kings',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()

sns.violinplot(x = 'popular',
               y = 'book3_A_Storm_Of_Swords',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()

sns.violinplot(x = 'popular',
               y = 'book4_A_Feast_For_Crows',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()

sns.violinplot(x = 'popular',
               y = 'book5_A_Dance_with_Dragons',
               hue = 'isAlive',
               data = got,
               orient = 'v',
               inner = None,
               color = 'green')
plt.show()
######################################################################
# Box Plots
########################
# age by male


got.boxplot(column = ['age'],
                by = ['male'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("age by male")
plt.suptitle("")


plt.show()


########################
# age v/s alive


got.boxplot(column = ['age'],
                by = ['isAlive'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("age by alive")
plt.suptitle("")


plt.show()



########################
# popularity v/s alive

got.boxplot(column = ['popularity'],
                by = ['isAlive'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("popularity v/s alive")
plt.suptitle("")


plt.show()

###############################################################################
# Part 3: Machine Learning Model 
###############################################################################


# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.metrics import roc_auc_score
###############################################################################
# Decision Tree
###############################################################################

got_data   = got.drop(['isAlive','name','title','culture','dateOfBirth',
                           'mother','father','heir','house','spouse'],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


# Full tree.
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))


########################
# Making Adjustment to the model
########################

tree_leaf_50 = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 50,
                                     random_state = 508)

tree_leaf_50.fit(X_train, y_train)

print('Training Score', tree_leaf_50.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_50.score(X_test, y_test).round(4))



# Defining a function to visualize feature importance

########################
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
########################

########################
# Tree with the important features
########################
plot_feature_importances(tree_leaf_50,
                         train = X_train,
                         export = True)

got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother'
                        ],axis=1)

got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)

######################

tree_leaf_50 = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 50,
                                     random_state = 508)

tree_leaf_50.fit(X_train, y_train)

print('Training Score', tree_leaf_50.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_50.score(X_test, y_test).round(4))


#AUC Score for Decision Tree
# Generating Predictions based on the optimal c_tree model
d_tree_optimal_fit_train = tree_leaf_50.predict(X_train)

d_tree_optimal_fit_test = tree_leaf_50.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,d_tree_optimal_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,d_tree_optimal_fit_test).round(4))

# Cross-Validation on tree_leaf_50 (cv = 3)

cv_tree_3 = cross_val_score(tree_leaf_50,
                             got_data,
                             got_target,
                             cv = 3,
                             scoring='roc_auc')


print(cv_tree_3)


print(pd.np.mean(cv_tree_3).round(3))


###############################################################################
# Classification Tree
###############################################################################
got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother'
                        ],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


#########################
# Building Classification Trees
#########################

from sklearn.tree import DecisionTreeClassifier # Classification trees

c_tree = DecisionTreeClassifier(random_state = 508)

c_tree_fit = c_tree.fit(X_train, y_train)

###########################
# Hyperparameter Tuning with GridSearchCV
############################

from sklearn.model_selection import GridSearchCV

########################
# Optimizing for two hyperparameters
########################


# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 500)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space}

# Building the model object one more time
c_tree_2_hp = DecisionTreeClassifier(random_state = 508)

# Creating a GridSearchCV object
c_tree_2_hp_cv = GridSearchCV(c_tree_2_hp, param_grid, cv = 3)


# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", c_tree_2_hp_cv.best_score_.round(4))

# Building a tree model object with optimal hyperparameters
c_tree_optimal = DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 4,
                                        min_samples_leaf = 31)

c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)

#AUC Score 
# Generating Predictions based on the optimal c_tree model
c_tree_optimal_fit_train = c_tree_optimal_fit.predict_proba(X_train)[:,1]

c_tree_optimal_fit_test = c_tree_optimal_fit.predict_proba(X_test)[:,1]

print('Training AUC Score:',roc_auc_score(
        y_train,c_tree_optimal_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,c_tree_optimal_fit_test).round(4))

# Cross-Validation on c_tree_optimal (cv = 3)

cv_tree_3 = cross_val_score(c_tree_optimal_fit,
                             got_data,
                             got_target,
                             cv = 5,
                             scoring='roc_auc')

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))

###############################################################################
# Logistic Regression 
###############################################################################from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'lbfgs',
                            C = 1)
logreg_fit = logreg.fit(X_train, y_train)

logreg_pred = logreg_fit.predict(X_test)

print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))


# Visualizing a confusion matrix

print(confusion_matrix(y_true = y_test,
                       y_pred = logreg_pred))

import seaborn as sns

labels = ['Alive', 'Dead']

cm = confusion_matrix(y_true = y_test,
                      y_pred = logreg_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Greys')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()

#AUC Score 
# Generating Predictions based on the optimal model
logreg_fit_train = logreg_fit.predict_proba(X_train)[:,1]

logreg_fit_train_test = logreg_fit.predict_proba(X_test)[:,1]

print('Training AUC Score:',roc_auc_score(
        y_train,logreg_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,logreg_fit_train_test).round(4))

# Cross-Validation on c_tree_optimal (cv = 3)

cv_tree_3 = cross_val_score(logreg_fit,
                             got_data,
                             got_target,
                             cv = 3,
                             scoring='roc_auc')

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))

###############################################################################
# Random Forest
###############################################################################
# Loading new libraries
from sklearn.ensemble import RandomForestClassifier

got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother'
                        ],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)


# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)

########################
# Parameter tuning with GridSearchCV
########################

from sklearn.model_selection import GridSearchCV

# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]

param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}

# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)



# Fit it to the training data
full_forest_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))


########################
# Building Random Forest Model Based on Best Parameters
########################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 16,
                                    n_estimators = 1100,
                                    warm_start = True)



rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)


#AUC Score 
# Generating Predictions based on the optimal Random Forest model
rf_optimal_pred_train = rf_optimal.predict_proba(X_train)[:,1]

rf_optimal_pred_test = rf_optimal.predict_proba(X_test)[:,1]

print('Training AUC Score:',roc_auc_score(
        y_train,rf_optimal_pred_train).round(3))
print('Testing AUC Score:',roc_auc_score(
        y_test,rf_optimal_pred_test).round(3))

# Cross-Validation on rf_optimal (cv = 3)

cv_tree_3 = cross_val_score(rf_optimal,
                             got_data,
                             got_target,
                             cv = 3,
                             scoring='roc_auc')

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))



###############################################################################
# Gradient Boosted Machines
###############################################################################

got_data   = got.drop(['isAlive',
                       'name',
                       'popular',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother'
                        ],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


from sklearn.ensemble import GradientBoostingClassifier

# Building a weak learner gbm
gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
                                  max_depth = 3,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )

gbm_basic_fit = gbm_3.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_basic_fit.score(X_train, y_train)
gmb_basic_test  = gbm_basic_fit.score(X_test, y_test)


########################
# Applying GridSearchCV
########################

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}



# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)



# Creating a GridSearchCV object
gbm_grid_cv = GridSearchCV(gbm_grid, param_grid, cv = 3)



# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))


########################
# Building GBM Model Based on Best Parameters
########################

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.1,
                                      max_depth = 1,
                                      n_estimators = 100,
                                      random_state = 508)



gbm_optimal.fit(X_train, y_train)


gbm_optimal_score = gbm_optimal.score(X_test, y_test)


gbm_optimal_pred = gbm_optimal.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))


#AUC Score 
# Generating Predictions based on the optimal GBM model
gbm_optimal_pred_train = gbm_optimal.predict_proba(X_train)[:,1]

gbm_optimal_pred_test = gbm_optimal.predict_proba(X_test)[:,1]

print('Training AUC Score:',roc_auc_score(
        y_train,gbm_optimal_pred_train).round(3))
print('Testing AUC Score:',roc_auc_score(
        y_test,gbm_optimal_pred_test).round(3))


# Cross-Validation on fbm_optimal (cv = 3)

cv_tree_3 = cross_val_score(gbm_optimal,
                             got_data,
                             got_target,
                             cv = 3,
                             scoring='roc_auc')

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))



###############################################################################
# Part 4: Model Results 
###############################################################################

#AUC Score for Decision Tree
print('Decision Tree Training AUC Score:',roc_auc_score(
        y_train,d_tree_optimal_fit_train).round(4))
print('Decision Tree Testing AUC Score:',roc_auc_score(
        y_test,d_tree_optimal_fit_test).round(4))


#AUC Score for Classification Tree
print('Classification Tree Training AUC Score:',roc_auc_score(
        y_train,c_tree_optimal_fit_train).round(4))
print('Classification Tree Testing AUC Score:',roc_auc_score(
        y_test,c_tree_optimal_fit_test).round(4))

#AUC Score for Randon Forest

print('Random Forest Training AUC Score:',roc_auc_score(
        y_train,rf_optimal_pred_train).round(3))
print('Random Forest Testing AUC Score:',roc_auc_score(
        y_test,rf_optimal_pred_test).round(3))

#AUC Score for GBM

print('GBM Training AUC Score:',roc_auc_score(
        y_train,gbm_optimal_pred_train).round(3))
print('GBM Testing AUC Score:',roc_auc_score(
        y_test,gbm_optimal_pred_test).round(3))

##export the Decision tree result to excel, Which is the Random Forest 
Predictions = pd.DataFrame({
        'Actual' : y_test,
        'Random Forest Prediction' : rf_optimal_pred})

Predictions.to_excel('Final Prediction.xlsx')