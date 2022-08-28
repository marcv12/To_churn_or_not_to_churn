import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense


dataset = pd.read_csv('churn.csv')


# Transforming Categorical data to numerical data using LabelEncoder and OneHotEncoding

# Encoding categorical data that has 2 unique values
le = LabelEncoder()
le_count = 0
for col in dataset.columns[:]:
    if dataset[col].dtype == 'object' and col != "Attrition_Flag":
        if len(list(dataset[col].unique())) <= 2:
            le.fit(dataset[col])
            dataset[col] = le.transform(dataset[col])
            print('{} column was label encoded.'.format(col))



def to_numeric(s):
  if s == "Attrited Customer":
    return 1
  elif s == "Existing Customer":
    return 0


dataset["Attrition_Flag"] = dataset["Attrition_Flag"].apply(to_numeric)


# Encoding categorical data that has more than 2 unique values
data = dataset.drop(['CID', 'CLIENTNUM', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'],
                    axis=1)

Education_Level = pd.get_dummies(dataset.Education_Level).iloc[:, 1:]
Marital_Status = pd.get_dummies(dataset.Marital_Status).iloc[:, 1:]
Income_Category = pd.get_dummies(dataset.Income_Category).iloc[:, 1:]
Card_Category = pd.get_dummies(dataset.Card_Category).iloc[:, 1:]

data = pd.concat([data, Marital_Status, Income_Category, Card_Category, Education_Level], axis=1)


# Splitting data according to what we want to predict
X = data.drop(['Attrition_Flag'], axis=1)
y = data['Attrition_Flag']

# Splitting data from previous phase into train sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Vanilla application
print("Vanilla models comparison")
models = [('LR', LogisticRegression(solver='liblinear')), ('KSVM', SVC(kernel='rbf')), ('KNN', KNeighborsClassifier()),
          ('CT', DecisionTreeClassifier()), ('RF', RandomForestClassifier())]
# Evaluating Model Results:
results = []
names = []
scoring_list = ['accuracy', 'f1']
for scoring in scoring_list:
    print("For {}".format(scoring))
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


score_array = []
for each in range(1, 25):
    knn_loop = KNeighborsClassifier(n_neighbors=each)
    knn_loop.fit(X_train, y_train)
    result = knn_loop.score(X_test, y_test)
    score_array.append(knn_loop.score(X_test, y_test))

fig = plt.figure(figsize=(15, 7))
plt.plot(range(1, 25), score_array, color='#ec838a')
plt.ylabel('Range\n', horizontalalignment="center",
           fontstyle="normal", fontsize="large",
           fontfamily="sans-serif")
plt.xlabel('Score\n', horizontalalignment="center",
           fontstyle="normal", fontsize="large",
           fontfamily="sans-serif")
plt.title('Optimal Number of K Neighbors \n',
          horizontalalignment="center", fontstyle="normal",
          fontsize="22", fontfamily="sans-serif")
# plt.legend(loc='top right', fontsize = "medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
plt.show()

score_array = []
for each in range(1, 100):
    rf_loop = RandomForestClassifier(
        n_estimators=each, random_state=1)
    rf_loop.fit(X_train, y_train)
    score_array.append(rf_loop.score(X_test, y_test))

fig = plt.figure(figsize=(15, 7))
plt.plot(range(1, 100), score_array, color='#ec838a')
plt.ylabel('Range\n', horizontalalignment="center",
           fontstyle="normal", fontsize="large",
           fontfamily="sans-serif")
plt.xlabel('Score\n', horizontalalignment="center",
           fontstyle="normal", fontsize="large",
           fontfamily="sans-serif")
plt.title('Optimal Number of Trees for Random Forest Model \n', horizontalalignment="center", fontstyle="normal",
          fontsize="22", fontfamily="sans-serif")
# plt.legend(loc='top right', fontsize = "medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
plt.show()

print("Model comparison with optimized versions")
# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Evaluate results
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results = pd.DataFrame([['Logistic Regression',
                         acc, prec, rec, f1]], columns=['Model',
                                                        'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)

# Fitting SVM (SVC class) to the Training set
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results y_pred = classifier.predict(X_test)
# Evaluate results
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame(
    [['SVM (kernel)', acc, prec, rec, f1]],
    columns=['Model', 'Accuracy', 'Precision',
             'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index=True)
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)

# Fitting KNN to the Training set: we found that 9 is optimal number of neighbors:
classifier = KNeighborsClassifier(
    n_neighbors=7,
    metric='minkowski', p=2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Evaluate results
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['K-Nearest Neighbours',
                               acc, prec, rec, f1]], columns=['Model',
                                                              'Accuracy', 'Precision', 'Recall',
                                                              'F1 Score'])
results = results.append(model_results, ignore_index=True)
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)

# Fitting Decision Tree to the Training set:
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Evaluate results
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([[
    'Decision Tree', acc, prec, rec, f1]],
    columns=['Model', 'Accuracy', 'Precision',
             'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index=True)
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)

# Fitting Random Forest to the Training set: we found that optimal number of trees is 35:


rf_classifier = RandomForestClassifier(n_estimators=27,
                                       criterion='entropy', random_state=0)

rf_classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = rf_classifier.predict(X_test)
# Evaluate results
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['Random Forest',
                               acc, prec, rec, f1]],
                             columns=['Model', 'Accuracy', 'Precision',
                                      'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index=True)
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)
print(results)

# improve random forest
print('Parameters currently in use:\n')
pprint(rf_classifier.get_params())


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators' : n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()

# Perform RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 35, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)


better_rf = rf_random.best_estimator_

better_rf.fit(X_train, y_train)

y_pred = better_rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['Better Random Forest',
                               acc, prec, rec, f1]],
                             columns=['Model', 'Accuracy', 'Precision',
                                      'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index=True)
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)
print(results)



# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [False],
    'max_depth': [15, 20, 25],
    'max_features': ['auto','sqrt'],
    'min_samples_leaf': [0.5, 1],
    'min_samples_split':  [1.0, 2],
    'n_estimators': [800, 1150, 1200]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
# n_jobs = -1 uses 100% of the cpu of one of the cores: makes the process faster
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)


grid_search.fit(X_train, y_train)
pprint(grid_search.best_params_)
best_grid = grid_search.best_estimator_
pprint(best_grid)
# Predicting the Test set results
y_pred = best_grid.predict(X_test)
# Evaluate results
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['Best Random Forest',
                               acc, prec, rec, f1]],
                             columns=['Model', 'Accuracy', 'Precision',
                                      'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index=True)
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)
print(results)



# Comparing ANN done in class to our models
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and 4 hidden layers
classifier.add(Dense(units=24, kernel_initializer='uniform', activation='relu', input_dim=28))
classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=20, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
#
# Evaluating results
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['ANN (class)',
                               acc, prec, rec, f1]],
                             columns=['Model', 'Accuracy', 'Precision',
                                      'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index=True)
results = results.sort_values(["Precision",
                               "Recall"], ascending=False)
print(results)
