import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data.preprocessing import load_and_prepare_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# randomforest

img_yes = "dataset/yes/*.*"

img_no = "dataset/no/*.*"

X_train, X_val, X_test, y_train, y_val, y_test, class_weights_dict = load_and_prepare_data(img_dir_yes=img_yes, img_dir_no=img_no)

 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid=param_grid,
    cv=3,  
    scoring='accuracy',
    n_jobs=-1,  
    verbose=1
)

grid_search.fit(X_train, y_train)

rfc = grid_search.best_estimator_

y_pred = rfc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_reports = classification_report(y_test, y_pred)

print("Best parameters:", grid_search.best_params_)
print("Accuracy: ", accuracy)
print("Classification", classification_reports)



# # KNN
# img_yes = "dataset/yes/*.*"
# img_no = "dataset/no/*.*"

# X_train, X_val, X_test, y_train, y_val, y_test, class_weights_dict = load_and_prepare_data(img_dir_yes=img_yes, img_dir_no=img_no)

# knn = KNeighborsClassifier(n_neighbors=21)


# knn.fit(X_train.reshape(len(X_train), -1), y_train)


# y_pred = knn.predict(X_test.reshape(len(X_test), -1))


# accuracy = accuracy_score(y_test, y_pred)
# classification_reports = classification_report(y_test, y_pred)

# print("Accuracy: ", accuracy)
# print("Classification report:\n", classification_reports)