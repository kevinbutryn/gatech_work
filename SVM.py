import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit

import pandas as pd
import numpy as np

algo = "SVM"
##########################################################################################################
algoName = "./census/census_"+ algo +"_"
data = './modified_data_orig'
datatype = "census"

bank_df  = pd.read_csv(data, delimiter=',')
bank_df = bank_df.drop(['native-country'], axis=1)
cat_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
bank_df_dummies = pd.get_dummies(bank_df, columns=cat_vars)
bank_df_dummies['sallary'] = bank_df_dummies['sallary'].map({'>50K':0, '<=50K': 1})

labels = bank_df_dummies[['sallary']]
features = bank_df_dummies.drop(['sallary'], axis=1)

##########################################################################################################
# algoName = "creditcard/CC_"+ algo +"_"
# data = './creditCard.csv'
# datatype = "CC"

# bank_df  = pd.read_csv(data, delimiter=',')
# print (bank_df.columns.tolist())
# labels = bank_df[['default']]
# features = bank_df.drop(['default'], axis=1)

##########################################################################################################
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, stratify=labels)
##########################################################################################################
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12



def plot_learning_curve(clf, X, y, title="Insert Title"):
    
    n = len(y)
    train_mean = []; train_std = [] #model performance score (f1)
    cv_mean = []; cv_std = [] #model performance score (f1)
    fit_mean = []; fit_std = [] #model fit/training time
    pred_mean = []; pred_std = [] #model test/prediction times
    # train_sizes=(np.linspace(.05, 1.0, 20)*n).astype('int')  
    train_sizes= [1000,2000,4000]
    print(train_sizes)
    for i in train_sizes:
        print(i)
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X.iloc[idx,:]
        y_subset = y.iloc[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)
        
        train_mean.append(np.mean(scores['train_score'])); train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score'])); cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time'])); fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time'])); pred_std.append(np.std(scores['score_time']))
    
    train_mean = np.array(train_mean); train_std = np.array(train_std)
    cv_mean = np.array(cv_mean); cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean); fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean); pred_std = np.array(pred_std)
    
    plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)
    
    return train_sizes, train_mean, fit_mean, pred_mean
    

def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    
    title = "Learning Curve: "+  algo + '-' + datatype
    plt.figure()
    plt.title( title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(algoName +"LC"+".png")
    
    
def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    title = "Modeling Time: "+  algo + '-' + datatype
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(algoName + "Modeling"+".png")
    
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    title = 'Confusion Matrix - ' + algo + '-' + datatype 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Real Label')
    plt.xlabel('Predicted Label')
    # plt.show()
    plt.savefig(algoName + "confusion"+".png")
    
    
def final_classifier_evaluation(clf,X_train, X_test, y_train, y_test):
    
    print("1")
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    print("2")
    start_time = timeit.default_timer()    
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time
    print("3")
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')


############################################################################

from sklearn.svm import SVC

def hyperSVM(X_train, y_train, X_test, y_test, title):

    title = "Model Complexity Curve: "+  algo + '-' + datatype
    f1_test = []
    f1_train = []
    # kernel_func = ['linear','poly','rbf','sigmoid']
    kernel_func = ['linear','poly',]
    for i in kernel_func:         
            print(i)
            if i == 'poly':
                for j in [2,3]:
                    print(j)
                    clf = SVC(kernel=i, degree=j,random_state=100)
                    clf.fit(X_train, y_train)
                    y_pred_test = clf.predict(X_test)
                    y_pred_train = clf.predict(X_train)
                    f1_test.append(f1_score(y_test, y_pred_test))
                    f1_train.append(f1_score(y_train, y_pred_train))
            else:    
                clf = SVC(kernel=i, random_state=100)
                clf.fit(X_train, y_train)
                y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
                f1_test.append(f1_score(y_test, y_pred_test))
                f1_train.append(f1_score(y_train, y_pred_train))
     
    xvals = ['linear','poly2','poly3']          
    # xvals = ['linear','poly2','poly3','poly4','poly5','rbf','sigmoid']
    plt.plot(xvals, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Kernel Function')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.show()
    plt.savefig(algoName+ "hyper"+".png")
    
def SVMGridSearchCV(X_train, y_train):
    #parameters to search:
    #penalty parameter, C
    #
    Cs = [1e-4, 1e-3, 1e-2, 1e01]
    gammas = [1,10,100]
    param_grid = {'C': Cs, 'gamma': gammas}

    # clf = GridSearchCV(estimator = SVC(kernel='linear',random_state=100), param_grid=param_grid, cv=10)
    clf = GridSearchCV(estimator = SVC(kernel='poly', degree=2,random_state=100), param_grid=param_grid, cv=10)
    clf.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(clf.best_params_)
    print("*********************************************************************************************************************")
    return clf.best_params_['C'], clf.best_params_['gamma']


print("stage 1")
# hyperSVM(X_train, y_train, X_test, y_test,title="Model Complexity Curve for SVM (Phishing Data)\nHyperparameter : Kernel Function")
print("stage 2")
# C_val, gamma_val = SVMGridSearchCV(X_train, y_train)
print("stage 3")
# estimator_phish = SVC(C=C_val, gamma=gamma_val, kernel='linear', random_state=100)
# estimator_phish = SVC(C=C_val, gamma=gamma_val, kernel='poly', degree=2, random_state=100)
estimator_phish = SVC(C=1e-3, gamma=1, kernel='poly', degree=2, random_state=100)
print("stage 4")
# train_samp_phish, SVM_train_score_phish, SVM_fit_time_phish, SVM_pred_time_phish = plot_learning_curve(estimator_phish, X_train, y_train,title="SVM Phishing Data")
print("stage 5")
final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)




