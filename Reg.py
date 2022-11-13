# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, RocCurveDisplay, auc, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from numpy import arange, array
from itertools import cycle
from time import perf_counter, time
#sns.set()

def plot_ROC(x, y, title = ""):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(x[:,i], y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    colors = cycle(['blue', 'red', 'green'])
    
    plt.figure()
    classes = ["Low", "Normal", "High"]
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class "%s" (area = %.2f)'%(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()    

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

data = pd.read_csv("Stress-Lysis.csv", sep = ',')

data["Temperature"] = 5/9*(data["Temperature"] - 32)
data = data.replace({'Stress Level': {0: "Low", 1: "Normal", 2:"High"}})
print(data.head())
print()

print("Количество строк: %d"%(data.shape[0]))
print("Количество колонок: %d"%(data.shape[1]))
print()

print("Типы данных")
print(data.dtypes)
print()

print("Проверка на пропуски")
print(data.isnull().sum())
print()


print("Основные статистические показатели")
print(data.describe())
print()

order = ["Low", "Normal", "High"]

fig1, axes1 = plt.subplots(nrows=1, ncols=3)
sns.barplot(data, x = "Stress Level", y = "Humidity", ax = axes1[0], order = order)
sns.barplot(data, x = "Stress Level", y = "Temperature", ax = axes1[1], order = order)
sns.barplot(data, x = "Stress Level", y = "Step count", ax = axes1[2], order = order)


fig2, axes2 = plt.subplots(nrows=1, ncols=3)
sns.boxplot(data, x = "Humidity", y = "Stress Level", ax = axes2[0], order = order)
sns.boxplot(data, x = "Temperature", y = "Stress Level", ax = axes2[1], order = order)
sns.boxplot(data, x = "Step count", y = "Stress Level", ax = axes2[2], order = order)

fig3, axes3 = plt.subplots(nrows=1, ncols=3)
sns.violinplot(data, x = "Humidity", y = "Stress Level", ax = axes3[0], order = order)
sns.violinplot(data, x = "Temperature", y = "Stress Level", ax = axes3[1], order = order)
sns.violinplot(data, x = "Step count", y = "Stress Level", ax = axes3[2], order = order)

data = data.replace({'Stress Level': {"Low": 0, "Normal": 1, "High":2}})

X = data[["Humidity", "Temperature", "Step count"]]
y = data["Stress Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(time()))

lr = LogisticRegression(multi_class = "multinomial", solver ='newton-cg')
y_score = lr.fit(X_train,y_train).decision_function(X_test)

plot_ROC(label_binarize(y_test,classes=[0,1,2]), y_score, "ROC for MLR")

y_pred = lr.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

class_names = array(["Low", "Normal", "High"])

fig, ax = plt.subplots()
tick_marks = arange(len(class_names)) 
ax.set_yticks([0,1,2])
ax.set_xticks([0,1,2])
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual diagnosis')
plt.xlabel('Predicted diagnosis')
plt.show()

print("Матрица весов MLR")
print(pd.DataFrame(lr.coef_, ['Low', 'Normal', 'High'], columns=["Humidity", "Temperature", "Step count"]))
print()

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

print(pd.DataFrame({"Class": ['Low', 'Normal', 'High'], "Precision": precision, "Recall": recall, "FScore": fscore, "Support": support}))

y = label_binarize(data["Stress Level"], classes=[0, 1, 2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(time()))

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

plot_ROC(y_test, y_score,"ROC for MCC")