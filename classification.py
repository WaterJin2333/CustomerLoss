from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve, auc,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_excel('new_new_data.xlsx')

df = df.loc[df['each_cost']>1]
class_pay = {label:idx for idx,label in enumerate(set(df['most_pay']))}
print(class_pay)
df['most_pay'] = df['most_pay'].map(class_pay)
class_type = {label:idx for idx,label in enumerate(set(df['most_type']))}
print(class_type)
df['most_type'] = df['most_type'].map(class_type)

tags=df['tag'].tolist()
df=df.drop(columns=['tag'])
feature_name=df.columns.values.tolist()
df_list=df.values.tolist()
scaler = StandardScaler()
df_list = scaler.fit_transform(df_list)
x_train, x_test, y_train, y_test = train_test_split(df_list, tags, test_size=0.33, random_state=42)

#SVM
# rnd_clf = SVC(kernel='rbf', class_weight='balanced',probability=True)
#随机森林
rnd_clf = RandomForestClassifier(n_estimators=50, max_leaf_nodes=15, n_jobs=2)
#NB
# rnd_clf = GaussianNB()
#LOGISTIC
# rnd_clf = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
#KNN
# rnd_clf = KNeighborsClassifier()

rnd_clf.fit(x_train, y_train)
y_predict_rf = rnd_clf.predict(x_test)
y_predict_prob=rnd_clf.predict_proba(x_test)
print('accuracy_score:%f'%accuracy_score(y_test, y_predict_rf))
print('precision_score:%f'%precision_score(y_test, y_predict_rf))
print('recall_score:%f'%recall_score(y_test, y_predict_rf))
print('f_score:%f'%f1_score(y_test, y_predict_rf))
print(classification_report(y_test,y_predict_rf,target_names=['true','false']))
print(confusion_matrix(y_test,y_predict_rf))
# for name, score in zip(feature_name, rnd_clf.feature_importances_):
#     print(name, score)


#
# y_predict_all=rnd_clf.predict_proba(df_list)
# for i in range(len(df_list)):
#     df_list[i].append(y_predict_all[i][1])
# new_df=pd.DataFrame(df_list)
# new_df.to_excel('cluster_data.xlsx')

y_score=[]
for prob in y_predict_prob:
    y_score.append(prob[1])

fpr, tpr, thresholds  =  roc_curve(y_test, y_score)
roc_auc =auc(fpr, tpr)


plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
