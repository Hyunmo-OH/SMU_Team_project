
# 모듈 Import
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 불러오기
data=pd.read_csv(u"H:/2018.2학기/4_경영 빅데이터 분석/프로젝트 최종/데이터 SNS언급량 포함.csv",
                encoding="cp949")

# 관객수 등급 파생변수 생성
data.loc[data[u"관객수"]>10000,u"관객수 등급"]="K"
data.loc[data[u"관객수"]>1000000,u"관객수 등급"]="J"
data.loc[data[u"관객수"]>2000000,u"관객수 등급"]="I"
data.loc[data[u"관객수"]>3000000,u"관객수 등급"]="H"
data.loc[data[u"관객수"]>4000000,u"관객수 등급"]="G"
data.loc[data[u"관객수"]>5000000,u"관객수 등급"]="F"
data.loc[data[u"관객수"]>6000000,u"관객수 등급"]="E"
data.loc[data[u"관객수"]>7000000,u"관객수 등급"]="D"
data.loc[data[u"관객수"]>8000000,u"관객수 등급"]="C"
data.loc[data[u"관객수"]>9000000,u"관객수 등급"]="B"
data.loc[data[u"관객수"]>10000000,u"관객수 등급"]="A"

# 종속변수, 목표변수 데이터 셋 분리
feature_names=[u"스크린수",u"개봉월",u"요일",u"상영횟수",u"국적",u"배급사등급",u"상영등급",u"트윗언급량"]
dfX=data[feature_names].copy()
dfy=data[u"관객수 등급"].copy() # 분류용 데이터 셋
dfy2=data[u'관객수'].copy()  # 회귀용 데이터 셋

# str 변수 정수형 인코딩
dfX[u"요일"]=LabelEncoder().fit_transform(dfX[u"요일"])
dfX[u"배급사등급"]=LabelEncoder().fit_transform(dfX[u"배급사등급"])

# 훈련 셋, 데이터 셋 분류 (분류)
X_train, X_test, y_train, y_test = train_test_split(
     dfX, dfy, test_size=0.2, random_state=123456)

# 훈련 셋, 데이터 셋 분류 (회귀)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
     dfX, dfy2, test_size=0.2, random_state=123456)

# 변수간 관계 개괄적 확인

%matplotlib inline
sns.pairplot(data, hue =u'관객수 등급',)





# 분류 (의사결정 나무)

model_cl_dt = DecisionTreeClassifier(
    criterion='entropy', max_depth=4)

model_cl_dt.fit(X_train,y_train)

y_pred_cl_dt = model_cl_dt.predict(X_test)

print 'Accuracy: {0:.2f}%'.format(accuracy_score(y_test, y_pred_cl_dt)*100)

print "precision_score : {0:.2f}%".format(metrics.precision_score(y_test, y_pred_cl_dt,average='weighted')*100)

print "recall_score : {0:.2f}%".format(metrics.recall_score(y_test, y_pred_cl_dt,average='weighted')*100)

print "f1_score : {0:.2f}%".format(metrics.f1_score(y_test, y_pred_cl_dt,average='weighted')*100)

dot_data = StringIO()
export_graphviz(model_cl_dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())



# 분류 (랜덤 포레스트)

from sklearn.ensemble import RandomForestClassifier

model_cl_rf= RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=4, random_state=123456)

model_cl_rf.fit(X_train,y_train)

y_pred_cl_rf = model_cl_rf.predict(X_test)

print 'Accuracy: {0:.2f}%'.format(accuracy_score(y_test, y_pred_cl_rf)*100)

print "precision_score : {0:.2f}%".format(metrics.precision_score(y_test, y_pred_cl_rf,average='weighted')*100)

print "recall_score : {0:.2f}%".format(metrics.recall_score(y_test, y_pred_cl_rf,average='weighted')*100)

print "f1_score : {0:.2f}%".format(metrics.f1_score(y_test, y_pred_cl_rf,average='weighted')*100)



# 분류 (KNN)

from sklearn.neighbors import KNeighborsClassifier

model_cl_knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski')

model_cl_knn.fit(X_train,y_train)

y_pred_cl_knn = model_cl_knn.predict(X_test)

print 'Accuracy: {0:.2f}%'.format(accuracy_score(y_test, y_pred_cl_knn)*100)

print "precision_score : {0:.2f}%".format(metrics.precision_score(y_test, y_pred_cl_knn,average='weighted')*100)

print "recall_score : {0:.2f}%".format(metrics.recall_score(y_test, y_pred_cl_knn,average='weighted')*100)

print "f1_score : {0:.2f}%".format(metrics.f1_score(y_test, y_pred_cl_knn,average='weighted')*100)



# 분류 (SVM)
from sklearn.svm import SVC

model_cl_svm = SVC(kernel="linear", C=100)

model_cl_svm.fit(X_train,y_train)

y_pred_cl_svm=model_cl_svm.predict(X_test)

print 'Accuracy: {0:.2f}%'.format(accuracy_score(y_test, y_pred_cl_svm)*100)

print "precision_score : {0:.2f}%".format(metrics.precision_score(y_test, y_pred_cl_svm,average='weighted')*100)

print "recall_score : {0:.2f}%".format(metrics.recall_score(y_test, y_pred_cl_svm,average='weighted')*100)

print "f1_score : {0:.2f}%".format(metrics.f1_score(y_test, y_pred_cl_svm,average='weighted')*100)



# 회귀 (의사결정 나무)

from sklearn.tree import DecisionTreeRegressor

model_reg_dt=DecisionTreeRegressor(max_depth=500)
model_reg_dt.fit(X_train2,y_train2)

y_pred_reg_dt = model_reg_dt.predict(X_test2)

sse = 0
for i,x in enumerate(y_pred_reg_dt):
    sse += (float(y_test2.get_values()[i])-float(y_pred_reg_dt[i]))**2

print "SSE:", sse

y_test2_list=[]
for i,x in enumerate(y_test2):
    y_test2_list.append(x)


sum=0
for i, x in enumerate(y_test2_list):
    sum+= x

mean_y = sum/(i+1)

sst = 0
for i,x in enumerate(y_pred_reg_dt):
    sst += (float(y_test2.get_values()[i])- mean_y)**2

print "SST:", sst

Rsquared = (1-sse/sst)

print "결정계수:", Rsquared




# 회귀 (랜덤 포레스트)

from sklearn.ensemble import RandomForestRegressor

model_reg_rf=RandomForestRegressor(n_estimators=1000, criterion='mse',random_state=1, n_jobs=-1)
model_reg_rf.fit(X_train2,y_train2)

y_pred_reg_rf = model_reg_rf.predict(X_test2)



sse = 0
for i,x in enumerate(y_pred_reg_rf):
    sse += (float(y_test2.get_values()[i])-float(y_pred_reg_rf[i]))**2

print "SSE:", sse

y_test2_list=[]
for i,x in enumerate(y_test2):
    y_test2_list.append(x)

sum=0
for i, x in enumerate(y_test2_list):
    sum+= x

mean_y = sum/(i+1)

sst = 0
for i,x in enumerate(y_pred_reg_rf):
    sst += (float(y_test2.get_values()[i])- mean_y)**2

print "SST:", sst

Rsquared = (1-sse/sst)

print "결정계수:", Rsquared




# 회귀 (KNN)

from sklearn.neighbors import KNeighborsRegressor

model_reg_knn = KNeighborsRegressor(n_neighbors=5, p=2, metric='minkowski')
model_reg_knn.fit(X_train2,y_train2)

y_pred_reg_knn = model_reg_knn.predict(X_test2)

sse = 0
for i,x in enumerate(y_pred_reg_knn):
    sse += (float(y_test2.get_values()[i])-float(y_pred_reg_knn[i]))**2

print "SSE:", sse

sum=0
for i, x in enumerate(y_test2_list):
    sum+= x

mean_y = sum/(i+1)

sst = 0
for i,x in enumerate(y_pred_reg_knn):
    sst += (float(y_test2.get_values()[i])- mean_y)**2

print "SST:", sst

Rsquared = (1-sse/sst)

print "결정계수:", Rsquared




# 회귀 (SVM)

from sklearn.svm import SVR

model_reg_svm = SVR(kernel='linear', C=100)

model_reg_svm.fit(X_train2,y_train2)

y_pred_reg_svm = model_reg_svm.predict(X_test2)

sse = 0
for i,x in enumerate(y_pred_reg_svm):
    sse += (float(y_test2.get_values()[i])-float(y_pred_reg_svm[i]))**2

print "SSE:", sse

sum=0
for i, x in enumerate(y_test2_list):
    sum+= x

mean_y = sum/(i+1)

sst = 0
for i,x in enumerate(y_pred_reg_svm):
    sst += (float(y_test2.get_values()[i])- mean_y)**2

print "SST:", sst

Rsquared = (1-sse/sst)

print "결정계수:", Rsquared