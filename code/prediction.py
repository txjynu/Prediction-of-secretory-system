from evaluation_indexRF import *
from helpers import *
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier # 导入k近邻算法
import pandas as pd
from sklearn.linear_model import LogisticRegression  #导入逻辑回归模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import xgboost
import lightgbm
import numpy as np
pd.set_option('display.max_rows',None)#取消行限制
pd.set_option('display.max_columns',None)#取消列限制
pd.set_option('display.width',1000)#增加每行的宽度
np.set_printoptions(threshold=np.inf)

np.set_printoptions(linewidth=1000) #矩阵在pycharm中全显示（不自动换行）
def load_data(file):
    lista = []
    records = list(open(file, "r"))
    records = records[1:]
    for seq in records:
        elements = seq.split("\t")
        level = elements[0].split("\n")
        classe = level[0]
        lista.append(classe)

    lista = set(lista)
    classes = list(lista)
    #print("class",classes)
    X = []
    Y = []
    for seq in records:
        elements = seq.strip().split("\t")
        # print("elements",elements)
        X.append(elements[1:])
        level = elements[0].split("\n")
        classe = level[0]
        ##Y.append(ciao.index(classe))
        Y.append(classe)
        #print(Y)

    X = np.array(X, dtype=float)
    #print("****************")
    Y = np.array(Y, dtype=int)
    #print("y", Y)

    return X, Y, len(classes), len(X[0])

def evaluation(model,x,y_label):
    print("model——————", model)
    preds = model.predict_proba(x)
    preds1 = model.predict(x)
    #print(model.predict_classes(x))
    np.set_printoptions(suppress=True)
    print(len(preds))

    print(preds1)
    print(preds)
    #print(np.argmax(preds,axis=1))



import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

X, labels, nb_classes, input_length = load_data('D:\lunwen\Multi classification\code\Multi\data//all.tsv' )

print('Generating labels and features...')
print('Shuffling the data...')
index = np.arange(len(labels))
np.random.shuffle(index)
# np.random.seed(300)
X_train1 = X[index, :]
y_train = labels[index]
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train1)

path="D:\lunwen\Multi classification\code\Multi\data//feature//genome_assemblies//"  #待读取的文件夹
path_list=os.listdir(path)
path_list.sort() #对读取的路径进行排序
for filename in path_list:
    print('filename:',os.path.join(path,filename))
    X,labels, nb_classes, input_length = load_data('D:\lunwen\Multi classification\code\Multi\data//feature//genome_assemblies//'+filename)
    
    print('Generating labels and features...')
    print('Shuffling the data...')
    index=np.arange(len(labels))
    np.random.shuffle(index)
    #np.random.seed(300)
    X_test1=X[index,:]
    y_test=labels[index]
    min_max_scaler = MinMaxScaler()
    X_test = min_max_scaler.fit_transform(X_test1)

    DT = tree.DecisionTreeClassifier().fit(X_train, y_train)

    NB = GaussianNB().fit(X_train, y_train)
    KNN = KNeighborsClassifier().fit(X_train, y_train)
    RF = RandomForestClassifier().fit(X_train, y_train)
    clf1 = svm.SVC(kernel='poly', C=1,probability=True).fit(X_train, y_train)
    #evaluation(clf1, X_test, y_test)
    clf2 = svm.SVC(kernel='linear', C=1,probability=True).fit(X_train, y_train)
    MLP = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=5000).fit(X_train, y_train)
    XGB = xgboost.XGBClassifier(learning_rate=0.16,  # 通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2。
                                n_estimators=200,  # 200
                                max_depth=6,
                                # 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。 需要使用CV函数来进行调优。 典型值：3-10
                                min_child_weight=1,
                                # 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整
                                gamma=0,
                                # Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
                                subsample=0.8,
                                # 控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1
                                colsample_bytree=0.8,  # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
                                objective='binary:logistic',
                                # binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。 multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。num_class(类别数目)。 multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
                                nthread=4,  # 这个参数用来进行多线程控制，应当输入系统的核数。 如果你希望使用CPU全部的核，那就不要输入这个参数，算法会自动检测它。
                                seed=36)
    xgb=XGB.fit(X_train, y_train)
    lgb = lightgbm.LGBMClassifier().fit(X_train, y_train)

    model2 = VotingClassifier(
        estimators=[('svm', clf1), ('linear', clf2), ('xgb', xgb), ('lgb', lgb), ('rf', RF), ('mlp', MLP)],
        voting='soft')
    model2.fit(X_train, y_train)


    #print('DT准确率：', DT.score(X_test, y_test))
    #print('NB准确率：', NB.score(X_test, y_test))
    #print('KNN准确率：', KNN.score(X_test, y_test))
    #print('RF准确率：', RF.score(X_test, y_test))
    #print('SVM_poly准确率：', clf1.score(X_test, y_test))  # 计算测试集的度量值（准确率）
    #print('SVM_linear准确率：', clf2.score(X_test, y_test))  # 计算测试集的度量值（准确率）
    #print('MLP准确率：', MLP.score(X_test, y_test))  # 计算测试集的度量值（准确率）
    #print('xgb准确率：', xgb.score(X_test, y_test))
    #print('lgb：', lgb.score(X_test, y_test))

    #print('xgb--svm准确率：',model1.score(X_test, y_test))  # 0.866
    print('xgb--svm准确率：', model2.score(X_test, y_test))

   # evaluation(model1, X_test, y_test)
    print("*********************************************************")

    evaluation(model2, X_test, y_test)






