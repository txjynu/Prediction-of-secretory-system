from evaluation_indexRF import *
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import xgboost
import lightgbm


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

def score(model):
    print("model——————",model)
    print('accuracy：', cross_val_score(model, x, y_label, cv=10, scoring='accuracy').mean())
    print('balanced_accuracy：', cross_val_score(model, x, y_label, cv=10, scoring='balanced_accuracy').mean())
    print('f1_weighted：', cross_val_score(model, x, y_label, cv=10, scoring='f1_weighted').mean())
    print('f1_micro：', cross_val_score(model, x, y_label, cv=10, scoring='f1_micro').mean())
    print('f1_macro：', cross_val_score(model, x, y_label, cv=10, scoring='f1_macro').mean())
    print('precision_weighted：', cross_val_score(model, x, y_label, cv=10, scoring='precision_weighted').mean())
    print('precision_micro：', cross_val_score(model, x, y_label, cv=10, scoring='precision_micro').mean())
    print('precision_macro：', cross_val_score(model, x, y_label, cv=10, scoring='precision_macro').mean())
    print('recall_weighted：', cross_val_score(model, x, y_label, cv=10, scoring='recall_weighted').mean())
    print('recall_micro：', cross_val_score(model, x, y_label, cv=10, scoring='recall_micro').mean())
    print('recall_macro：', cross_val_score(model, x, y_label, cv=10, scoring='recall_macro').mean())


def evaluation(model,x,y_label):
    print("model——————", model)
    preds = cross_val_predict(model, x, y_label, cv=10)
    confusion_matrix1 = confusion_matrix(y_label, preds)
    print("confusion_matrix:", confusion_matrix1)

    ma = matthews_corrcoef(y_label, preds)
    print("ma:", ma)

    FP = confusion_matrix1.sum(axis=0) - np.diag(confusion_matrix1)
    FN = confusion_matrix1.sum(axis=1) - np.diag(confusion_matrix1)
    TP = np.diag(confusion_matrix1)
    TN = confusion_matrix1.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    print("sum:", confusion_matrix1.sum())
    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)

    # 3-其他的性能参数的计算
    TPR = TP / (TP + FN)  # Sesnitivity/ hit rate/ recall/ true positive rate
    TNR = TN / (TN + FP)  # Specificity/ true negative rate
    PPV = TP / (TP + FP)  # Precision/ positive predictive value
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # Fall out/ false positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # accuracy of each class\
    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)
    # mcc=math.sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # MCC=(TP*TN-FP*FN)/mcc

    print("Sesnitivity:", TPR)
    print("Specificity:", TNR)
    print("Precision", PPV)
    print("NPV:", NPV)
    print("FPR:", FPR)
    print("FNR:", FNR)
    print("FDR:", FDR)
    print("accuracy:", ACC)
    print("F1:", F1)
    t = classification_report(y_label, preds, target_names=['1', '2', '3', '4', '5', '6', '7', '8'])
    print("report:", t)


import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from minepy import MINE
from sklearn.feature_selection import SelectKBest

def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)

path="D:\lunwen\Multi classification\code\Multi\data//feature//"  #待读取的文件夹
path_list=os.listdir(path)
path_list.sort() #对读取的路径进行排序
for filename in path_list:
    print('filename:',os.path.join(path,filename))
    X,labels, nb_classes, input_length = load_data('D:\lunwen\Multi classification\code\Multi\data//feature//'+filename)
   
    print('Generating labels and features...')
    print('Shuffling the data...')
    index=np.arange(len(labels))
    np.random.shuffle(index)
    #np.random.seed(300)
    x1=X[index,:]
    Y=labels[index]
    y_label = Y
    #MLP = MLPClassifier(hidden_layer_sizes=(30, 50), max_iter=300)
    #print('MLP准确率：', cross_val_score(MLP, x, y_label, cv=10))

    min_max_scaler = MinMaxScaler()
    x_scaler = min_max_scaler.fit_transform(x1)
    for i in range(300, 900, 50):
        x = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: mic(x, Y), X.T))).T)),
                        k=i).fit_transform(x_scaler, y_label)

        print(x_scaler.shape, x.shape)

        DT = tree.DecisionTreeClassifier()
        print('DT准确率：', cross_val_score(DT, x, y_label, cv=10))

        NB = GaussianNB()
        print('NB准确率：', cross_val_score(NB, x, y_label, cv=10))

        KNN = KNeighborsClassifier()
        print('KNN准确率：', cross_val_score(KNN, x, y_label, cv=10))

        RF = RandomForestClassifier()
        print('RF准确率：', cross_val_score(RF, x, y_label, cv=10))

        clf1 = svm.SVC(kernel='poly', C=1, probability=True)
        print('SVM_poly准确率：', cross_val_score(clf1, x, y_label, cv=10))

        clf2 = svm.SVC(kernel='linear', C=1, probability=True)
        print('SVM_linear准确率：', cross_val_score(clf2, x, y_label, cv=10))

        MLP = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=5000)
        # print('MLP准确率：', cross_val_score(MLP, x, y_label, cv=5))

        xgb = xgboost.XGBClassifier(learning_rate=0.16,  # 通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2。
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
        # print('xgb准确率：', cross_val_score(xgb, x, y_label, cv=5))

        lgb = lightgbm.LGBMClassifier()
        # print('lgb准确率：', cross_val_score(lgb, x, y_label, cv=5))
        score(DT)
        evaluation(DT, x, y_label)
        score(NB)
        evaluation(NB, x, y_label)
        score(KNN)
        evaluation(KNN, x, y_label)
        score(RF)
        evaluation(RF, x, y_label)
        score(clf1)
        evaluation(clf1, x, y_label)
        score(clf2)
        evaluation(clf2, x, y_label)

        score(MLP)
        evaluation(MLP, x, y_label)
        score(lgb)
        evaluation(lgb, x, y_label)
        score(xgb)
        evaluation(xgb, x, y_label)

        model1 = VotingClassifier(estimators=[('svm1', clf1), ('svm2', clf2), ('lgb', lgb), ('rf', RF), ('mlp', MLP)],
                                  voting='hard')
        #print('xgb-lgb-svm-rf准确率：', cross_val_score(model1, x, y_label, cv=10))  # 0.831
      
        score(model1)
        evaluation(model1, x, y_label)
       


