#coding=utf-8
import copy
import csv
import logging

import matplotlib.pyplot as plt
# import lightgbm as lgb
from AAAA_model_xin.feature import  feature, cols_data, details, redundancy, rmdup, cols_data_rmdup
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import learning_curve, validation_curve
# from sklearn.learning_curve import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB

logging.getLogger().setLevel(logging.INFO)

import logging
logging.getLogger().setLevel(logging.INFO)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = '4' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically




class Model:
    def __init__(self):
        self.count = 0
        self.Dicts = {}

    @staticmethod
    def DataInit(evalute=1):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        if evalute == 0:
            # df = pd.read_csv("D:\H\chi_paper_xin\Os14\Feature_select\Sample_label_feature_MFE_onehot_zscore_remove.csv")
            # df = pd.read_csv("D:\H\chi_paper_xin\sly14_xin\sample_label_feature_onehot _noseq_sly14_zscore4_remove.csv")
            df = pd.read_csv("D:\H\chi_paper_xin\potato14\sample_class_feature_onehot_nosequence_zscore_remove.csv")

            cols = list(df.columns.values)
            cols_data = copy.deepcopy(cols)
            cols_data.remove('class')
            x_data = df[list(cols_data)]
            x = np.array(x_data)
            # x = StandardScaler().fit_transform(x)
            y_data = df['class']
            y = np.array(y_data)
            data = np.insert(x, x[0].size, values=y, axis=1)
            #########################
            # np.random.shuffle(data)#打乱顺序
            y = data[:, data[0].size - 1]
            x = np.delete(data, -1, axis=1)
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=3)
            with open("D:\H\chi_paper_xin\sly14_xin\sample_label_feature_onehot _noseq_sly14_zscore6_remove.csv", "w") as ft:
                f_csv = csv.writer(ft)
                f_csv.writerow(cols)
                f_csv.writerows(data)
            # return
        elif evalute==1:
            # dftrain = pd.read_csv("../Script/Mapping/Mfe/SamTr.csv")
            dftrain = pd.read_csv("D:\H\chi_paper_xin\potato14\sample_class_feature_onehot_nosequence_zscore_remove_tr.csv")
            cols = list(dftrain.columns.values)
            cols_data = copy.deepcopy(cols)
            cols_data.remove('class')
            #cols_data.remove('sequence')
            # for i in rmdup:
            #     cols_data.remove(i)
            x_data = dftrain[list(cols_data)]
            x = np.array(x_data)
            # X_train = x
            X_train = StandardScaler().fit_transform(x)
            y_data = dftrain['class']
            Y_train = np.array(y_data)

            # dftest = pd.read_csv("../Script/Mapping/Mfe/SamTe.csv")
            dftest = pd.read_csv("D:\H\chi_paper_xin\potato14\sample_class_feature_onehot_nosequence_zscore_remove_te.csv")
            cols = list(dftest.columns.values)
            cols_data = copy.deepcopy(cols)
            cols_data.remove('class')
            #cols_data.remove('sequence')
            # for i in rmdup:
            #     cols_data.remove(i)
            x_data = dftest[list(cols_data)]
            x = np.array(x_data)
            # X_test = x
            X_test = StandardScaler().fit_transform(x)
            y_data = dftest['class']
            Y_test = np.array(y_data)
        elif evalute== 2:
            df = pd.read_csv("D:\H\chi_paper_xin\potato14\sample_class_feature_onehot_nosequence_zscore_remove.csv")
            cols = list(df.columns.values)
            cols_data = copy.deepcopy(cols)
            cols_data.remove('class')
            x_data = df[list(cols_data)]
            x = np.array(x_data)
            x = StandardScaler().fit_transform(x)
            y_data = df['class']
            y = np.array(y_data)
            data = np.insert(x, x[0].size, values=y, axis=1)
            # np.random.shuffle(data)
            y = data[:, data[0].size - 1]
            x = np.delete(data, -1, axis=1)
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=1)
            data_train = np.insert(X_train, X_train[0].size, values=Y_train, axis=1)
            data_test = np.insert(X_test, X_test[0].size, values=Y_test, axis=1)
            with open("D:\H\chi_paper_xin\potato14\sample_class_feature_onehot_nosequence_zscore_remove_tr.csv", "w") as f:
                f_csv = csv.writer(f)
                f_csv.writerow(cols)
                f_csv.writerows(data_train)
            with open("D:\H\chi_paper_xin\potato14\sample_class_feature_onehot_nosequence_zscore_remove_te.csv", "w") as ft:
                f_csv = csv.writer(ft)
                f_csv.writerow(cols)
                f_csv.writerows(data_test)
            # return
        else:
            return "输入不合法"
        #
        return X_train, X_test, Y_train, Y_test


    # def lossResult(mod, X_train, Y_train, X_test, Y_test,m_name):
    #     # case1：学习曲线
    #     # 构建学习曲线评估器，train_sizes：控制用于生成学习曲线的样本的绝对或相对数量
    #     train_sizes, train_scores, test_scores = learning_curve(estimator=mod, X=X_train, y=Y_train,
    #                                                             train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    #     # 统计结果
    #     train_mean = np.mean(train_scores, axis=1)
    #     train_std = np.std(train_scores, axis=1)
    #     test_mean = np.mean(test_scores, axis=1)
    #     test_std = np.std(test_scores, axis=1)
    #     # 绘制效果
    #     plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    #     plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    #     plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    #     plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    #     plt.grid()
    #     plt.xlabel('Number of training samples')
    #     plt.ylabel('Accuracy')
    #     plt.legend(loc='lower right')
    #     plt.ylim([0.8, 1.0])
    #     plt.show()
    #
    #     # case2：验证曲线
    #     param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    #     # 10折，验证正则化参数C
    #     train_scores, test_scores = validation_curve(estimator=mod, X=X_train, y=Y_train, param_name='clf__C',
    #                                                  param_range=param_range, cv=10)
    #     # 统计结果
    #     train_mean = np.mean(train_scores, axis=1)
    #     train_std = np.std(train_scores, axis=1)
    #     test_mean = np.mean(test_scores, axis=1)
    #     test_std = np.std(test_scores, axis=1)
    #     plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    #     plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    #     plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    #     plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    #     plt.grid()
    #     plt.xscale('log')
    #     plt.xlabel('Parameter C')
    #     plt.ylabel('Accuracy')
    #     plt.legend(loc='lower right')
    #     plt.ylim([0.8, 1.0])
    #     plt.show()

    @staticmethod
    def FigResult(mod, X_train, Y_train, X_test, Y_test,m_name):
        print(m_name)
        Y_test_predict = mod.predict(X_test)
        Y_train_predict = mod.predict(X_train)
        acc_scores = cross_val_score(mod, X_train, Y_train, cv=5, scoring='accuracy')
        rec_scores = cross_val_score(mod, X_train, Y_train, cv=5, scoring='recall')
        fon_scores = cross_val_score(mod, X_train, Y_train, cv=5, scoring='f1')
        pre_scores = cross_val_score(mod, X_train, Y_train, cv=5, scoring='precision')
        # return [list(acc_scores), list(rec_scores), list(pre_scores), list(fon_scores)]
        print([list(acc_scores), list(rec_scores), list(pre_scores), list(fon_scores)])
        print("十折交叉平均值(accuracy) ：", acc_scores.mean())
        print("十折交叉平均值(recall)   ：", rec_scores.mean())
        print("十折交叉平均值(f1)       ：", fon_scores.mean())
        print("十折交叉平均值(precision)：", pre_scores.mean())

        print("accuracy ：", accuracy_score(Y_test, Y_test_predict),"-", accuracy_score(Y_train, Y_train_predict))
        print("recall   ：", recall_score(Y_test, Y_test_predict),"-", recall_score(Y_train, Y_train_predict))
        print("f1       ：", f1_score(Y_test, Y_test_predict),"-", f1_score(Y_train, Y_train_predict))
        print("precision：", precision_score(Y_test, Y_test_predict),"-", precision_score(Y_train, Y_train_predict))

        Y_test_predict_pro = mod.predict_proba(X_test)
        y_scores = pd.DataFrame(Y_test_predict_pro, columns=mod.classes_.tolist())[1].values
        auc_value = roc_auc_score(Y_test, y_scores)
        Y_train_predict_pro = mod.predict_proba(X_train)
        y_train_scores = pd.DataFrame(Y_train_predict_pro, columns=mod.classes_.tolist())[1].values
        auc_value_train = roc_auc_score(Y_train, y_train_scores)
        print("auc  :", auc_value,"-", auc_value_train)
        # print(classification_report(Y_test, Y_test_predict))
        # print(classification_report(Y_train, Y_train_predict))
        fpr, tpr, thresholds = roc_curve(Y_test, y_scores, pos_label=1.0)
        plt.figure(figsize=(6.4, 6.4))
        plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % auc_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(m_name)
        plt.legend(loc="lower right")
        plt.show()


    @staticmethod
    #def FigAllImg(m5, m6, m7, f, t, s, X_train, Y_train, X_test, Y_test):
    def FigAllImg(m1,m2,m3,m4, m5, X_train, Y_train, X_test, Y_test):

        Y_tpp_1 = m1.predict_proba(X_test)
        y_s_1 = pd.DataFrame(Y_tpp_1, columns=m1.classes_.tolist())[1].values
        auc_1 = roc_auc_score(Y_test, y_s_1)
        print(auc_1)
        fpr1, tpr1, thresholds1 = roc_curve(Y_test, y_s_1, pos_label=1.0)

        Y_tpp_2 = m2.predict_proba(X_test)
        y_s_2 = pd.DataFrame(Y_tpp_2, columns=m2.classes_.tolist())[1].values
        auc_2 = roc_auc_score(Y_test, y_s_2)
        print(auc_2)
        fpr2, tpr2, thresholds2 = roc_curve(Y_test, y_s_2, pos_label=1.0)

        Y_tpp_3 = m3.predict_proba(X_test)
        y_s_3 = pd.DataFrame(Y_tpp_3, columns=m3.classes_.tolist())[1].values
        auc_3 = roc_auc_score(Y_test, y_s_3)
        print(auc_3)
        fpr3, tpr3, thresholds3 = roc_curve(Y_test, y_s_3, pos_label=1.0)

        Y_tpp_4 = m4.predict_proba(X_test)
        y_s_4 = pd.DataFrame(Y_tpp_4, columns=m4.classes_.tolist())[1].values
        auc_4 = roc_auc_score(Y_test, y_s_4)
        print(auc_4)
        fpr4, tpr4, thresholds4 = roc_curve(Y_test, y_s_4, pos_label=1.0)

        Y_tpp_5 = m5.predict_proba(X_test)
        y_s_5 = pd.DataFrame(Y_tpp_5, columns=m5.classes_.tolist())[1].values
        auc_5 = roc_auc_score(Y_test, y_s_5)
        print(auc_5)
        fpr5, tpr5, thresholds5 = roc_curve(Y_test, y_s_5, pos_label=1.0)

        # Y_tpp_6 = m6.predict_proba(X_test)
        # y_s_6 = pd.DataFrame(Y_tpp_6, columns=m6.classes_.tolist())[1].values
        # auc_6 = roc_auc_score(Y_test, y_s_6)
        # print(auc_6)
        # fpr6, tpr6, thresholds6 = roc_curve(Y_test, y_s_6, pos_label=1.0)
        #
        # Y_tpp_7 = m7.predict_proba(X_test)
        # y_s_7 = pd.DataFrame(Y_tpp_7, columns=m7.classes_.tolist())[1].values
        # auc_7 = roc_auc_score(Y_test, y_s_7)
        # print(auc_7)
        # fpr7, tpr7, thresholds7 = roc_curve(Y_test, y_s_7, pos_label=1.0)


        plt.figure(figsize=(8, 8))
        plt.plot(fpr1, tpr1, 'b-', label='GBDT AUC = %0.4f' % auc_1)
        plt.plot(fpr2, tpr2, 'g-', label='RF AUC = %0.4f' % auc_2)
        plt.plot(fpr3, tpr3, 'y-', label='Adaboost AUC = %0.4f' % auc_3)
        plt.plot(fpr4, tpr4, 'k-', label='XGBoost AUC = %0.4f' % auc_4)
        plt.plot(fpr5, tpr5, 'r-', label='LightGBM AUC = %0.4f' % auc_5)
        # plt.plot(fpr6, tpr6, 'k-', label='SVM AUC = %0.4f' % auc_6)
        # plt.plot(fpr7, tpr7, 'orange', label='XGB AUC = %0.4f' % auc_7)
        # plt.plot(f, t, color="orange", label='MLP AUC = %0.4f' % s)

        #plt.plot(fpr1, tpr1, 'r-', label='KNN')
        #plt.plot(fpr2, tpr2, 'k-', label='BAY')
        #plt.plot(fpr3, tpr3, 'y-', label='DTR')
        #plt.plot(fpr4, tpr4, 'g-', label='LRG')
        #plt.plot(fpr5, tpr5, 'c-', label='RFT')
        #plt.plot(fpr6, tpr6, 'b-', label='SVM')
        #plt.plot(fpr7, tpr7, 'm-', label='XGB')
       # plt.plot(f, t, color="orange", label='MLP')

        plt.plot([0, 1], [0, 1], color='r', linestyle='--')
        plt.plot([0, 0], [0, 1], color='r', linestyle='--')
        plt.plot([1, 0], [1, 1], color='r', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of All Model')
        plt.legend(loc="lower right")
        plt.savefig("sly_model_300.png",dpi=300)
        plt.savefig("sly_model_600.png", dpi=600)
        plt.show()

    # def KnnDemo(self, X_train, X_test, Y_train, Y_test):
    #     from sklearn.neighbors import KNeighborsClassifier
    #
    #     # mod = KNeighborsClassifier(n_neighbors=19)#水稻
    #     mod = KNeighborsClassifier(n_neighbors=18)  # 马铃薯
    #     # mod = KNeighborsClassifier(n_neighbors=45)  # 番茄
    #     mod.fit(X_train, Y_train)
    #     Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "KNN")
    #     return mod
    #     from sklearn.neighbors import KNeighborsClassifier
    #
    #     from sklearn import neighbors
    #     K = np.arange(1, 51)
    #     length = len(K)
    # # 构建空的列表，用于存储平均准确率
    #     accuracy = []
    #     for k in K:
    #     # 使用10重交叉验证的方法，比对每一个k值下KNN模型的预测准确率
    #         cv_result = model_selection.cross_val_score(
    #         neighbors.KNeighborsClassifier(n_neighbors=int(k), weights='distance'),
    #         X_train, Y_train, cv=10, scoring='accuracy')
    #         accuracy.append(cv_result.mean())
    #
    # # 从k个平均准确率中挑选出最大值所对应的下标
    #     arg_max = np.array(accuracy).argmax()
    # # 中文和负号的正常显示
    #     plt.rcParams['font.sans-serif'] = [u'SimHei']
    #     plt.rcParams['axes.unicode_minus'] = False
    # # 绘制不同K值与平均预测准确率之间的折线图
    #     plt.plot(K, accuracy,color = 'y')
    # # 添加点图
    #     plt.scatter(K, accuracy)
    # # 添加文字说明
    #     plt.text(K[arg_max], accuracy[arg_max], '最佳k值为%s' % int(K[arg_max]))
    #     plt.xticks(np.arange(51), np.arange(51))
    # # 显示图形
    #     plt.show()




    # def Bysdemo(self, X_train, X_test, Y_train, Y_test):
    #     # GaussianNB 参数只有一个：先验概率priors; 参数的意义主要参考https://www.cnblogs.com/pinard/p/6074222.html
    #     from sklearn.naive_bayes import BernoulliNB
    #     # # #水稻、马铃薯、番茄
    #     mod = BernoulliNB()
    #     mod.fit(X_train, Y_train)
    #     Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "Bays")
    #     return mod
    #
    #     #宋
    #     # mod = GaussianNB()
    #     # mod.fit(X_train, Y_train)
    #     # Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "Bays")
    #     # return mod
    #
    #     option = [GaussianNB, BernoulliNB]
    #     evalution = ['accuracy', 'recall', 'precision', 'f1']
    #
    #     scores = [[], [], [], []]
    #     count = -1
    #
    #     for e in evalution:
    #         count += 1
    #         for o in option:
    #             mod = o()
    #             mod.fit(X_train, Y_train)
    #             scores[count].append(np.mean(cross_val_score(mod, X_train, Y_train, cv=5, scoring=e)))
    #
    #     print(scores)
    #     plt.xlabel('Evaluation Criteria')
    #     plt.ylabel('Value')
    #     plt.ylim(0, 1)
    #     # num_list = [0.7673668360624883, 0.7739463601532567, 0.7693773688785859, 0.7714057714669738]
    #     x = list(range(len(evalution)))
    #     total_width, n = 0.8, 3
    #     width = total_width / n
    #     # plt.bar(x, num_list, width=width, label='Multinomial', fc='b')
    #     for i in range(len(x)):
    #         x[i] += width
    #     num_list3 = [x[1] for x in scores]
    #     plt.bar(x, num_list3, width=width, label='Bernoulli', fc='y')
    #     for i in range(len(x)):
    #         x[i] += width
    #     num_list2 = [x[0] for x in scores]
    #     plt.bar(x, num_list2, width=width, label='Gaussian', tick_label=evalution, fc='orange')
    #
    #     plt.legend()
    #     plt.show()


    # def Dtree(self, X_train, X_test, Y_train, Y_test):
    #     from sklearn import tree
    #     #水稻
    #     # mod = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)#水稻
    #     # mod = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)  # 番茄
    #     mod = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)  # 马铃薯
    #     mod = mod.fit(X_train, Y_train)
    #     Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "Dtree")
    #     return mod
    #     # # 马铃薯
    #
    #     # mod = tree.DecisionTreeClassifier(criterion='gini', max_depth=5)
    #     # mod = mod.fit(X_train, Y_train)
    #     # Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "Dtree")
    #     # return mod
    #     # mod = mod.fit(X_train, Y_train)
    #     # Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "Dtree")
    #     # return mod
    #     # #番茄
    #     # mod = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
    #     # mod = mod.fit(X_train, Y_train)
    #     # Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "Dtree")
    #     # return mod
    #
    #     scores_en = []
    #     scores_gn = []
    #     for i in range(1, 50):
    #         mod = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    #         mod.fit(X_train, Y_train)
    #         scores_en.append(np.mean(cross_val_score(mod, X_test, Y_test, cv=3, scoring='accuracy')))
    #     print(scores_en)
    #     plt.plot(range(len(scores_en)), scores_en, label='entropy', marker='.',color='y')
    #     for i in range(1, 50):
    #         mod = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)
    #         mod.fit(X_train, Y_train)
    #         scores_gn.append(np.mean(cross_val_score(mod, X_test, Y_test, cv=3, scoring='accuracy')))
    #     print(scores_gn)
    #     plt.plot(range(len(scores_gn)), scores_gn, label='gini', marker='.', color='orange')
    #     plt.xlabel("max_depth")
    #     plt.legend(loc=4)
    #     plt.show()
    #     return

    # def LogisticRegression(self, X_train, X_test, Y_train, Y_test):
    #     from sklearn.linear_model import LogisticRegression as LR
    #     # mod = LR(penalty="l1", solver="liblinear")
    #     # mod.fit(X_train, Y_train)
    #     # Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "LR")
    #     # return mod
    #
    #     lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
    #     lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)
    #
    #     lrl1 = lrl1.fit(X_train, Y_train)
    #     lrl2 = lrl2.fit(X_train, Y_train)
    #
    #     (lrl1.coef_ != 0).sum(axis=1)  # l1正则化，参数被稀疏化，可以防止过拟合
    #     (lrl2.coef_ != 0).sum(axis=1)  # l2正则化，非零稀疏比较多
    #     # 调参数
    #     l1test = []
    #     l2test = []
    #     l1 = []
    #     l2 = []
    #
    #     for i in np.linspace(0.05, 2, 39):
    #         lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
    #         lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
    #
    #         lrl1 = lrl1.fit(X_train, Y_train)
    #         l1.append(accuracy_score(lrl1.predict(X_train), Y_train))
    #         l1test.append(accuracy_score(lrl1.predict(X_test), Y_test))
    #
    #         lrl2 = lrl2.fit(X_train, Y_train)
    #         l2.append(accuracy_score(lrl2.predict(X_train), Y_train))
    #         l2test.append(accuracy_score(lrl2.predict(X_test), Y_test))
    #
    #     print(l1test)
    #     print(l2test)
    #     graph = [l1, l2, l1test, l2test]
    #     color = ["darkblue", "darkgreen", "deepskyblue", "turquoise"]
    #     label = ["L1 train", "L2 train", "L1 test", "L2 test"]
    #
    #     for i in range(len(graph)):
    #         plt.plot(np.linspace(0.05, 2, 39), graph[i], color[i], label=label[i])
    #     plt.xlabel("Regularization strength")
    #     plt.legend()  # loc = 4, 表示右下角
    #     plt.show()
    def GBDTClassdemo(self,X_train,X_test,Y_train,Y_test):
        from sklearn.ensemble import GradientBoostingClassifier
        # mod = GradientBoostingClassifier()
        # mod = GradientBoostingClassifier(learning_rate=0.1,n_estimators=20, max_depth=3,
        #                                  max_features='auto', subsample=0.6,
        #                                 )


        #
        # # 十折交叉平均值(accuracy) ： 0.8398496240601503
        # # 十折交叉平均值(recall)   ： 0.9407963504296217
        # # 十折交叉平均值(f1)       ： 0.898090431312996
        # # 十折交叉平均值(precision)： 0.869842498991677
        # # accuracy ： 0.8468468468468469 - 0.8834586466165414
        # # recall   ： 0.9267605633802817 - 0.9680451127819549
        # # f1       ： 0.9063360881542699 - 0.9300225733634312
        # # precision： 0.8867924528301887 - 0.894874022589053
        # # auc: 0.8471118847918975 - 0.9170618039459552
        #水稻2021、7、12
        # mod = GradientBoostingClassifier(learning_rate=0.1,n_estimators=25, max_depth=4,
        #                                  max_features='auto', subsample=0.5,
        #                                 )
        #番茄 2021、7、12
        # mod = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50, max_depth=4,
        #                                  max_features='auto', subsample=0.5,
        #                                 )
        # accuracy ： 0.9118236472945892 - 0.9605614973262032
        # recall   ： 0.9921671018276762 - 0.9975267930750206
        # f1       ： 0.945273631840796 - 0.9762000806776926
        # precision： 0.9026128266033254 - 0.9557661927330173
        # auc: 0.9269604753758891 - 0.9932183442622473
        # mod = GradientBoostingClassifier(learning_rate=0.1, n_estimators=49, max_depth=5,
        #                                  max_features='sqrt', subsample=0.8,
        #                                  )
        # accuracy ： 0.9178356713426854 - 0.9538770053475936
        # recall   ： 0.9825 - 0.9991638795986622
        # f1       ： 0.950423216444982 - 0.9719398129320862
        # precision： 0.9203747072599532 - 0.946159936658749
        # auc: 0.9436111111111112 - 0.9971488294314381


        #番茄选这个2021、7、13
        # mod = GradientBoostingClassifier(learning_rate=0.1, n_estimators=49, max_depth=5,
        #                                  max_features='sqrt', subsample=0.8,
        #                                  )

        # accuracy ： 0.9218436873747495 - 0.9505347593582888
        # recall   ： 0.99 - 0.9983277591973244
        # f1       ： 0.9530685920577617 - 0.9699431356620634
        # precision： 0.9187935034802784 - 0.943127962085308
        # auc: 0.9465656565656566 - 0.9983639910813824
        #水稻
        # mod = GradientBoostingClassifier(
        #     # boosting参数
        #     # init=None,
        #     n_estimators=31,
        #     learning_rate=0.1,
        #     subsample=0.6,
        #     loss='deviance',
        #     # 分割参数
        #     max_features='auto',
        #     # criterion='friedman_mse',
        #     # 分割停止参数
        #     # min_samples_split=1200,
        #     # min_impurity_split=None,
        #     # min_impurity_decrease=0.0,
        #     max_depth=7,
        #     # max_leaf_nodes=None,
        #     # 剪枝参数
        #     # min_samples_leaf=60,
        #     # warm_start=False,
        #     # random_state=10
        # )
        # 番茄
        # mod = GradientBoostingClassifier(
        #     # boosting参数
        #     # init=None,
        #     n_estimators=41,
        #     learning_rate=0.1,
        #     subsample=0.8,
        #     loss='deviance',
        #     # 分割参数
        #     max_features='sqrt',
        #     # criterion='friedman_mse',
        #     # 分割停止参数
        #     # min_samples_split=1200,
        #     # min_impurity_split=None,
        #     # min_impurity_decrease=0.0,
        #     max_depth=7,
        #     # max_leaf_nodes=None,
        #     # 剪枝参数
        #     # min_samples_leaf=60,
        #     # warm_start=False,
        #     # random_state=10
        # )
        # 马铃薯2021/7/13
        mod = GradientBoostingClassifier(learning_rate=0.1, n_estimators=95, max_depth=3,
                                         max_features='auto', subsample=0.8,
                                         )
        # accuracy ： 0.8930581613508443 - 0.9555416405760802
        # recall   ： 0.9743589743589743 - 0.9945098039215686
        # f1       ： 0.9361702127659574 - 0.9727656309934792
        # precision： 0.9008620689655172 - 0.9519519519519519
        # auc: 0.9106150259996414 - 0.9907733528193886

        mod.fit(X_train, Y_train)
        print(mod)
        Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "GBDT")
        return mod

        #马铃薯
        # mod = GradientBoostingClassifier(
        #     # boosting参数
        #     # init=None,
        #     n_estimators=81,
        #     learning_rate=1,
        #     subsample=1,
        #     loss='deviance',
        #     # 分割参数
        #     max_features='auto',
        #     # criterion='friedman_mse',
        #     # 分割停止参数
        #     # min_samples_split=1200,
        #     # min_impurity_split=None,
        #     # min_impurity_decrease=0.0,
        #     max_depth=9,
        #     # max_leaf_nodes=None,
        #     # 剪枝参数
        #     # min_samples_leaf=60,
        #     # warm_start=False,
        #     # random_state=10
        # )
        # mod.fit(X_train, Y_train)
        # print(mod)
        # Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "GBDT")
        # return mod
        # 水稻
        from sklearn.model_selection import GridSearchCV
        param_test1 = {
            # 'n_estimators': range(10, 50, 5),
            # 'learning_rate': [0.01,0.02,0.04,0.05,0.06,0.08,0.1,],
            # 'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            # 'max_depth': range(3, 16, 1),
            # 'max_features':['auto','log2','sqrt'],
        }
        #调参2021、6、13 水稻
        # mod = GridSearchCV(estimator= GradientBoostingClassifier(learning_rate=0.1,n_estimators=25, max_depth=5,
        #                                  max_features='auto', subsample=0.6,), param_grid=param_test1, scoring='accuracy', refit=True, cv=3)
        # # mod = GridSearchCV(estimator= GradientBoostingClassifier(), param_grid=param_test, scoring='accuracy', refit=True, cv=3)
        # mod.fit(X_train, Y_train)
        # print(mod.best_params_)
        # print(mod.best_score_)
        # return

        #2021/7/13调参代码

        from sklearn.model_selection import GridSearchCV
        param_test1 = {
            # 'n_estimators': range(30, 50, 1),
            # 'learning_rate': [0.001,0.01,0.1],
            'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            # 'max_depth': range(3,5, 1),
            # 'max_features':['auto','log2','sqrt'],
        }
        # 调参2021、7、12 番茄
        mod = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=35, max_depth=4,
                                                                max_features='auto', subsample=0.7 ),
                           param_grid=param_test1, scoring='accuracy', refit=True, cv=3)
        # mod = GridSearchCV(estimator= GradientBoostingClassifier(), param_grid=param_test, scoring='accuracy', refit=True, cv=3)
        mod.fit(X_train, Y_train)
        print(mod.best_params_)
        print(mod.best_score_)
        return



    def AdaboostClassdemo(self, X_train, X_test, Y_train, Y_test):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier
        # mod=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=3,max_features='auto'),n_estimators=60,learning_rate=0.03)
        # [[0.8045112781954887, 0.8120300751879699, 0.8195488721804511, 0.8120300751879699, 0.8157894736842105],
        #  [0.9953051643192489, 0.9812206572769953, 0.9812206572769953, 0.9906103286384976, 0.9858490566037735],
        #  [0.8185328185328186, 0.8108108108108109, 0.8087649402390438, 0.8156862745098039, 0.8178294573643411],
        #  [0.8955223880597014, 0.8949579831932774, 0.8851063829787233, 0.8888888888888888, 0.8983050847457628]]
        # 十折交叉平均值(accuracy) ： 0.812781954887218
        # 十折交叉平均值(recall)   ： 0.986841172823102
        # 十折交叉平均值(f1)       ： 0.8925561455732707
        # 十折交叉平均值(precision)： 0.8143248602913635
        # accuracy ： 0.8265765765765766 - 0.8488721804511278
        # recall   ： 0.9915492957746479 - 0.9962406015037594
        # f1       ： 0.9014084507042254 - 0.9133993968117191
        # precision： 0.8262910798122066 - 0.843277645186953
        # auc: 0.8529197657857255 - 0.9428069704336028
        #2021/7/17 OS

        # mod=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=3,max_features='auto'),n_estimators=60,learning_rate=0.03)
        # accuracy ： 0.8333333333333334 - 0.868421052631579
        # recall   ： 0.9802816901408451 - 0.993421052631579
        # f1       ： 0.9038961038961039 - 0.9235474006116209
        # precision： 0.8385542168674699 - 0.8628571428571429
        # auc: 0.8519544231682229 - 0.9546434224659393

        #2021、7、15 sly
        # mod=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=4,max_features='auto'),n_estimators=60,learning_rate=0.04)
        # accuracy ： 0.9258517034068137 - 0.9538770053475936
        # recall   ： 0.9925 - 1.0
        # f1       ： 0.9554753309265944 - 0.9719626168224299
        # precision： 0.9211136890951276 - 0.9454545454545454
        # auc: 0.9368181818181818 - 0.9987931995540691

        # 2021、7、15 potato
        mod=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=3,max_features='auto'),n_estimators=60,learning_rate=0.08)
        # accuracy ： 0.8874296435272045 - 0.965560425798372
        # recall   ： 0.9533799533799534 - 0.9764705882352941
        # f1       ： 0.9316628701594533 - 0.9783889980353634
        # precision： 0.910913140311804 - 0.9803149606299213
        # auc: 0.8880446476600323 - 0.9939300937766411


        mod.fit(X_train, Y_train)
        print(mod)
        Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "AdaBoost")
        return mod








        #水稻
        # mod=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=3,max_features='auto'),n_estimators=31,learning_rate=0.1)
        #番茄
        # mod = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='auto'),
        #                          n_estimators=91, learning_rate=0.1)

        #马铃薯
        # mod = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=7, max_features='auto'),
        #                          n_estimators=101, learning_rate=1)
        # mod.fit(X_train, Y_train)
        # print(mod)
        # Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "AdaBoost")
        # return mod

        # param_test1 = {"max_depth": range(2, 15, 1),"max_feature":['auto','log2','sqrt']}

        from sklearn.model_selection import GridSearchCV
        param_test1 = {
            # 'max_depth': range(3, 16, 2),
            # 'n_estimators': range(10, 100, 10),
            'n_estimators': [60],
            'learning_rate':[0.02],

            # 'learning_rate': [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            # 'learning_rate': [0.001,0.01,0.1,1],
            # 'max_feature':['auto','log2','sqrt']

        }
        param_test2 = {
            # 'max_depth': range(3, 16, 2),
            'n_estimators': range(10, 100, 10),
            # 'learning_rate': [0.001, 0.01, 0.1, 1],
            'learning_rate': [0.001,0.002, 0.004,0.006,0.008,0.01,0.02, 0.04,0.06,0.08,0.1],
            # 'max_feature': ['auto', 'log2', 'sqrt']

        }
        mod = GridSearchCV(estimator= AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=5,max_features='auto')), param_grid=param_test1, scoring='accuracy', refit=True, cv=3)
        mod.fit(X_train, Y_train)
        print(mod.best_params_)
        print(mod.best_score_)
        return


    def RFClassdemo(self, X_train, X_test, Y_train, Y_test):
        from sklearn.ensemble import RandomForestClassifier
        #2021、6、14
        # mod = RandomForestClassifier(criterion='entropy', n_estimators=14, oob_score= 'True',max_depth=7,max_features="auto")#水稻
        # 十折交叉平均值(accuracy) ： 0.8045112781954888
        # 十折交叉平均值(recall)   ： 0.9877889981397819
        # 十折交叉平均值(f1)       ： 0.896988908980604
        # 十折交叉平均值(precision)： 0.8205004347844034
        # accuracy ： 0.8175675675675675 - 0.8729323308270677
        # recall   ： 0.9887323943661972 - 0.9990601503759399
        # f1       ： 0.896551724137931 - 0.9263616557734203
        # precision： 0.8200934579439252 - 0.8635255889520714
        # auc: 0.8200348156353854 - 0.9709459268471932

        # mod = RandomForestClassifier(criterion='entropy', n_estimators=14, oob_score='True', max_depth=5,
        #                              max_features="auto")  # 水稻
        #
        # 十折交叉平均值(accuracy) ： 0.8097744360902256
        # 十折交叉平均值(recall)   ： 0.9934183718664187
        # 十折交叉平均值(f1)       ： 0.888700390807508
        # 十折交叉平均值(precision)： 0.806569980806622
        # accuracy ： 0.8018018018018018 - 0.8233082706766918
        # recall   ： 0.9915492957746479 - 0.9981203007518797
        # f1       ： 0.8888888888888887 - 0.9003815175922001
        # precision： 0.8054919908466819 - 0.8200772200772201
        # auc: 0.8361607849343251 - 0.9132971055458194
        # mod = RandomForestClassifier(criterion='entropy', n_estimators=20, oob_score='True', max_depth=5,
        #                              max_features="auto")  # 水稻
        # 十折交叉平均值(accuracy) ： 0.8097744360902256
        # 十折交叉平均值(recall)   ： 0.9962441314553991
        # 十折交叉平均值(f1)       ： 0.8917845693229449
        # 十折交叉平均值(precision)： 0.8088911654545109
        # accuracy ： 0.8018018018018018 - 0.8172932330827067
        # recall   ： 0.9915492957746479 - 0.9962406015037594
        # f1       ： 0.8888888888888887 - 0.8971646212441812
        # precision： 0.8054919908466819 - 0.8160123171670516
        # auc: 0.8463839215065675 - 0.9184910113629938
        # mod = RandomForestClassifier(criterion='entropy', n_estimators=20, oob_score='True', max_depth=5,
        #                              max_features="auto")  # 水稻
#2021/7/17 OS
        # mod = RandomForestClassifier(criterion='gini', n_estimators=50, oob_score='False', max_depth=6,
        #                              max_features="sqrt")  # 水稻
        # accuracy ： 0.8220720720720721 - 0.8714285714285714
        # recall   ： 0.9887323943661972 - 0.9990601503759399
        # f1       ： 0.8988476312419975 - 0.9255550718328254
        # precision： 0.823943661971831 - 0.862124898621249
        # auc: 0.8548188004431081 - 0.9638935213974786






        # mod = RandomForestClassifier(n_estimators=11,max_depth=7,max_features="sqrt")#水稻
        # mod = RandomForestClassifier(n_estimators=50, max_depth=30, max_features="auto")
        # mod = RandomForestClassifier(n_estimators=29, max_depth=8, max_features="auto" )#马铃薯
        # # mod = RandomForestClassifier(n_estimators=101, max_depth=11, max_features=9)#水稻1
        # # mod = RandomForestClassifier(n_estimators=81, max_depth=7, max_features=9)#水稻12
        # mod = RandomForestClassifier(n_estimators=30, max_depth=8, max_features="auto") #番茄


        #番茄 2021/7/13
        # mod = RandomForestClassifier(criterion='entropy', n_estimators=50, oob_score='False', max_depth=7,
        #                              max_features="sqrt")
        # accuracy ： 0.9258517034068137 - 0.93048128342246
        # recall   ： 0.995 - 0.9991638795986622
        # f1       ： 0.9555822328931572 - 0.958299919807538
        # precision： 0.9191685912240185 - 0.9206471494607088
        # auc: 0.9357828282828282 - 0.9921711259754739


        # mod = RandomForestClassifier(criterion='entropy', n_estimators=50, oob_score='False', max_depth=7,
        #                              max_features="auto")#番茄最终
        # accuracy ： 0.9218436873747495 - 0.9258021390374331
        # recall   ： 0.9975 - 0.9991638795986622
        # f1       ： 0.953405017921147 - 0.9556177528988404
        # precision： 0.9130434782608695 - 0.9157088122605364
        # auc: 0.9348232323232324 - 0.9940746934225195

        mod = RandomForestClassifier(criterion='entropy', n_estimators=68, oob_score='False', max_depth=9,
                                     max_features="auto")#马铃薯
        # accuracy ： 0.8649155722326454 - 0.949906073888541
        # recall   ： 0.9836829836829837 - 0.9992156862745099
        # f1       ： 0.9213973799126638 - 0.969558599695586
        # precision： 0.86652977412731 - 0.9416112342941612
        # auc: 0.8964048771741079 - 0.998745585190598


        mod.fit(X_train, Y_train)
        print("RFC")
        # # # # # print("RFC_X_train")
        # # # # # print(X_train)
        # # # # # print("RFC_Y_train")
        # # # # # print(Y_train)
        print("RFC_mod")
        print(mod)
        print()
        Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "Random Forest")
        return mod

        from sklearn.model_selection import GridSearchCV
        param_test = {
            # 'n_estimators': range(10, 100, 10),
            # 'oob_score' : ["True","False"],
            # 'criterion' : ["entropy", "gini"],
            # 'n_estimators':[30],
            # 'max_depth': [8],
            'max_depth': range(3, 10, 1),
            # 'max_features': ["auto","log2","sqrt"],
            # 'max_features': range(3, 11, 2),
        }
        mod = GridSearchCV(estimator=RandomForestClassifier(n_estimators=50,criterion='gini', max_features='auto', oob_score= 'False',max_depth=5), param_grid=param_test, scoring='accuracy', refit=True, cv=3)
        mod.fit(X_train, Y_train)
        print(mod.best_params_)
        print(mod.best_score_)
        return


        # import numpy as np
        # import pylab as pl  # 画图用
        # from sklearn import svm
        # import matplotlib.pyplot as plt
        # from sklearn import svm
        # clf = svm.SVC(kernel='rbf',c=8,gamma=0.001)
        # clf.fit(X_train, Y_train)
        # title='SVC with RBF kernel'
        # x_min, x_max = x.min() - 1, x.max() + 1
        # y_min, y_max = y.min() - 1, y.max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
        #                      np.arange(y_min, y_max, 0.2))
        # fig, sub = plt.subplots(2, 2)
        # plt.subplots_adjust(wspace=0.4, hspace=0.4)
        #
        # X0, X1 = X[:, 0], X[:, 1]





    def SVMDemo(self, X_train, X_test, Y_train, Y_test):
        from sklearn import svm
        # grid = svm.SVC(probability=True, C=2, gamma=0.001)# 稻瘟菌
        grid = svm.SVC(probability=True, C=2, gamma=0.005) #马铃薯
        # grid = svm.SVC(probability=True, C=8, gamma=0.001)#番茄
        grid.fit(X_train, Y_train)
        print("SVM")
        # # # # #         # print("SVM_X_train")
        # # # # #         # print(X_train)
        # # # # #         # print("SVM_Y_train")
        # # # # #         # print(Y_train)
        print("SVM_grid")
        print(grid)
        Model.FigResult(grid, X_train, Y_train, X_test, Y_test, "SVM")
        return grid

        param_test = {
            # 'C': [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'C': [0.1, 1, 2],
            'gamma': [0.001, 0.005, 0.01, 0.02, 0.03, 0.04],
        }
        scores = ['precision', 'recall']
        grid = GridSearchCV(estimator=svm.SVC(probability=True), param_grid=param_test, scoring='accuracy', refit=True, cv=3)
        # grid = svm.SVC(probability=True, C=0.6, gamma=0.01)
        grid.fit(X_train, Y_train)
        print(grid)
        print(grid.best_params_)
        print(grid.best_score_)
        return

    # 特征重要性排序——随机森林
    def RFdemo(self):
        X_train, X_test, Y_train, Y_test = Model.DataInit(evalute=1)

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=81, max_depth=11, max_features=5)
        rf.fit(X_train, Y_train)
        print(sorted(zip(map(lambda x:round(x, 4), rf.feature_importances_), cols_data), reverse=True))
        from matplotlib import pyplot
        pyplot.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
        pyplot.show()

        pyplot.figure(figsize=(12, 6))
        top30 = sorted(zip(map(lambda x:round(x, 4), rf.feature_importances_), cols_data_rmdup), reverse=True)
        top30_title = []
        top30_conte = []
        top30 = top30[0:60]
        for i in top30:
            top30_title.append(i[1])
            top30_conte.append(i[0])
        pyplot.bar(top30_title, top30_conte, tick_label=top30_title)
        plt.setp(plt.gca().get_xticklabels(), rotation=75, horizontalalignment='right')
        pyplot.show()

    #
    # 特征重要性排序——XGBoost
    def XGBoostdemo(self, X_train, X_test, Y_train, Y_test):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        from xgboost import XGBClassifier
        from matplotlib import pyplot

        # model = XGBClassifier(gpu_id=0, n_estimators=70, max_depth=5, subsample=1)
        model = XGBClassifier(gpu_id=0, n_estimators=90, max_depth=2, subsample=0.7, gamma=0.001)
        model.fit(X_train, Y_train, eval_metric='auc')
        # Model.FigResult(model, X_train, Y_train, X_test, Y_test)

        top_sort = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), cols_data_rmdup), reverse=True)

        # 画图
        top30 = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), cols_data_rmdup), reverse=True)[0:30]
        top30_title = []
        top30_conte = []
        map_feature_details = {}
        for i in range(len(feature)):
            map_feature_details[feature[i]] = details[i]
        count = 0
        for i in top30:
            count += 1
            top30_title.append(map_feature_details[i[1]])
            top30_conte.append(i[0])
            print(count, ":", i[1])
        plt.figure(figsize=(8, 6))
        pyplot.bar(top30_title, top30_conte, tick_label=top30_title)
        plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right')
        plt.ylabel("Feature importance ranking")
        pyplot.show()
        return


        X_train_sort = []
        X_test_sort = []
        for k,v in top_sort:
            X_train_sort.append(X_train[:, cols_data.index(v)])
            X_test_sort.append(X_test[:, cols_data.index(v)])

        X_train_sort = np.array(X_train_sort)
        X_train_sort = X_train_sort.transpose()
        X_test_sort = np.array(X_test_sort)
        X_test_sort = X_test_sort.transpose()

        # X_train_i = np.delete(X_train_sort, range(202, 203), axis=1)
        # X_test_i = np.delete(X_test_sort, range(202, 203), axis=1)
        #
        # print(X_train_sort.shape)
        # print(X_test_sort.shape)
        # print(X_train_i.shape)
        # print(X_test_i.shape)
        # Model.XGBoost(X_train_sort, Y_train, X_test_sort, Y_test)
        # return

        parm = []
        t = 0
        max_count = 0
        max_index = 0
        length = 180
        for i in range(1, length):
            X_train_i = np.delete(X_train_sort, range(i, length), axis=1)
            X_test_i = np.delete(X_test_sort, range(i, length), axis=1)
            t = Model.XGBoost(X_train_i, Y_train, X_test_i, Y_test)
            print(t)
            parm.append(t)
            if t > max_count:
                max_count = t
                max_index = i
        print(max_index)
        plt.plot(range(1, len(parm)+1), parm)
        plt.xlabel("Number of features")
        plt.ylabel('accuracy')
        plt.show()
    #
    def LGBparm(self, X_train, X_test, Y_train, Y_test):
        import lightgbm as lgb
        # mod = lgb.LGBMClassifier(gpu_id=0,max_depth=-1,num_leaves=31, bagging_fraction=1.0,subsample=1, min_gain_to_split=20)#水稻第一次
        # mod = lgb.LGBMClassifier(gpu_id=0, max_depth=3, num_leaves=5, bagging_fraction=0.6, learning_rate=0.1,n_estimators=45)  # 水稻最终
        mod = lgb.LGBMClassifier(gpu_id=0, max_depth=5, num_leaves=18, bagging_fraction=0.6, learning_rate=0.1,n_estimators=58)  # 马铃薯最终
        # mod = lgb.LGBMClassifier(gpu_id=0, max_depth=5, num_leaves=20, bagging_fraction=0.6, learning_rate=0.1,n_estimators=74)  # 番茄 一般
        # # accuracy ： 0.9298597194388778 - 0.9779411764705882
        # # recall   ： 0.99 - 0.9991638795986622
        # # f1       ： 0.9576783555018138 - 0.9863805200165084
        # # precision： 0.927400468384075 - 0.973920130399348
        # # auc: 0.9477525252525253 - 0.999108138238573

        # mod = lgb.LGBMClassifier(gpu_id=0, max_depth=5, num_leaves=9, bagging_fraction=0.6, learning_rate=0.1,n_estimators=55)  # 番茄最终
        # accuracy ： 0.9258517034068137 - 0.9498663101604278
        # recall   ： 0.99 - 0.9941471571906354
        # f1       ： 0.9553679131483716 - 0.9694251936404403
        # precision： 0.9230769230769231 - 0.9459029435163087
        # auc: 0.9412878787878788 - 0.9922603121516166

        mod.fit(X_train, Y_train)
        print("LGB")
        # print("LGB_X_train")
        # print(X_train)
        # print("LGB_Y_train")
        # print(Y_train)
        print("LGBoost_mod")
        print(mod)
        Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "LGBparm")
        return mod



        # param_test = {
        #     "boosting_type": "gbdt",
        #     "objective" : "binary",
        #
        #     'learning_rate': 0.1,
        #     'max_depth':5,
        #     "metric": "auc",
        #     "num_leaves": 30,
        #     # "min_data_in_leaf" : 1,
        #     # "feature_fraction" : 0.8,
        #     "bagging_fraction" : 0.8,
        #     # "bagging_fraq" : 0.9
        # }
        # trn_data= lgb.Dataset(X_train,Y_train)
        # val_data= lgb.Dataset(X_test,Y_test)
        # cv_results = lgb.cv(
        # param_test,
        # trn_data,
        # #val_Sets=[trn_data,val_data],
        # nfold = 3,
        # # metrics=["accuracy"],
        # verbose_eval = True)
        # print('best n_estimators:', len(cv_results['auc-mean']))
        # print('best cv score:', pd.Series(cv_results['auc-mean']).max())

        # from sklearn.model_selection import GridSearchCV
        # ### 我们可以创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目)
        # model_lgb = lgb.LGBMClassifier(gpu_id=0,objective='binary', learning_rate=0.1, n_estimators=55,max_depth=5,num_leaves=21, bagging_fraction=0.8)#水稻第一次
        #
        # params_test1 = {
        #     'max_depth': range(3, 6, 1),
        #     'num_leaves': range(5, 20,1)
        # }
        # gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='accuracy', cv=3,
        #                         verbose=1, n_jobs=4)
        # gsearch1.fit(X_train, Y_train)
        # print(gsearch1.best_params_)
        # print(gsearch1.best_score_)



        model_lgb = lgb.LGBMClassifier(gpu_id=0, objective='binary', learning_rate=0.1, n_estimators=55, max_depth=5,
                                       num_leaves=9, bagging_fraction=0.8)  # 水稻第一次
        params_test4 = {
            'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='accuracy', cv=3,
                                verbose=1, n_jobs=4)
        gsearch1.fit(X_train, Y_train)
        print(gsearch1.best_params_)
        print(gsearch1.best_score_)

        # model_lgb = lgb.LGBMClassifier(gpu_id=0, objective='binary', learning_rate=0.1, n_estimators=50, max_depth=6,
        #                                num_leaves=30, bagging_fraction=0.6)  # 水稻第一次
        # params_test4 = {
        #     'n_estimators' : range(40,70,5),
        #     'learning_rate': [0.001, 0.01, 0.1],
        #
        # }
        # gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='accuracy', cv=3,
        #                         verbose=1, n_jobs=4)
        # gsearch1.fit(X_train, Y_train)
        # print(gsearch1.best_params_)
        # print(gsearch1.best_score_)


        # mod = GridSearchCV(param_grid=param_test, scoring='accuracy', refit=True, cv=3)
        # mod.fit(X_train, Y_train)
        # print(mod.best_params_)
        # print(mod.best_score_)
        return

    def XGBoostparm(self, X_train, X_test, Y_train, Y_test):

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        #mod = XGBClassifier(gpu_id=0, n_estimators=100, max_depth=3, subsample=0.9, gamma=0)
        from xgboost import XGBClassifier
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, objective='binary:logistic', gamma=3, max_depth=6,
        #                     min_child_weight=5, subsample=0.5)
        # 十折交叉平均值(accuracy) ： 0.830827067669173
        # 十折交叉平均值(recall)   ： 0.935175834883515
        # 十折交叉平均值(f1)       ： 0.8982702149073305
        # 十折交叉平均值(precision)： 0.8644969741778252
        # accuracy ： 0.8445945945945946 - 0.8984962406015038
        # recall   ： 0.9183098591549296 - 0.9680451127819549
        # f1       ： 0.9042995839112343 - 0.9384965831435079
        # precision： 0.8907103825136612 - 0.9106984969053935
        # auc: 0.8547396739990505 - 0.9468454972016507
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, objective='binary:logistic', gamma=3, max_depth=9,
        #                     min_child_weight=5, subsample=0.5)
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, objective='binary:logistic', gamma=3, max_depth=3,
        #                     min_child_weight=5, subsample=0.5)#水稻
        # 十折交叉平均值(accuracy) ： 0.8225563909774436
        # 十折交叉平均值(recall)   ： 0.9248339091150676
        # 十折交叉平均值(f1)       ： 0.892794123467081
        # 十折交叉平均值(precision)： 0.8632152260554206
        # accuracy ： 0.8445945945945946 - 0.8721804511278195
        # recall   ： 0.9267605633802817 - 0.9577067669172933
        # f1       ： 0.905089408528198 - 0.9230072463768116
        # precision： 0.8844086021505376 - 0.8907342657342657
        # auc: 0.8524608324101914 - 0.919047501271977





        # mod =  XGBClassifier(gpu_id=0,n_estimators=20, objective='binary:logistic',gamma= 3,max_depth=9, min_child_weight =5, subsample=0.5 )
        # accuracy ： 0.8536036036036037 - 0.9007518796992481
        # recall   ： 0.9323943661971831 - 0.9661654135338346
        # f1       ： 0.9105914718019258 - 0.9396709323583181
        # precision： 0.8897849462365591 - 0.9145907473309609
        # auc: 0.839895553093844 - 0.9454993216123014
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, objective='binary:logistic', gamma=3, max_depth=3,
        #                     min_child_weight=5, subsample=0.5)
        #
        # accuracy ： 0.8513513513513513 - 0.8766917293233083
        # recall   ： 0.923943661971831 - 0.9586466165413534
        # f1       ： 0.9085872576177286 - 0.9255898366606171
        # precision： 0.8937329700272479 - 0.8947368421052632
        # auc: 0.8466054755499288 - 0.921891783029001
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, objective='binary:logistic', gamma=3, max_depth=6,
        #                     min_child_weight=5, subsample=0.5)
        # accuracy ： 0.8423423423423423 - 0.8947368421052632
        # recall   ： 0.9098591549295775 - 0.9614661654135338
        # f1       ： 0.9022346368715084 - 0.9359560841720036
        # precision： 0.8947368421052632 - 0.9117647058823529
        # auc: 0.85244500712138 - 0.9449975973769009
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, objective='binary:logistic', gamma=4, max_depth=3,
        #                     min_child_weight=5, subsample=0.5)
        # accuracy ： 0.8378378378378378 - 0.8706766917293233
        # recall   ： 0.9098591549295775 - 0.9586466165413534
        # f1       ： 0.8997214484679666 - 0.9222423146473779
        # precision： 0.8898071625344353 - 0.8885017421602788
        # auc: 0.8492641240702643 - 0.9156361297981797
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, objective='binary:logistic', gamma=3, max_depth=3,
        #                     min_child_weight=5, subsample=0.6)
        # accuracy ： 0.8558558558558559 - 0.8714285714285714
        # recall   ： 0.9295774647887324 - 0.9548872180451128
        # f1       ： 0.9116022099447514 - 0.9223785746709033
        # precision： 0.8943089430894309 - 0.8920105355575065
        # auc: 0.8554201614179459 - 0.9256193821018711
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, gamma=4, max_depth=9, min_child_weight=5, subsample=0.6)
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, max_depth=3, subsample=0.7,gamma= 5 )#水稻6.6
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, max_depth=8, subsample=0.8,gamma=2)#马铃薯
        # mod = XGBClassifier(gpu_id=0, n_estimators=15, max_depth=3, subsample=0.8,gamma= 1 )#水稻
        # mod = XGBClassifier(gpu_id=0, n_estimators=20, max_depth=8, subsample=0.7,gamma=4)  # 番茄

        # mod = XGBClassifier(gpu_id=0, n_estimators=60, objective='binary:logistic', gamma=4, max_depth=6,
        #                     min_child_weight=4, subsample=0.8)#sly

        # accuracy ： 0.9278557114228457 - 0.9585561497326203
        # recall   ： 0.98 - 0.9933110367892977
        # f1       ： 0.9560975609756097 - 0.9745693191140279
        # precision： 0.9333333333333333 - 0.9565217391304348
        # auc: 0.9312626262626263 - 0.9934253065774805

        2021/7/15

        # mod = XGBClassifier(gpu_id=0, n_estimators=30, objective='binary:logistic', gamma=2, max_depth=3,
        #                     min_child_weight=1, subsample=0.6)#Potato

        # accuracy ： 0.8686679174484052 - 0.9386349405134627
        # recall   ： 0.9627039627039627 - 0.9819607843137255
        # f1       ： 0.9218750000000001 - 0.962336664104535
        # precision： 0.8843683083511777 - 0.9434815373021854
        # auc: 0.881522323830016 - 0.9786006576543661

        mod = XGBClassifier(gpu_id=0, n_estimators=50, objective='binary:logistic', gamma=2, max_depth=3,
                            min_child_weight=1, subsample=0.6)#Potato

        # accuracy ： 0.8855534709193246 - 0.965560425798372
        # recall   ： 0.9533799533799534 - 0.9882352941176471
        # f1       ： 0.9306029579067122 - 0.9786407766990292
        # precision： 0.9088888888888889 - 0.9692307692307692
        # auc: 0.9010892953200647 - 0.992066739739374

        # mod = XGBClassifier(gpu_id=0, n_estimators=30, objective='binary:logistic', gamma=2, max_depth=3,
        #                     min_child_weight=1, subsample=0.8)  # Potato

        # accuracy ： 0.874296435272045 - 0.939261114589856
        # recall   ： 0.9696969696969697 - 0.984313725490196
        # f1       ： 0.9254727474972191 - 0.9627924817798236
        # precision： 0.8851063829787233 - 0.9421921921921922
        # auc: 0.8892101488255334 - 0.9805017659237607

        mod.fit(X_train, Y_train)
        print("XGBoost")
        # print("XGBoost_X_train")
        # print(X_train)
        # print("XGBoost_Y_train")
        # print(Y_train)
        print("XGBoost_mod")
        print(mod)
        Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "XGBoost")
        # # Model.lossResult(mod, X_train, Y_train, X_test, Y_test, "XGBoost")
        return mod
        # {'gamma': 4, 'n_estimators': 20}
        # 0.825558063707048
        # {'max_depth': 9, 'min_child_weight': 5}
        # 0.8345772408977826
        # {'subsample': 0.6}
        # 0.8232990326669786
        param_test = {
            # 'n_estimators': range(10, 100, 10),
            # 'eta': range(0.01,0.2,0.001)
            'n_estimators': [50],
            'gamma': [2],
            # 'n_estimators': range(10, 30, 10),
            # 'subsample': [0.6],
            "gamma":range(2,5,1),
            # "gamma":[4],
            "max_depth":[3],
            # 'max_depth': range(3, 10, 1),
            'min_child_weight': range(1,6,1),
            # 'min_child_weight': [4],
            'subsample': [0.3,0.4,0.5, 0.6, 0.7, 0.8]

        }

        mod = GridSearchCV(estimator=XGBClassifier(gpu_id =0), param_grid=param_test, scoring='accuracy', refit=True, cv=3)
        mod.fit(X_train, Y_train)
        print(mod.best_params_)
        print(mod.best_score_)
        return

        # m = XGBClassifier(gpu_id=0)
        # m.fit(X_train, Y_train, eval_metric='auc')
        # acc_scores = cross_val_score(m, X_train, Y_train, cv=5, scoring='accuracy')
        # Model.FigResult(m, X_train, Y_train, X_test, Y_test)

        # return acc_scores.mean()

    @staticmethod
    def XGBoost(X_train, Y_train, X_test, Y_test):
        from xgboost import XGBClassifier
        mod = XGBClassifier(gpu_id=0, n_estimators=90, max_depth=2, subsample=0.7)
        # mod = XGBClassifier(gpu_id=0, n_estimators=90, max_depth=8, subsample=0.9)
        mod.fit(X_train, Y_train)
        # acc_scores = cross_val_score(mod, X_train, Y_train, cv=3, scoring='accuracy')
        Model.FigResult(mod, X_train, Y_train, X_test, Y_test, "XGBoost")
        # return acc_scores.mean()


    # @staticmethod
    #
    # def MLPdemo():
    #     import tensorflow.compat.v1 as tf
    #     tf.disable_v2_behavior()
    #     # 数据格式处理
    #     sample = "SamTr.csv"
    #     potus = list(csv.reader(open(sample)))
    #     dx = []
    #     dy = []
    #     potus = potus[1:]
    #     # shuffle(potus)
    #     for i in range(0, len(potus)):
    #         dx.append([int(x) for x in potus[i][0:len(potus[i]) - 1]])
    #         dy.append([int(potus[i][len(potus[i]) - 1])])
    #     train_dx = dx[0:864]
    #     test_dx = dx[864:]
    #     train_dy = dy[0:864]
    #     test_dy = dy[864:]
    #
    #     # 定义输入和输出
    #     x = tf.placeholder(tf.float32, shape=(None, 203), name="x-input")
    #     y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
    #     # 定义神经网络的参数
    #     w1 = tf.Variable(tf.random_normal([203, 10], mean=0, stddev=1, seed=1))
    #     w2 = tf.Variable(tf.random_normal([10, 1], mean=0, stddev=1, seed=1))
    #     b1 = tf.Variable(tf.random_normal([10], mean=0, stddev=1, seed=1))
    #     b2 = tf.Variable(tf.random_normal([1], mean=0, stddev=1, seed=1))
    #
    #     y1 = tf.matmul(x, w1) + b1
    #     y11 = tf.nn.relu(y1)
    #     y2 = tf.matmul(y11, w2) + b2
    #     y = tf.sigmoid(y2)
    #     # tf.clip_by_value(t, clip_value_min, clip_value_max,name=None)
    #     # cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    #     # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    #     loss = -tf.reduce_mean(
    #         y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
    #     train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    #     X = train_dx
    #     Y = train_dy
    #     # 创建会话运行TensorFlow程序
    #     with tf.Session() as sess:
    #         init = tf.initialize_all_variables()
    #         saver = tf.train.Saver()
    #         sess.run(init)
    #         steps = 1500
    #         for i in range(steps):
    #             # 通过选取样本训练神经网络并更新参数
    #             sess.run(train_step, feed_dict={x: X, y_: Y})
    #             # 每迭代1000次输出一次日志信息
    #             if i % 100 == 0:
    #                 # 计算所有数据的交叉熵
    #                 total_cross_entropy, prob = sess.run([loss, y], feed_dict={x: test_dx, y_: test_dy})
    #                 # 输出交叉熵之和
    #                 print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
    #         prob_train = sess.run(y, feed_dict={x: train_dx, y_: train_dy})
    #         from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
    #         roc_test = roc_auc_score(test_dy, prob)
    #         roc_train = roc_auc_score(train_dy, prob_train)
    #         prob_sig = []
    #         for i in prob:
    #             prob_sig.append(1 if float(i) > 0.5 else 0)
    #         print(accuracy_score(test_dy, prob_sig))
    #
    #         # save_path = saver.save(sess, '../ML/model.ckpt')
    #         # print("Model saved in file: %s" % save_path)
    #         result = []
    #         result.append([roc_test, str(w1.eval(session=sess)), str(w2.eval(session=sess)), str(b1.eval(session=sess)),
    #                        str(b2.eval(session=sess))])
    #
    #         import matplotlib.pyplot as plt
    #         from sklearn.metrics import roc_curve
    #         import pandas as pd
    #         fpr, tpr, thresholds = roc_curve(test_dy, prob, pos_label=1.0)
    #
    #         print("auc  :", roc_test,"-", roc_train)
    #         y_scores = prob_sig
    #         plt.figure(figsize=(6.4, 6.4))
    #         plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % roc_test)
    #         plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title('Receiver operating characteristic of MLP')
    #         plt.legend(loc="lower right")
    #         plt.show()
    #
    #         return fpr, tpr, roc_test


if __name__ == '__main__':
    modle = Model()
    # 特征提取
    # modle.preProcess('../Script/Mapping/NavSample_seq.fasta')

    # 数据分类
    X_train, X_test, Y_train, Y_test = Model.DataInit(evalute=0)

    # # 模型选择print("RFC_mod")
    #         print(mod)
    # m1 = modle.KnnDemo(X_train, X_test, Y_train, Y_test)
    # m2 = modle.Bysdemo(X_train, X_test, Y_train, Y_test)
    # m3 = modle.Dtree(X_train, X_test, Y_train, Y_test)
    # m1= modle.GBDTClassdemo(X_train, X_test, Y_train, Y_test)
    # m4 = modle.LogisticRegression(X_train, X_test, Y_train, Y_test)

    # m2 = modle.RFClassdemo(X_train, X_test, Y_train, Y_test)
    # m3=modle.AdaboostClassdemo(X_train, X_test, Y_train, Y_test)
    # m6 = modle.SVMDemo(X_train, X_test, Y_train, Y_test)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # # print(m6)
    # m4 = modle.XGBoostparm(X_train, X_test, Y_train, Y_test)
    #
    # m5 = modle.LGBparm(X_train, X_test, Y_train, Y_test)
    # # f, t, s = Model.MLPdemo()
    # Model.FigAllImg(m1=m1,m2=m2 , m3=m3,m4 = m4,m5=m5,X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
    # 特征选择
    # modle.RFdemo()
    modle.XGBoostdemo(X_train, X_test, Y_train, Y_test)
    # modle.XGBoost(X_train, Y_train, X_test, Y_test)

