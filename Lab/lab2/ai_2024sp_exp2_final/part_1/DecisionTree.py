from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd 
import os

def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

# 叶结点
class Node:
    def __init__(self, feature_index=None, threshold=-1, value=-1):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.child = []

class DecisionTreeClassifier:
    # 决策树默认深度100
    def __init__(self) -> None:
        self.tree = None
        
    def fit(self, X, y): 
        self.tree = self.build_tree(X, y)
        
    def build_tree(self, X, y):
        now_class = np.unique(y)
        counts = np.bincount(y)
        max_index = np.argmax(counts)
        # 如果当前y只有一种说明分类成功
        if len(now_class) == 1 :
            return Node(value=now_class[0])
        else :
            current_node = self.find_best_spilt(X, y)
            choose_indices = X[current_node.feature_index] < current_node.threshold
            choose_X = X[choose_indices]
            remain_X = X[~choose_indices]
            choose_y = y[choose_indices]
            remain_y = y[~choose_indices]
            if len(choose_X) <= 0 :
                current_node.child.append(Node(value=max_index))
            else:
                current_node.child.append(self.build_tree(choose_X,choose_y))
            if len(remain_X) <= 0:
                current_node.child.append(Node(value=max_index))
            else:
                current_node.child.append(self.build_tree(remain_X,remain_y))
            return current_node

    def find_best_spilt(self, X, y):
        ent_D = self.get_ent(y)
        sample_count = len(y)

        threshold = np.zeros(len(X.columns))
        ent = np.zeros(len(X.columns))
        current = 0
        for col in X.columns : 
            sample_col = X[col]
            sample_unique = np.unique(sample_col)
            if len(sample_unique) > 1:
                sample_decide = np.zeros(len(sample_unique)-1)
                for i in range(len(sample_unique)-1) :
                    sample_decide[i] = (sample_unique[i] + sample_unique[i+1])/2 
                record_ent = np.zeros(len(sample_decide))
                for i in range(len(sample_decide)) : 
                    choose_indices = X[col] < sample_decide[i]
                    choose_y = y[choose_indices]
                    remain_y = y[~choose_indices]
                    choose_y_len = len(choose_y)
                    remain_y_len = len(remain_y)
                    choose_y_ent = self.get_ent(choose_y)
                    remain_y_ent = self.get_ent(remain_y)
                    current_ent = (choose_y_len * choose_y_ent + remain_y_len * remain_y_ent) / sample_count
                    add_ent = ent_D - current_ent
                    record_ent[i] = add_ent
                max_index = np.argmax(record_ent)
                threshold[current] = sample_decide[max_index]
                ent[current] = record_ent[max_index]
            current = current + 1
        max_index = np.argmax(ent)
        max_threshold = threshold[max_index]
        return Node(feature_index=X.columns[max_index],threshold=max_threshold)

        
    # 计算信息熵  
    def get_ent(self, y):
        y_unique = np.unique(y)
        count_array = np.zeros(np.max(y_unique)+1)
        for k in y:
            count_array[k] = count_array[k] + 1
        all_counts = len(y)
        if all_counts == 0 or len(y_unique) == 1:
            return 0
        else:
            for i in y_unique:
                count_array[i] = count_array[i] / all_counts
            ent = 0
            for i in range(len(count_array)):
                if count_array[i] > 0:
                    ent = ent - count_array[i]*np.log2(count_array[i])
            return ent

    # 对测试集X进行判断
    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sample_i = X.iloc[i , :]
            y[i] = self.get_decision(self.tree, sample_i) 
        return y
    
    # 递归遍历决策树
    def get_decision(self, node, sample):
        if node.value == -1:
            if sample[node.feature_index] > node.threshold : 
                return self.get_decision(node.child[1], sample)
            else :
                return self.get_decision(node.child[0], sample)
        else :
            return node.value
        


os.chdir(r'C:\Users\86153\Desktop\AI\Lab\lab2\ai_2024sp_exp2_final\part_1')

def load_data(datapath: str = './data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight','FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    discrete_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    labelencoder = LabelEncoder()
    for col in discrete_features:
        X[col] = labelencoder.fit(X[col]).transform(X[col])
    y = labelencoder.fit(y).fit_transform(y)
    # 分割测试集和训练集，测试集20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_data('./data/ObesityDataSet_raw_and_data_sinthetic.csv')
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(accuracy(y_test, y_pred))