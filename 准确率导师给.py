# 确定最优参数
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# 提取特征和标签

df = pd.read_csv('D:\Desktop\个人\实习\埃森哲\消费者画像建立\消费者画像分类转数值ces1.csv')
df = pd.get_dummies(df, columns=['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])
X = df.drop('Segmentation', axis=1)
y = df['Segmentation']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# 定义参数范围
param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}

# 创建KNN模型
knn = KNeighborsClassifier()

# 使用GridSearchCV进行参数搜索和交叉验证
grid_search = GridSearchCV(knn, param_grid,cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 打印最优参数和对应的准确率
print("Best Parameter: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)
# 进行模型训练

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=7)

# 训练分类器
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

print("准确率:", accuracy)


