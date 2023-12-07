import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 假设df是你的数据集
df = pd.read_csv('D:\Desktop\个人\实习\埃森哲\消费者画像建立\消费者画像分类转数值ces1.csv')

# 将分类特征转换为数值形式
df = pd.get_dummies(df, columns=['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])

# 提取特征和标签
X = df.drop('Segmentation', axis=1)
y = df['Segmentation']

# 将标签编码为数值
le = LabelEncoder()
y = le.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)

# 定义参数范围
k_range = list(range(1, 10))
param_grid = dict(n_neighbors=k_range)

# 创建KNN模型
KNN = KNeighborsClassifier()

# 使用GridSearchCV进行参数搜索和交叉验证
grid = GridSearchCV(KNN, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)

# 打印最优参数和对应的准确率
print('最优参数：', grid.best_params_)
print('最优参数对应的准确率：', grid.best_score_)
