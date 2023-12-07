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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7172, random_state=1)

# 定义参数范围
k_range = list(range(1, 101))
param_grid = dict(n_neighbors=k_range)

# 创建KNN模型
knn = KNeighborsClassifier()

# 使用GridSearchCV进行参数搜索和交叉验证
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)

# 打印最优参数和对应的准确率
print('最优参数：', grid.best_params_)
print('最优参数对应的准确率：', grid.best_score_)

# 从CSV文件中读取测试集数据
df_test = pd.read_csv(r'D:\Desktop\个人\实习\埃森哲\消费者画像建立\测试集.csv')

# 删除包含缺失值的行
df_test = df_test.dropna()

# 将测试集的分类特征转换为数值形式
df_test = pd.get_dummies(df_test, columns=['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])

# 使用最优参数的模型进行预测
y_pred = grid.predict(df_test)

# 将预测结果转换为字母形式
y_pred = le.inverse_transform(y_pred)

# 打印预测结果
print('测试集的预测结果：', y_pred)

# 将预测结果保存为CSV文件
pd.DataFrame(y_pred, columns=['Segmentation']).to_csv(r'D:\Desktop\个人\实习\埃森哲\消费者画像建立\segmentation预测值.csv', index=False)
