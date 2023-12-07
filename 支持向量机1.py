import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\Deng.ttf", size=14) 

df = pd.read_csv('D:\Desktop\个人\实习\埃森哲\消费者画像建立\消费者画像分类转数值ces1.csv')

# 对指定列进行独热编码
df = pd.get_dummies(df, columns=['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])

# 获取独热编码生成的列名
encoded_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])]

# 添加数值型特征
features = encoded_columns + ['Age', 'Work_Experience', 'Family_Size']

# 选择特征
x = df[features]

# 选择目标变量
y = df['Segmentation']

# 将标签编码为数值
le = LabelEncoder()
y = le.fit_transform(y)

# 划分训练集和测试集
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=1)

# 定义参数范围,使用网格搜索和交叉验证
param_grid = {'C': [1000,1500,1800,1900,2000,2100,2150,2200,2250,2300,2400], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 

# 创建SVM模型
svm = SVC()

# 使用GridSearchCV进行网格参数搜索和交叉验证
grid = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy')
grid.fit(x_train, y_train)

# 打印最优参数和对应的准确率
print('最优参数：', grid.best_params_)
print('最优参数对应的准确率：', grid.best_score_)

# 从CSV文件中读取测试集数据
df_test = pd.read_csv(r'D:\Desktop\个人\实习\埃森哲\消费者画像建立\测试集.csv')

# 删除包含缺失值的行
df_test = df_test.dropna()

# 保存'ID'列
id_column = df_test['ID']

# 将测试集的分类特征转换为数值形式
df_test = pd.get_dummies(df_test, columns=['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])

# 获取独热编码生成的列名
encoded_columns = [col for col in df_test.columns if any(col.startswith(prefix) for prefix in ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])]

# 添加数值型特征
features = encoded_columns + ['Age', 'Work_Experience', 'Family_Size']

# 选择特征
x_test = df_test[features]

#创建SVM模型
svm = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel=grid.best_params_['kernel'])

#训练模型
svm.fit(x_train,y_train)

# 使用最优参数的模型进行预测
y_pred = svm.predict(x_test)

# 将预测结果转换为字母形式
y_pred = le.inverse_transform(y_pred)

# 创建一个新的数据框来保存预测结果和对应的'ID'
result = pd.DataFrame({'ID': id_column, 'Segmentation': y_pred})

# 打印预测结果
print('测试集的预测结果：', result)

# 将预测结果保存为CSV文件
result.to_csv(r'D:\Desktop\个人\实习\埃森哲\消费者画像建立\segmentation预测值SVM3.csv', index=False)

# 绘制饼图
labels, counts = np.unique(y_pred, return_counts=True)
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.title('客户类型占比', fontproperties=font)
plt.show()
