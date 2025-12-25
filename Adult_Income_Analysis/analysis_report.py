# 1. 导入所需库（4种以上第三方库）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os

# 2. 设置路径与环境配置
# 图片保存路径（确保路径存在，不存在则创建）
save_path = r'F:\Python\PythonProject\Adult_Income_Analysis'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 设置中文显示（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)  # 默认图表大小

# 3. 数据加载
# 定义字段名（对应adult.data的15个特征）
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_per_week', 'native_country', 'income']

# 读取CSV文件（若你的文件是txt格式，将read_csv改为read_csv，sep=','保持不变）
df = pd.read_csv('adult_data.csv', names=columns, sep=',', skipinitialspace=True)  # skipinitialspace忽略字段间空格

# 4. 数据预处理与清洗
print("=== 数据清洗前基本信息 ===")
print(f"数据形状：{df.shape}")
print(f"缺失值统计：\n{df.isnull().sum()}")

# 4.1 处理缺失值（将'?'替换为NaN并删除）
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)  # 删除含缺失值的行
print(f"\n删除缺失值后数据形状：{df.shape}")

# 4.2 去除重复值
df.drop_duplicates(inplace=True)
print(f"删除重复值后数据形状：{df.shape}")

# 4.3 数据类型转换（确保数值型字段正确）
numeric_cols = ['age', 'education_num', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4.4 分类特征编码（用于相关性分析）
label_encoders = {}
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                   'relationship', 'race', 'sex', 'native_country', 'income']
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4.5 数值特征标准化（可选，用于后续建模，此处展示功能）
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n=== 数据清洗后基本信息 ===")
print(f"最终数据形状：{df.shape}")
print(f"数据类型：\n{df.dtypes[:10]}")  # 展示前10个字段类型

# 5. 数据分布可视化（保存图片到指定路径）
# 5.1 收入分布柱状图
plt.figure(figsize=(8, 5))
income_count = df['income'].value_counts()
income_count.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('收入分布情况', fontsize=14, fontweight='bold')
plt.xlabel('收入水平', fontsize=12)
plt.ylabel('人数', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(save_path, '收入分布.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 收入分布图已保存")

# 5.2 年龄分布直方图
plt.figure(figsize=(10, 5))
sns.histplot(df['age'], bins=20, kde=True, color='#2ca02c', edgecolor='black')
plt.title('年龄分布直方图', fontsize=14, fontweight='bold')
plt.xlabel('年龄（标准化后）', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(save_path, '年龄分布.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 年龄分布图已保存")

# 5.3 教育程度分布饼图（取前8个主要教育程度）
top_education = df['education'].value_counts().head(8)
plt.figure(figsize=(10, 8))
plt.pie(top_education.values, labels=top_education.index, autopct='%1.1f%%',
        colors=plt.cm.Set3(np.linspace(0, 1, len(top_education))), startangle=90)
plt.title('主要教育程度分布', fontsize=14, fontweight='bold')
plt.axis('equal')  # 保证饼图为正圆形
plt.savefig(os.path.join(save_path, '教育程度分布.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 教育程度分布图已保存")

# 5.4 工作时长与收入关系箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='hours_per_week', data=df, hue='income', palette=['#d62728', '#9467bd'], legend=False)
plt.title('不同收入水平的工作时长分布', fontsize=14, fontweight='bold')
plt.xlabel('收入水平', fontsize=12)
plt.ylabel('每周工作时长（标准化后）', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(save_path, '工作时长与收入关系.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 工作时长与收入关系图已保存")

# 6. 相关性分析
# 6.1 数值特征相关性矩阵
corr_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss', 'income_encoded']
corr_matrix = df[corr_cols].corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('特征相关性热力图', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(save_path, '特征相关性热力图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 相关性热力图已保存")

# 6.2 性别与收入关系交叉分析（可视化）
sex_income = pd.crosstab(df['sex'], df['income'], normalize='index') * 100
plt.figure(figsize=(8, 5))
sex_income.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('不同性别的收入占比', fontsize=14, fontweight='bold')
plt.xlabel('性别', fontsize=12)
plt.ylabel('占比（%）', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.legend(title='收入水平')
plt.savefig(os.path.join(save_path, '性别与收入关系.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 性别与收入关系图已保存")

# 6.3 职业与收入关系（取前10个职业）
top_occupations = df['occupation'].value_counts().head(10).index
occ_income = pd.crosstab(df[df['occupation'].isin(top_occupations)]['occupation'],
                         df[df['occupation'].isin(top_occupations)]['income'],
                         normalize='index') * 100

plt.figure(figsize=(12, 6))
occ_income['>50K'].sort_values(ascending=False).plot(kind='bar', color='#8c564b')
plt.title('主要职业的高收入（>50K）占比', fontsize=14, fontweight='bold')
plt.xlabel('职业', fontsize=12)
plt.ylabel('高收入占比（%）', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(save_path, '职业与高收入关系.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 职业与高收入关系图已保存")

# 7. 输出关键统计结果（用于报告撰写）
print("\n=== 关键统计结果 ===")
print("1. 收入占比：")
print(df['income'].value_counts(normalize=True) * 100)

print("\n2. 特征相关性排序（与收入相关性）：")
income_corr = corr_matrix['income_encoded'].sort_values(ascending=False)
print(income_corr)

print("\n3. 不同性别的高收入占比：")
print(sex_income['>50K'])

print("\n🎉 所有代码执行完成！图片已保存至：", save_path)