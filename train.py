# -*- coding: utf-8 -*-
"""
Titanic 生存预测 - 集成学习模型对比
使用三种集成算法：随机森林、梯度提升树、直方图梯度提升树
"""

import warnings
warnings.filterwarnings('ignore')

# 基础库
import pandas as pd
import numpy as np

# 机器学习库
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 绘图库
import matplotlib.pyplot as plt

# 设置 matplotlib 风格（避免中文乱码，若系统无中文字体则使用英文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 也可尝试 'SimHei' 如果系统有
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("Titanic 生存预测 - 集成学习模型对比实验")
print("="*60)

# ----------------------------- 1. 加载数据 ----------------------------------
print("\n[1] 加载数据集...")
# 方式一：使用 seaborn 内置数据集（无需额外下载）
try:
    import seaborn as sns
    titanic = sns.load_dataset('titanic')
    print("   使用 seaborn 内置数据集，共 {} 条记录".format(len(titanic)))
except Exception as e:
    # 备选：从本地 CSV 读取（如果需要）
    print("    seaborn 加载失败，尝试读取本地 train.csv ...")
    titanic = pd.read_csv('/root/autodl-tmp/train.csv')
    # 将列名统一为 seaborn 的格式
    titanic.rename(columns={
        'Survived': 'survived', 'Pclass': 'pclass', 'Sex': 'sex',
        'Age': 'age', 'SibSp': 'sibsp', 'Parch': 'parch',
        'Fare': 'fare', 'Embarked': 'embarked'
    }, inplace=True)
    print("   本地数据加载完成，共 {} 条记录".format(len(titanic)))

# ----------------------------- 2. 数据预处理 --------------------------------
print("\n[2] 数据预处理...")
# 处理缺失值
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
titanic['fare'].fillna(titanic['fare'].median(), inplace=True)

# 编码分类变量
le_sex = LabelEncoder()
le_emb = LabelEncoder()
titanic['sex'] = le_sex.fit_transform(titanic['sex'])  # male=1, female=0
titanic['embarked'] = le_emb.fit_transform(titanic['embarked'].astype(str))

# 选择特征
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = titanic[features]
y = titanic['survived']

# 在提取 X, y 之前，对全表所有数值列填充缺失值
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

print("   特征维度: {}".format(X.shape))
print("   目标变量分布:\n", y.value_counts())

# ----------------------------- 3. 定义模型 ---------------------------------
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'HistGradient Boosting': HistGradientBoostingClassifier(random_state=42)
}


# ----------------------------- 4. 交叉验证评估 ------------------------------
print("\n[3] 开始 5 折交叉验证评估...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}           # 存储平均指标
all_scores = {}        # 存储每次 fold 的分数（用于箱线图）

for name, model in models.items():
    print("   正在评估: {} ...".format(name))
    # 准确率（5折）
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    # F1 分数（宏平均）
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    
    results[name] = {
        'Accuracy': acc_scores.mean(),
        'Accuracy_std': acc_scores.std(),
        'F1': f1_scores.mean()
    }
    all_scores[name] = acc_scores   # 保存原始得分
    
    print("       Accuracy = {:.4f} ± {:.4f}".format(acc_scores.mean(), acc_scores.std()))
    print("       F1 score = {:.4f}".format(f1_scores.mean()))

# 输出汇总表格
print("\n[4] 汇总结果（表格）:")
results_df = pd.DataFrame(results).T
print(results_df.round(4))

# ----------------------------- 5. 绘图 -------------------------------------
print("\n[5] 生成对比图表...")

# 准备数据
model_names = list(results.keys())
accuracies = [results[name]['Accuracy'] for name in model_names]
errors = [results[name]['Accuracy_std'] for name in model_names]

# 图1：柱状图（带误差棒）
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, yerr=errors, capsize=8,
               color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy Comparison (5-fold Cross Validation)', fontsize=14)
plt.ylim(0.7, 0.9)  # 根据实际输出可调整
# 在柱顶显示数值
for bar, acc, err in zip(bars, accuracies, errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/accuracy_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# 图2：箱线图（展示5次交叉验证的得分分布）
plt.figure(figsize=(8, 5))
box_data = [all_scores[name] for name in model_names]
bp = plt.boxplot(box_data, labels=model_names, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', linewidth=1.5),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
plt.ylabel('Accuracy', fontsize=12)
plt.title('Cross-Validation Score Distribution', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n[6] 图片已保存至:")
print("   - /root/autodl-tmp/accuracy_bar.png")
print("   - /root/autodl-tmp/boxplot.png")

# 找出最佳模型
best_model = max(results, key=lambda x: results[x]['Accuracy'])
print("\n[7] 最佳模型: {} (准确率 = {:.4f})".format(best_model, results[best_model]['Accuracy']))

print("\n" + "="*60)
print("实验完成！")
print("="*60)