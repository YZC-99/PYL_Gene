import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visual(data,path):
    correlation_matrix = np.corrcoef(data, rowvar=False)  # rowvar=False表示每列是一个特征
    # 绘制相关系数图
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(correlation_matrix,
                     square=True,
                     annot=False,
                     fmt='.3f',
                     linewidths=.5,
                     cmap='YlGnBu',
                     cbar_kws={'fraction': 0.046, 'pad': 0.03}
                     )
    plt.title('Correlation Matrix of Embedding Dimensions')
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Embedding Dimensions')
    plt.savefig(path,dpi=300)
    # plt.show()

# 读取数据
ccds_path = "../../data/human/ccds_nonan_data_with_feature.csv"
deg_path = "../../data/human/deg_nonan_data_with_feature.csv"
ccds_data = pd.read_csv(ccds_path)
deg_data = pd.read_csv(deg_path)

# 提取特征
ccds_features = ccds_data.iloc[:, -4545:]
deg_features = deg_data.iloc[:, -4545:]

# 定义降维维度的范围
n_components_range = range(1, 512)

# 计算每个维度数量对应的累计方差
explained_variances = []
for n in n_components_range:
    pca = PCA(n_components=n)
    ccds_features_pca = pca.fit_transform(ccds_features)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# 绘制累计方差与维度数量的关系图
plt.plot(n_components_range, explained_variances)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance (%)')
plt.title('PCA: Explained Variance vs Number of Components')
plt.savefig("ccds_pca.png",dpi=500)
# plt.show()


# # 计算每个维度数量对应的累计方差
explained_variances = []
for n in n_components_range:
    pca = PCA(n_components=n)
    deg_features = pca.fit_transform(deg_features)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# 绘制累计方差与维度数量的关系图
plt.plot(n_components_range, explained_variances)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance (%)')
plt.title('PCA: Explained Variance vs Number of Components')
plt.savefig("deg_pca.png",dpi=500)
# plt.show()


# # 定义降维维度
# n_components = 64  # 自定义的降维维度
#
# # 创建PCA对象并拟合数据
# pca = PCA(n_components=n_components)
# ccds_features_pca = pca.fit_transform(ccds_features)
# deg_features_pca = pca.fit_transform(deg_features)
#
#
#
# visual(ccds_features_pca,"../../data/human/ccds_nonan_data_with_pca{}feature.png".format(n_components))
# visual(deg_features_pca,"../../data/human/deg_nonan_data_with_pca{}feature.png".format(n_components))


# # 将降维后的结果转换为DataFrame对象
# ccds_features_pca_df = pd.DataFrame(ccds_features_pca, columns=[f"feature_{i+1}" for i in range(n_components)])
# deg_features_pca_df = pd.DataFrame(deg_features_pca, columns=[f"feature_{i+1}" for i in range(n_components)])
#
# # 取出ccds_features的倒数4545列以前的数据
# ccds_features_previous = ccds_data.iloc[:, :-4545]
#
# # 将降维后的结果与原始数据拼接
# ccds_features_pca_concat = pd.concat([ccds_features_previous, ccds_features_pca_df], axis=1)
#
# # 取出deg_features的倒数4545列以前的数据
# deg_features_previous = deg_data.iloc[:, :-4545]
#
# # 将降维后的结果与原始数据拼接
# deg_features_pca_concat = pd.concat([deg_features_previous, deg_features_pca_df], axis=1)
#
# ccds_features_pca_concat.to_csv("../../data/human/ccds_nonan_data_with_pca{}feature.csv".format(n_components))
# deg_features_pca_concat.to_csv("../../data/human/deg_nonan_data_with_pca{}feature.csv".format(n_components))






# 打印结果
# print("ccds_features_pca:")
# print(ccds_features_pca_concat.head())
# print(ccds_features.head())
# print("\ndeg_features_pca:")
# print(deg_features_pca_concat.head())
