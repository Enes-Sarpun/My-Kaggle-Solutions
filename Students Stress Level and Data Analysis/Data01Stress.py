import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import subplots
from pandas import DataFrame
import matplotlib.pylab as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statistics import mean, median
from scipy import stats

blue_palette=sns.color_palette("Blues")
green_palette=sns.color_palette("Greens")
red_palette=sns.color_palette("Reds")
sns.set_palette("viridis")

# Data seti okundu, gerekli bilgiler alındı ve ön işlemler yapıldı;
df = pd.read_csv(r'C:\Users\kingm\OneDrive\Masaüstü\Phyton\Students Stress Level and Data Analysis\DataStress.csv')

print("\nBasic Statistics:")
print("-" * 100)
print(df.info())
print("-" * 100)
print(df.describe().T)
print("-" * 100)
print("\nColumn Names:")
print(df.columns.tolist())

# Data Tip, Eksik Değer ve Temel İstatistikler;
print("-" * 100)
print("Data Types:")
print(df.dtypes)

print("\nMissing Values:")
print("-" * 100)
Missing_Values=df.isnull().sum()
if Missing_Values.sum() == 0:
    print("No missing values found in the dataset.")
else:
    print(Missing_Values)

print("-" * 100)

# Duplicate Analizi;
print("Duplicate Records Analysis:")
duplicates = df.duplicated().sum()
if duplicates == 0:
    print("No duplicate records found in the dataset.") 
else:
    print(f"Number of duplicate records: {duplicates}")

print("-" * 100)

# Kategorik Değişkenlerin Dağılımı;(Kategorisi farklı olan değişkenler için)
categorical_columns = df.select_dtypes(include=['object']).columns  
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    plt.title(f'Distribution of {col}', fontsize=10, fontweight='bold',color='black', loc='left')
    sns.countplot(data=df, x=col, palette="viridis")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print(f"\nValue Counts for {col}:")
    print(df[col].value_counts())

print("-" * 100)


# Tek değişken analizi
print("Özelliklerin Dağılımı-Tek Değişken Analizi")
print("-" * 100)
fig,axes=plt.subplots(7,3,figsize=(20,28))
axes=axes.ravel()

for index,col in enumerate(df.columns):
    axes[index].hist(df[col],bins=10,edgecolor="black",alpha=0.7,color="steelblue")
    axes[index].set_title(f"Distribution of {col}",fontsize=10,fontweight="bold",color="black",loc="left")
    axes[index].set_xlabel(col)
    axes[index].set_ylabel("Frequency") 
    
    mean_value=df[col].mean()
    median_value=df[col].median()
    axes[index].axvline(mean_value,color="darkgreen",linestyle="--",linewidth=2,label=f"Mean: {mean_value:.2f}")
    axes[index].axvline(median_value,color="darkorange",linestyle="--",linewidth=2,label=f"Median: {median_value:.2f}")
    axes[index].legend()

plt.tight_layout()
plt.savefig("feature_distributions.png",dpi=300,bbox_inches="tight")
plt.show()
print("-" * 100)
print("\nSkewness of Features:")
skewness = df.skew()
print(skewness.sort_values(ascending=False))
print("-" * 100)


# Değişken Değerleri Analizi
print("Değişen Değerlerin Analizi(Stress Level)")
print("-"*100)

plt.figure(figsize=(10,6))
stress_counts=df['stress_level'].value_counts().sort_index()
bars = plt.bar(stress_counts.index,stress_counts.values,edgecolor="black",color="teal",alpha=0.8)
plt.xlabel('Stress Level',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.title('Distribution of Stress Levels',fontsize=14,fontweight='bold')

for index, v in enumerate(stress_counts.values):
    plt.text(stress_counts.index[index],v+5,str(v),ha="center",va="bottom",fontweight="bold")

plt.savefig('stress_level_distribution.png',dpi=300,bbox_inches="tight")
plt.show()

print("\nStress Level Value Counts:")
print(stress_counts)
print(f"\nPercentage Distribution:")
print((stress_counts / len(df) * 100).round(2))
print("-"*100)


# Korelasyon Analizi;
print("Korelasyon Analizi")
print("-"*100)

correlation_matrix=df.corr()

plt.figure(figsize=(16,14))
mask=np.triu(np.ones_like(correlation_matrix,dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='YlGnBu', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": .8})

plt.title('Feature Correlation Matrix',fontsize=16,fontweight='bold',pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png',dpi=300,bbox_inches='tight')
plt.show()

stress_correlations = correlation_matrix['stress_level'].drop('stress_level').sort_values(ascending=False)
print("\nTop 10 Features Correlated with Stress Level:")
print(stress_correlations.head(10))
print("\nBottom 10 Features Correlated with Stress Level:")
print(stress_correlations.tail(10))


# Stres seviyeleri görselleştirme;
plt.figure(figsize=(10, 8))
colors = ['darkgreen' if x > 0 else 'darkorange' for x in stress_correlations]
stress_correlations.plot(kind='barh', color=colors)
plt.xlabel('Correlation with Stress Level', fontsize=12)
plt.title('Feature Correlations with Stress Level', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('stress_correlations.png', dpi=300, bbox_inches='tight')
plt.show()


# Çok Doğrusallık Analizi;
print("Çok Doğrusallık Analizi")
print("-"*100)

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({'feature-1': correlation_matrix.columns[i], 'feature-2': correlation_matrix.columns[j], 'correlation': correlation_matrix.iloc[i, j]})

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
    print(high_corr_df)
else:
    print("No highly correlated feature pairs found.")
print("-"*100)


# İkili Değişken Analizi
print("İkili Değişken Analizi")
print("-"*100)
top_features = stress_correlations.abs().nlargest(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i,feature in enumerate(top_features):
    data_to_plot = [df[df['stress_level'] == level][feature].values 
                    for level in sorted(df['stress_level'].unique())]
    bp=axes[i].boxplot(data_to_plot,patch_artist=True)
    colors=plt.cm.Blues(np.linspace(0.4,0.9,len(bp['boxes'])))
    for patch,color in zip(bp['boxes'],colors):
        patch.set_facecolor(color)
    
    for elements in ['whiskers','fliers','caps','medians']:
        plt.setp(bp[elements],color='black',linewidth=1.5)

    axes[i].set_title(f'{feature} vs Stress Level',fontsize=10,fontweight='bold',color='black',loc='left')
    axes[i].set_xlabel('Stress Level')
    axes[i].set_ylabel(feature)
    axes[i].set_xticklabels(sorted(df['stress_level'].unique()))

plt.suptitle('Top Features vs Stress Level', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('top_features_vs_stress_level.png', dpi=300, bbox_inches='tight')
plt.show()
print("-"*100)


# Aykırı Değer Analizi;
print("Aykırı Değer Analizi")
print("-"*100)

outlier_summary = {}
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = {
        'count': len(outliers),
        'percentage_outliers': (len(outliers) / len(df)) * 100
    }

outlier_df = pd.DataFrame(outlier_summary).T
outlier_df = outlier_df.sort_values(by='count', ascending=False)
print("\nOutlier Summary(IQR Method):")
print(outlier_df[outlier_df['count'] > 0])
print("-"*100)


# Aykırı değerlerin görselleştirilmesi;

plt.figure(figsize=(12,6))
outlier_df[outlier_df['count'] > 0]['percentage_outliers'].plot(kind='bar', color='salmon', edgecolor='black',alpha=0.8)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Percentage of Outliers', fontsize=12)
plt.title('Percentage of Outliers by Feature', fontsize=14, fontweight='bold')
plt.xticks(rotation=45,ha='right')
plt.grid(axis='y',alpha=0.75)
plt.tight_layout()
plt.savefig('outlier_percentage.png', dpi=300, bbox_inches='tight')
plt.show()
print("-"*100)


# Özellik Önemi Analizi;
print("Özellik Önemi Analizi")
print("-"*100)
x=df.drop('stress_level',axis=1)# Bağımsız değişkenler
y=df['stress_level']# Hedef değişken

# Bize dönen grafik ve özellikler Hedef değişken ile en güçü bağa sahip olanları gösteriyor.
mi_scores = mutual_info_regression(x, y, random_state=42)
mi_scores = pd.DataFrame({'Feature': x.columns, 'MI Score': mi_scores}).sort_values(by='MI Score', ascending=False)

print("\nMutual Information Scores:")
print(mi_scores)
print("-"*100)

plt.figure(figsize=(10,8))

colors=plt.cm.viridis(np.linspace(0.4,0.9,len(mi_scores)))
plt.barh(mi_scores['Feature'], mi_scores['MI Score'], color=colors, edgecolor='black',alpha=0.8)
plt.xlabel('Mutual Information Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance based on Mutual Information', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_mi.png', dpi=300, bbox_inches='tight')
plt.show()
print("-"*100)


# Boyutluk Azaltma Analizi;
print("Boyutluk Azaltma Analizi (PCA)")
print("-"*100)

StandardScaler=StandardScaler()
x_scaled=StandardScaler.fit_transform(x)

pca=PCA()
x_pca=pca.fit_transform(x_scaled)

expalined_variance=pca.explained_variance_ratio_
cum_explained_variance=np.cumsum(expalined_variance)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))
ax1.plot(range(1,len(expalined_variance)+1),expalined_variance,marker='o',color='green',markersize=8,linewidth=2)
ax1.set_xlabel('Number of Principal Components',fontsize=12)
ax1.set_ylabel('Explained Variance Ratio',fontsize=12)
ax1.set_title('Explained Variance by Principal Components',fontsize=14,fontweight='bold')
ax1.grid(True,alpha=0.3)

ax2.plot(range(1,len(cum_explained_variance)+1),cum_explained_variance,marker='o',color='blue',markersize=8,linewidth=2)
ax2.axhline(y=0.95,color='red',linestyle='--',linewidth=1,label='95% Variance')
ax2.set_xlabel('Number of Principal Components',fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance',fontsize=12)
ax2.set_title('Cumulative Explained Variance by Principal Components',fontsize=14,fontweight='bold')
ax2.legend()
ax2.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('pca_explained_variance.png',dpi=300,bbox_inches='tight')
plt.show()

n_components_95 = np.argmax(cum_explained_variance >= 0.95) + 1
print(f"\nNumber of Principal Components to explain 95% variance: {n_components_95}")
print("-"*100)


# İstatiksel Test Analizi;
print("İstatiksel Test Analizi (ANOVA)")    
print("-"*100)

statics, p_values = stats.normaltest(df['stress_level'])
print(f"\nNormality Test For Stress Level")
print(f"Statistic:{statics:.4f}")
print(f"P-Value:{p_values:.4f}")
print(f"Is normally distributed? {'Yes' if p_values > 0.05 else 'No'}")
print("-"*100)

print(f"\nANOVA Test (Feature groups by Stress Level):")
for feature in ['anxiety_level', 'depression', 'academic_performance']:
    group=[group[feature].values for name,group in df.groupby('stress_level')]
    fstat, p=stats.f_oneway(*group)
    print(f"\nFeature: {feature}")
    print(f"F-Statistic: {fstat:.4f}")
    print(f"P-Value: {p:.4f}")
    print(f"Significant difference? {'Yes' if p < 0.05 else 'No'}")
print("-"*100)


# Özet Ve Raporlama;
summary_repors=(f""" 
Data Analysis Summary Report
================================================
1. Dataset Overview:
   - Total Records: {df.shape[0]}
   - Total Features: {df.shape[1]}
   - No missing values or duplicate records found.
   - Duplicates: {duplicates}

Target Variable Distribution:
{stress_counts.value_counts().to_dict()}
Top 5 Features Correlated with Stress Level:
{stress_correlations.head(5).to_dict()}
Feature Importance (Top 5 by Mutual Information):
{mi_scores.head(5).to_dict()}
Dimensionality Reduction:
    - Compenents to explain 95% variance: {n_components_95}
Outliers Summary:
    - Features with %5>outliers: {len(outlier_df[outlier_df['percentage_outliers'] > 5])}
Preprocessing Recommendations:
    1.Scaling: StandardScaler
    2.Feature Selection: Start with top {len(mi_scores[mi_scores['MI Score'] > 0])} features by MI score.
    3.Handle outliers based on analysis.    
    4.Consider ensemble methods for modeling.
""")

with open('data_analysis_summary.txt','w') as f:
    f.write(summary_repors)

print("\n"+"="*100)
print("Data Analysis Summary Report")
print("="*100)
print("\nFiles Saved")
print("- data_analysis_summary.txt")
print("- feature_distributions.png")
print("- correlation_matrix.png")
print("- stress_correlations.png")
print("- stress_level_distribution.png")
print("- top_features_vs_stress_level.png")
print("- outlier_percentage.png")
print("- feature_importance_mi.png")
print("- pca_explained_variance.png")
print("\nReady for Modeling!")
