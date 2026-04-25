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

# Data Seti Okuma;

data = pd.read_csv(r"C:\Users\kingm\OneDrive\Masaüstü\Phyton\Students Stress Level and Data Analysis\Data.csv")
print("Data Seti Basit Görüntüleme;")
print("-"*100)
data.info()
data.describe().T
print("\nSütun İsimleri;")
print("-"*100)
print(data.columns.to_list())
print("\n Data Tipleri;")
print(data.dtypes)

# Eksik Değer Analizi;
print("\n Eksik Değer Analizi;")
print("-"*100)

Eksik_Değerler=data.isnull().sum()
if Eksik_Değerler.sum()==0:
    print("\nEksik Değer Yoktur.")
else:
    print(("\nHer Sütundaki Eksik Değer Sayısı;"))
    print(Eksik_Değerler)
print("-" * 100)

# Duplicate(Çift) Analizi;
print("\n Duplicate(Çift) Analizi;")
duplicate = data.duplicated().sum()
if duplicate == 0:
    print("\nData Setinde Duplicate(Çift) Değer Yoktur.")       
else:
    print("\nData Setinde Duplicate(Çift) Değer Vardır.")
    print("Duplicate(Çift) Değer Sayısı:", duplicate)

# Kategorik ve Sayısal Değişkenlerin Belirlenmesi;
print("\n Kategorik ve Sayısal Değişkenlerin Belirlenmesi;")
kategorik = data.select_dtypes(include=['object']).columns

for col in data.columns:
    if data[col].dtype == 'object':
        plt.figure(figsize=(16, 12))
        plt.title(f'{col}',fontsize=16,fontweight='bold',color='blue')
        ax=sns.countplot(data=data, x=col, palette='viridis')
        plt.xticks(rotation=45)

        for p in ax.patches:# Grafik Üzerine Değer Yazdırma
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 5,f"{height}",ha="center", va="bottom",fontweight='bold',color='black')
            
        plt.tight_layout()
        print(f"\nKategorik Değişken: {col} Görselleştirme Tamamlandı.")
        print(data[col].value_counts())
        plt.savefig("Kategorik_görselleştirme.png",dpi=300,bbox_inches='tight')
        plt.show()

print("-" * 100)
        
# Tek Değişken Analizi;
print("\n Tek Değişken Analizi;")
fig,axes=plt.subplots(5,5,figsize=(30,30))
axes=axes.ravel()
Not_Object=data.select_dtypes(include=np.number).columns

for index,col in enumerate(Not_Object):
    axes[index].hist(data[col],bins=10,color='green',alpha=0.7,edgecolor='black')
    axes[index].set_title("Tek Değişken Analizi",fontsize=12,fontweight='bold',color='blue')
    axes[index].set_xlabel(f'{col}',fontsize=10,fontweight='bold',color='black')
    axes[index].set_ylabel('Frekans',fontsize=10,fontweight='bold',color='black')

    mean_value=data[col].mean()
    median_value=data[col].median()
    axes[index].axvline(mean_value,color='red',linestyle='dashed',linewidth=2, label=f'Mean: {mean_value:.2f}')
    axes[index].axvline(median_value,color='yellow',linestyle='dashed',linewidth=2, label=f'Median: {median_value:.2f}')
    axes[index].legend()

plt.tight_layout()
plt.savefig("Tek_Değişken_Analizi.png",dpi=300,bbox_inches='tight')
plt.show()
print("\n Tek Değişken Analizi Tamamlandı.")

skweness_values = data.skew(numeric_only=True)
print("\n Skewness (Çarpıklık) Değerleri:")
print(skweness_values.sort_values(ascending=False))
print("-" * 100)


# Değişken Değerleri Analizi;
print("\n Değişken Değerleri Analizi;")

fig,axes=plt.subplots(figsize=(16,8))
age_count = data['Age'].value_counts()

axes.plot(age_count.index, age_count.values, marker='o', linestyle='-', color='blue',markersize=8,linewidth=2)
axes.set_xlabel("Yaş Değerleri", fontsize=10, fontweight='bold', color='black')
axes.set_ylabel("Yaş Değerleri Frekansları", fontsize=10, fontweight='bold', color='black')
axes.set_title("Yaş Değişkeni Değerleri Analizi", fontsize=16,fontweight='bold', color='Green')
axes.grid(True,alpha=0.7)

plt.savefig("Yaş_Değişkeni_Değerleri_Analizi.png",dpi=300,bbox_inches='tight')
plt.show()

print("\n Yaş Analizi;")
print(age_count.sort_values(ascending=False))
print("\n En Yaygın Yaş Değeri:", age_count.idxmax())
print("\n En Az Yaygın Yaş Değeri:", age_count.idxmin())
print("\n Yaş Değerleri Yüzdeleri:")
print((age_count / age_count.sum() * 100).round(2))
print("\n Değişken Değerleri Analizi Tamamlandı.")
print("-" * 100)

# Korelasyon Analizi;
print("\n Korelasyon Analizi;")
print("-" * 100)

numeric_data=data.select_dtypes(include=np.number)
Sayısal_Değişkenler=data.select_dtypes(include=np.number)
Sayısal_Değişkenler_Kolerasyon=numeric_data.corr()

plt.figure(figsize=(16,14))
mask=np.triu(np.ones_like(Sayısal_Değişkenler_Kolerasyon,dtype=bool))
sns.heatmap(Sayısal_Değişkenler_Kolerasyon,mask=mask,annot=True,fmt=".2f",cmap='coolwarm',linewidths=1,square=True,linecolor='black', cbar_kws={"shrink": .8})

plt.title("Korelasyon Matrisi",fontsize=16,fontweight='bold',color='blue',pad=20)
plt.tight_layout()
plt.savefig("Korelasyon_Matrisi.png",dpi=300,bbox_inches='tight')
plt.show()
print("\n Korelasyon Analizi Tamamlandı.")


# Cinsiyet ile Sorular arasındaki ilişki Analizi;
print("\n Cinsiyet ile Sorular arasındaki ilişki Analizi;")

fig,axes=plt.subplots(5,5,figsize=(30,30))
axes=axes.ravel()
for index,col in enumerate(data.columns[0:25]):
    sns.countplot(data=data,x=col,hue='Gender',palette='viridis',ax=axes[index])
    axes[index].set_title(f'Cinsiyet ile {col} arasındaki ilişki',fontsize=10,fontweight='bold',color='black')
    axes[index].set_xlabel(f'{col}',fontsize=8,fontweight='bold',color='black')
plt.tight_layout()
plt.savefig("Cinsiyet_ile_Sorular_arasındaki_ilişki.png",dpi=300,bbox_inches='tight')
plt.show()
print("\n Cinsiyet ile Sorular arasındaki ilişki Analizi Tamamlandı.")
print("-" * 100)

# Çok Doğrusallık Analizi;
print("\n Çok Doğrusallık Analizi;")

high_corr=[]

for i in range(len(Sayısal_Değişkenler_Kolerasyon.columns)):
    for j in range(i+1,len(Sayısal_Değişkenler_Kolerasyon.columns)):
        if abs(Sayısal_Değişkenler_Kolerasyon.iloc[i,j])>0.8:
            high_corr.append({"Feature-1":Sayısal_Değişkenler_Kolerasyon.columns[i],"Feature-2":Sayısal_Değişkenler_Kolerasyon.columns[j],"Correlation":Sayısal_Değişkenler_Kolerasyon.iloc[i,j]})

if high_corr:
    high_corr_data=pd.DataFrame(high_corr)
    print("\nYüksek Korelasyona Sahip Değişken Çiftleri (|Korelasyon| > 0.8):")
    print(high_corr_data)
else:
    print("\nYüksek Korelasyona Sahip Değişken Çiftleri Bulunamadı.")
    print("-" * 100)

print("\n Çok Doğrusallık Analizi Tamamlandı.")
print("-" * 100)

# Aykırı Değer Analizi;
print("\n Aykırı Değer Analizi;")

özet={}
for col in numeric_data.columns:
    Q1=data[col].quantile(0.25)
    Q3=data[col].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=numeric_data[(numeric_data[col]<lower_bound) | (numeric_data[col]>upper_bound)]
    özet[col]={'Değer':len(outliers), 'Yüzdelik':(len(outliers)/len(data))*100}

outliers_data=pd.DataFrame(özet).T
outliers_data=outliers_data.sort_values(by='Değer',ascending=False)
print("\n Aykırı Değer Özeti:")
print(outliers_data)

# Aykırı Değerlerin Boxplot Görselleştirmesi;
print("Aykırı Değerlerin Boxplot Görselleştirmesi;")
fig,axes=plt.subplots(3,2,figsize=(16,10))
axes=axes.ravel()

for index,col in enumerate(data.columns[0:6]):

    sns.boxplot(data=numeric_data,x=col,color='lightblue',ax=axes[index])
    axes[index].set_xlabel(f'{col}',fontsize=10,fontweight='bold',color='black')

plt.tight_layout()
plt.savefig("Aykırı_Değer_Boxplot_Görselleştirmesi.png",dpi=300,bbox_inches='tight')
plt.show()

print("\n Aykırı Değerlerin IQR Metodu;")
print(outliers_data[outliers_data['Değer']>0])
print("\n Aykırı Değer Analizi Tamamlandı.")
print("-" * 100)

# Özellik Önemi Analizi;
print("\nÖzellik Önemi Analizi")
print("-" * 100)

hedef_sütun="Age"
özellik_sütunu=[col for col in numeric_data if col != hedef_sütun]
X = data[özellik_sütunu]
Y = data[hedef_sütun]

mi_Scores=mutual_info_regression(X,Y,random_state=42)
mi_Scores_df=pd.DataFrame({'Feature':X.columns,'MI Score':mi_Scores}).sort_values('MI Score',ascending=False)

print("\nÖzellik Önemi Analizi:")
print(mi_Scores_df)

plt.figure(figsize=(10,12))
colors=plt.cm.Oranges(np.linspace(0.4,0.9,len(mi_Scores)))
plt.barh(mi_Scores_df['Feature'],mi_Scores_df["MI Score"],color=colors)
plt.xlabel('MI Score',fontsize=12,fontweight='bold')
plt.ylabel('Frekans',fontsize=12,fontweight='bold')

plt.gca().invert_yaxis()
plt.grid(axis='x',alpha=0.3)
plt.tight_layout()
plt.savefig("Özellik Önemi.png",dpi=300,bbox_inches='tight')
plt.show()
print("-" * 100)

# PCA
print("\nPCA")
print("-" * 100)

scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)

pca=PCA()
x_pca=pca.fit_transform(x_scaled)

aciklanan_varyans=pca.explained_variance_ratio_
cum_varyans=np.cumsum(aciklanan_varyans)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))

ax1.plot(range(1,len(aciklanan_varyans)+1),aciklanan_varyans,marker="o",color='steelblue',markersize=8)
ax1.set_xlabel('Temel Bileşen',fontsize=12)
ax1.set_ylabel("Açıklanan Varyans",fontsize=12)
ax1.set_title("PCA Scree Plot",fontsize=14,fontweight='bold')
ax.grid(True,alpha=0.3)

ax2.plot(range(1,len(cum_varyans)+1),cum_varyans,marker="o",color='darkgreen',markersize=8,linewidth=2)
ax2.axhline(y=0.95,color="darkorange",linestyle="--",linewidth=2,label="%95 Varyans")
ax2.set_xlabel('Bileşen Sayısı',fontsize=12)
ax2.set_ylabel('Kümulatif Açıklanan Varyans',fontsize=12)
ax2.set_title("Kümulatif Varyans",fontsize=14,fontweight='bold')

ax2.legend()
ax2.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analizi.png',dpi=300,bbox_inches='tight')
plt.show()

n_bilesen_sayisi=np.argmax(cum_varyans>=95)+1
print(f"\n%95 Varyans:{n_bilesen_sayisi}")
print("-" * 100)

# İstatistik Testi;
print("İstatistik Testi")
print("-" * 100)

statistic,p_value=stats.normaltest(data[hedef_sütun])
print(f"Normal Test:")
print(f"İstatistik:{statistic:.4f}")
print(f"P-Value:{p_value:.4f}")
print(f"Normal Dağılıma Sahip mi?{'Yes' if p_value>0.05 else 'No'}")

# ANOVA
print("\nANOVA Testi;")
korelasyon=Sayısal_Değişkenler_Kolerasyon[hedef_sütun].drop(hedef_sütun).sort_values(ascending=False)

for feature in korelasyon.abs().nlargest(3).index:
    groups=[group[feature].values for name,group in data.groupby(hedef_sütun)] 
    f_stat,p_val=stats.f_oneway(*groups)
    short_feature=feature[:50]+"...." if len(feature)>50 else feature
    print(f"\n{short_feature}")
    print(f"F-Statistic:{f_stat:.4f}")
    print(f"P-Value:{p_val:.4f}")
    print(f"Önemli Farklılıklar Var mı?{'Yes' if p_val<0.05 else 'No'}")






