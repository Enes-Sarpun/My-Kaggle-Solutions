import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "mimic-iii-clinical-database-demo-1.4/"

print("=" * 60)
print("BÖLÜM 1: VERİ YÜKLEME VE KEŞİF")
print("=" * 60)

print("\n📊 Tablolar yükleniyor...")

patients = pd.read_csv(DATA_PATH + "PATIENTS.csv")
print(f"✓ PATIENTS: {len(patients)} hasta")

admissions = pd.read_csv(DATA_PATH + "ADMISSIONS.csv")
print(f"✓ ADMISSIONS: {len(admissions)} yatış kaydı")

icustays = pd.read_csv(DATA_PATH + "ICUSTAYS.csv")
print(f"✓ ICUSTAYS: {len(icustays)} yoğun bakım kaydı")

diagnoses = pd.read_csv(DATA_PATH + "DIAGNOSES_ICD.csv")
print(f"✓ DIAGNOSES_ICD: {len(diagnoses)} teşhis kaydı")

labevents = pd.read_csv(DATA_PATH + "LABEVENTS.csv")
print(f"✓ LABEVENTS: {len(labevents)} lab sonucu")

d_icd_diagnoses = pd.read_csv(DATA_PATH + "D_ICD_DIAGNOSES.csv")
print(f"✓ D_ICD_DIAGNOSES: {len(d_icd_diagnoses)} teşhis tanımı")

d_labitems = pd.read_csv(DATA_PATH + "D_LABITEMS.csv")
print(f"✓ D_LABITEMS: {len(d_labitems)} lab item tanımı")

chartevents = pd.read_csv(DATA_PATH + "CHARTEVENTS.csv")
print(f"✓ CHARTEVENTS: {len(chartevents)} chart eventi")

patients.columns = patients.columns.str.upper()
admissions.columns = admissions.columns.str.upper()
icustays.columns = icustays.columns.str.upper()
diagnoses.columns = diagnoses.columns.str.upper()
labevents.columns = labevents.columns.str.upper()
d_icd_diagnoses.columns = d_icd_diagnoses.columns.str.upper()
d_labitems.columns = d_labitems.columns.str.upper()
chartevents.columns = chartevents.columns.str.upper()

print("\n" + "=" * 60)
print("BÖLÜM 2: TABLO YAPILARI")
print("=" * 60)

print("\n📋 PATIENTS Tablosu:")
print(patients.head())
print(f"\nSütunlar: {list(patients.columns)}")

print("\n📋 ADMISSIONS Tablosu:")
print(admissions.head())
print(f"\nSütunlar: {list(admissions.columns)}")

print("\n📋 ICUSTAYS Tablosu:")
print(icustays.head())
print(f"\nSütunlar: {list(icustays.columns)}")

print("\n📋 LABEVENTS Tablosu:")
print(labevents.head())
print(f"\nSütunlar: {list(labevents.columns)}")

print("\n📋 CHARTEVENTS Tablosu:")
print(chartevents.head())
print(f"\nSütunlar: {list(chartevents.columns)}")

print("\n📋 DIAGNOSES_ICD Tablosu:")
print(diagnoses.head())
print(f"\nSütunlar: {list(diagnoses.columns)}")

print("\n" + "=" * 60)
print("BÖLÜM 3: TEMEL İSTATİSTİKLER")
print("=" * 60)

print("\n👥 Cinsiyet Dağılımı:")
print(patients['GENDER'].value_counts())

print("\n🏥 Yatış Türleri:")
print(admissions['ADMISSION_TYPE'].value_counts())

print("\n📤 Taburcu Yeri:")
print(admissions['DISCHARGE_LOCATION'].value_counts())

print("\n⚠️ Hastane İçi Mortalite:")
print(admissions['HOSPITAL_EXPIRE_FLAG'].value_counts())
mortality_rate = admissions['HOSPITAL_EXPIRE_FLAG'].mean() * 100
print(f"Mortalite Oranı: {mortality_rate:.2f}%")

print("\n" + "=" * 60)
print("BÖLÜM 4: VERİ BİRLEŞTİRME")
print("=" * 60)

merged_data = patients.merge(admissions, on='SUBJECT_ID', how='inner')
merged_data = merged_data.merge(icustays, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

print(f"\n✓ Birleştirilmiş veri boyutu: {merged_data.shape}")
print(f"  - Satır sayısı: {len(merged_data)}")
print(f"  - Sütun sayısı: {len(merged_data.columns)}")

print("\n📊 Birleştirilmiş Veri Sütunları:")
for i, col in enumerate(merged_data.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "=" * 60)
print("BÖLÜM 5: ÖZELLİK MÜHENDİSLİĞİ")
print("=" * 60)

date_columns = ['DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'INTIME', 'OUTTIME']
for col in date_columns:
    if col in merged_data.columns:
        merged_data[col] = pd.to_datetime(merged_data[col], errors='coerce')

merged_data['AGE'] = merged_data['ADMITTIME'].dt.year - merged_data['DOB'].dt.year

# MIMIC'te 89 yaş üstü hastalar gizlilik için 300+ yaş olarak gösterilir
merged_data.loc[merged_data['AGE'] > 100, 'AGE'] = 90
merged_data.loc[merged_data['AGE'] < 0, 'AGE'] = 0

print(f"✓ Yaş hesaplandı")
print(f"  Ortalama yaş: {merged_data['AGE'].mean():.1f}")
print(f"  Min yaş: {merged_data['AGE'].min():.1f}")
print(f"  Max yaş: {merged_data['AGE'].max():.1f}")

merged_data['LOS_HOSPITAL'] = (merged_data['DISCHTIME'] - merged_data['ADMITTIME']).dt.total_seconds() / 86400
print(f"\n✓ Hastane yatış süresi (gün) hesaplandı")
print(f"  Ortalama: {merged_data['LOS_HOSPITAL'].mean():.1f} gün")

merged_data['LOS_ICU'] = (merged_data['OUTTIME'] - merged_data['INTIME']).dt.total_seconds() / 86400
print(f"\n✓ ICU yatış süresi (gün) hesaplandı")
print(f"  Ortalama: {merged_data['LOS_ICU'].mean():.1f} gün")

merged_data['GENDER_NUM'] = (merged_data['GENDER'] == 'M').astype(int)

print("\n" + "=" * 60)
print("BÖLÜM 6: DEEP LEARNING HAZIRLIKLARI")
print("=" * 60)

print("""
🎯 DEEP LEARNING PROBLEMLERİ (Seçenekler):

1. MORTALİTE TAHMİNİ (Binary Classification)
   - Hedef: HOSPITAL_EXPIRE_FLAG (0/1)
   - Hasta hastanede ölecek mi?

2. YATIŞ SÜRESİ TAHMİNİ (Regression)
   - Hedef: LOS_HOSPITAL veya LOS_ICU
   - Hasta kaç gün yatacak?

3. YENİDEN YATIŞ TAHMİNİ (Binary Classification)
   - Hedef: 30 gün içinde tekrar yatış

4. TEŞHİS TAHMİNİ (Multi-label Classification)
   - Hedef: ICD kodları
   - Hangi teşhisler konulacak?

Şimdilik en yaygın problem olan MORTALİTE TAHMİNİ üzerine çalışalım.
""")

features_for_model = ['AGE', 'GENDER_NUM', 'LOS_ICU']
target = 'HOSPITAL_EXPIRE_FLAG'

print("\n📊 Eksik Değer Analizi:")
print(merged_data[features_for_model + [target]].isnull().sum())

model_data = merged_data[features_for_model + [target]].dropna()
print(f"\n✓ Model için hazır veri boyutu: {model_data.shape}")

print("\n" + "=" * 60)
print("BÖLÜM 7: GÖRSELLEŞTİRME")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(merged_data['AGE'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Yaş Dağılımı')
axes[0, 0].set_xlabel('Yaş')
axes[0, 0].set_ylabel('Frekans')

merged_data.groupby(pd.cut(merged_data['AGE'], bins=10))['HOSPITAL_EXPIRE_FLAG'].mean().plot(
    kind='bar', ax=axes[0, 1], color='coral', edgecolor='black'
)
axes[0, 1].set_title('Yaş Gruplarına Göre Mortalite Oranı')
axes[0, 1].set_xlabel('Yaş Grubu')
axes[0, 1].set_ylabel('Mortalite Oranı')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].hist(merged_data['LOS_ICU'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_title('ICU Kalış Süresi Dağılımı')
axes[1, 0].set_xlabel('Gün')
axes[1, 0].set_ylabel('Frekans')

mortality_by_admission = merged_data.groupby('ADMISSION_TYPE')['HOSPITAL_EXPIRE_FLAG'].mean()
mortality_by_admission.plot(kind='bar', ax=axes[1, 1], color='purple', edgecolor='black')
axes[1, 1].set_title('Yatış Türüne Göre Mortalite')
axes[1, 1].set_xlabel('Yatış Türü')
axes[1, 1].set_ylabel('Mortalite Oranı')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('mimic_eda_plots.png', dpi=150)
print("✓ Grafikler 'mimic_eda_plots.png' olarak kaydedildi")
plt.show()

print("\n" + "=" * 60)
print("BÖLÜM 8: BASİT DEEP LEARNING MODELİ (PyTorch)")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    print("✓ PyTorch ve sklearn yüklü")

    X = model_data[features_for_model].values
    y = model_data[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    class MortalityPredictor(nn.Module):
        def __init__(self, input_size):
            super(MortalityPredictor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    model = MortalityPredictor(input_size=len(features_for_model))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n📊 Model Mimarisi:")
    print(model)

    print("\n🚀 Model Eğitimi Başlıyor...")
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    print("\n📈 Model Değerlendirmesi:")
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))

except ImportError as e:
    print(f"⚠️ Gerekli kütüphaneler eksik: {e}")
    print("\nKurulum için terminalde şu komutları çalıştırın:")
    print("  pip install torch scikit-learn")

print("\n" + "=" * 60)
print("SONRAKİ ADIMLAR")
print("=" * 60)

print("""
🎯 Model geliştirmek için yapabileceklerin:

1. DAHA FAZLA ÖZELLİK EKLE:
   - Lab sonuçları (LABEVENTS)
   - Vital bulgular (CHARTEVENTS)
   - Teşhis kodları (DIAGNOSES_ICD)
   - İlaçlar (PRESCRIPTIONS)

2. MODEL MİMARİSİNİ GELİŞTİR:
   - LSTM/GRU: Zaman serisi veriler için
   - Transformer: Dikkat mekanizması ile
   - CNN: Özellik çıkarımı için

3. İLERİ TEKNİKLER:
   - Class imbalance için SMOTE
   - Cross-validation
   - Hyperparameter tuning
   - Ensemble methods

4. EK ANALİZLER:
   - SHAP/LIME ile model yorumlama
   - Feature importance analizi
""")

model_data.to_csv('prepared_data.csv', index=False)
print("✓ Hazırlanan veri 'prepared_data.csv' olarak kaydedildi")
