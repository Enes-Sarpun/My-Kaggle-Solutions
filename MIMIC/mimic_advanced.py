import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "mimic-iii-clinical-database-demo-1.4/"

print("=" * 70)
print("MIMIC-III GELİŞMİŞ DEEP LEARNING MODELİ")
print("=" * 70)

print("\n📊 Tablolar yükleniyor...")

patients = pd.read_csv(DATA_PATH + "PATIENTS.csv")
admissions = pd.read_csv(DATA_PATH + "ADMISSIONS.csv")
icustays = pd.read_csv(DATA_PATH + "ICUSTAYS.csv")
labevents = pd.read_csv(DATA_PATH + "LABEVENTS.csv")
diagnoses = pd.read_csv(DATA_PATH + "DIAGNOSES_ICD.csv")
d_labitems = pd.read_csv(DATA_PATH + "D_LABITEMS.csv")
d_icd = pd.read_csv(DATA_PATH + "D_ICD_DIAGNOSES.csv")

for df in [patients, admissions, icustays, labevents, diagnoses, d_labitems, d_icd]:
    df.columns = df.columns.str.upper()

print(f"✓ PATIENTS: {len(patients)} hasta")
print(f"✓ ADMISSIONS: {len(admissions)} yatış")
print(f"✓ ICUSTAYS: {len(icustays)} ICU kaydı")
print(f"✓ LABEVENTS: {len(labevents)} lab sonucu")
print(f"✓ DIAGNOSES: {len(diagnoses)} teşhis")

print("\n" + "=" * 70)
print("BÖLÜM 2: TEMEL VERİ SETİ OLUŞTURMA")
print("=" * 70)

base_data = patients.merge(admissions, on='SUBJECT_ID', how='inner')
base_data = base_data.merge(icustays, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

base_data['DOB'] = pd.to_datetime(base_data['DOB'])
base_data['ADMITTIME'] = pd.to_datetime(base_data['ADMITTIME'])
base_data['DISCHTIME'] = pd.to_datetime(base_data['DISCHTIME'])
base_data['INTIME'] = pd.to_datetime(base_data['INTIME'])
base_data['OUTTIME'] = pd.to_datetime(base_data['OUTTIME'])

base_data['AGE'] = base_data['ADMITTIME'].dt.year - base_data['DOB'].dt.year
base_data.loc[base_data['AGE'] > 100, 'AGE'] = 90
base_data.loc[base_data['AGE'] < 0, 'AGE'] = 0

base_data['GENDER_NUM'] = (base_data['GENDER'] == 'M').astype(int)
base_data['LOS_ICU'] = (base_data['OUTTIME'] - base_data['INTIME']).dt.total_seconds() / 86400
base_data['LOS_HOSPITAL'] = (base_data['DISCHTIME'] - base_data['ADMITTIME']).dt.total_seconds() / 86400

print(f"✓ Temel veri seti: {base_data.shape}")

print("\n" + "=" * 70)
print("BÖLÜM 3: LAB SONUÇLARINI ENTEGRE ETME")
print("=" * 70)

print("\n📋 En sık kullanılan lab testleri:")
lab_counts = labevents.groupby('ITEMID').size().reset_index(name='COUNT')
lab_counts = lab_counts.merge(d_labitems[['ITEMID', 'LABEL']], on='ITEMID')
lab_counts = lab_counts.sort_values('COUNT', ascending=False)
print(lab_counts.head(15))

IMPORTANT_LABS = {
    50912: 'CREATININE',
    50971: 'POTASSIUM',
    50983: 'SODIUM',
    50902: 'CHLORIDE',
    50882: 'BICARBONATE',
    51221: 'HEMATOCRIT',
    51222: 'HEMOGLOBIN',
    51265: 'PLATELET',
    51301: 'WBC',
    50931: 'GLUCOSE',
    50960: 'MAGNESIUM',
    50893: 'CALCIUM',
    51006: 'BUN',
    50813: 'LACTATE',
    50820: 'PH',
}

print(f"\n✓ {len(IMPORTANT_LABS)} önemli lab testi seçildi")

lab_filtered = labevents[labevents['ITEMID'].isin(IMPORTANT_LABS.keys())].copy()
lab_filtered['LAB_NAME'] = lab_filtered['ITEMID'].map(IMPORTANT_LABS)

print(f"✓ Filtrelenmiş lab kayıtları: {len(lab_filtered)}")

print("\n🔄 Her hasta için lab istatistikleri hesaplanıyor...")

def calculate_lab_features(group):
    result = {}
    for lab_name in IMPORTANT_LABS.values():
        lab_data = group[group['LAB_NAME'] == lab_name]['VALUENUM']
        if len(lab_data) > 0:
            result[f'{lab_name}_MEAN'] = lab_data.mean()
            result[f'{lab_name}_MIN'] = lab_data.min()
            result[f'{lab_name}_MAX'] = lab_data.max()
            result[f'{lab_name}_STD'] = lab_data.std() if len(lab_data) > 1 else 0
            result[f'{lab_name}_COUNT'] = len(lab_data)
        else:
            result[f'{lab_name}_MEAN'] = np.nan
            result[f'{lab_name}_MIN'] = np.nan
            result[f'{lab_name}_MAX'] = np.nan
            result[f'{lab_name}_STD'] = np.nan
            result[f'{lab_name}_COUNT'] = 0
    return pd.Series(result)

lab_features = lab_filtered.groupby('HADM_ID').apply(calculate_lab_features).reset_index()

print(f"✓ Lab özellikleri hesaplandı: {lab_features.shape}")
print(f"  - {len(lab_features)} yatış için")
print(f"  - {len(lab_features.columns)-1} özellik")

print("\n" + "=" * 70)
print("BÖLÜM 4: TEŞHİSLERİ ENTEGRE ETME")
print("=" * 70)

print("\n📋 En sık konulan teşhisler:")
diag_counts = diagnoses.groupby('ICD9_CODE').size().reset_index(name='COUNT')
diag_counts = diag_counts.merge(d_icd[['ICD9_CODE', 'SHORT_TITLE']], on='ICD9_CODE', how='left')
diag_counts = diag_counts.sort_values('COUNT', ascending=False)
print(diag_counts.head(15))

diagnoses['ICD9_GROUP'] = diagnoses['ICD9_CODE'].astype(str).str[:3]

def calculate_diagnosis_features(group):
    result = {
        'TOTAL_DIAGNOSES': len(group),
        'UNIQUE_DIAGNOSES': group['ICD9_CODE'].nunique(),
    }
    important_groups = {
        'HAS_HEART_ATTACK': '410',
        'HAS_HEART_FAILURE': '428',
        'HAS_KIDNEY_FAILURE': '584',
        'HAS_RESPIRATORY_FAILURE': '518',
        'HAS_SEPSIS': '038',
        'HAS_DIABETES': '250',
        'HAS_HYPERTENSION': '401',
    }
    for name, code in important_groups.items():
        result[name] = int(code in group['ICD9_GROUP'].values)
    return pd.Series(result)

diagnosis_features = diagnoses.groupby('HADM_ID').apply(calculate_diagnosis_features).reset_index()

print(f"\n✓ Teşhis özellikleri hesaplandı: {diagnosis_features.shape}")
print(f"  Özellikler: {list(diagnosis_features.columns[1:])}")

print("\n" + "=" * 70)
print("BÖLÜM 5: TÜM ÖZELLİKLERİ BİRLEŞTİRME")
print("=" * 70)

final_data = base_data.merge(lab_features, on='HADM_ID', how='left')
final_data = final_data.merge(diagnosis_features, on='HADM_ID', how='left')

print(f"✓ Final veri seti boyutu: {final_data.shape}")

TARGET = 'HOSPITAL_EXPIRE_FLAG'

DEMOGRAPHIC_FEATURES = ['AGE', 'GENDER_NUM', 'LOS_ICU']

LAB_FEATURES = [col for col in final_data.columns if any(
    lab in col for lab in ['_MEAN', '_MIN', '_MAX']
)]

DIAGNOSIS_FEATURES = ['TOTAL_DIAGNOSES', 'UNIQUE_DIAGNOSES',
                      'HAS_HEART_ATTACK', 'HAS_HEART_FAILURE',
                      'HAS_KIDNEY_FAILURE', 'HAS_RESPIRATORY_FAILURE',
                      'HAS_SEPSIS', 'HAS_DIABETES', 'HAS_HYPERTENSION']

ALL_FEATURES = DEMOGRAPHIC_FEATURES + LAB_FEATURES + DIAGNOSIS_FEATURES

print(f"\n📊 Özellik Özeti:")
print(f"  - Demografik: {len(DEMOGRAPHIC_FEATURES)} özellik")
print(f"  - Lab: {len(LAB_FEATURES)} özellik")
print(f"  - Teşhis: {len(DIAGNOSIS_FEATURES)} özellik")
print(f"  - TOPLAM: {len(ALL_FEATURES)} özellik")

print(f"\n📊 Eksik Değer Analizi:")
missing = final_data[ALL_FEATURES + [TARGET]].isnull().sum()
missing_pct = (missing / len(final_data) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Percent', ascending=False)
if len(missing_df) > 0:
    print(missing_df.head(20))
else:
    print("✓ Eksik değer yok!")

print("\n" + "=" * 70)
print("BÖLÜM 6: MODEL İÇİN VERİ HAZIRLAMA")
print("=" * 70)

model_df = final_data[ALL_FEATURES + [TARGET]].copy()

for col in ALL_FEATURES:
    if model_df[col].isnull().any():
        median_val = model_df[col].median()
        if pd.isna(median_val):
            median_val = 0
        model_df[col] = model_df[col].fillna(median_val)

model_df = model_df.dropna(subset=[TARGET])

print(f"✓ Model verisi hazır: {model_df.shape}")
print(f"  - Örnekler: {len(model_df)}")
print(f"  - Özellikler: {len(ALL_FEATURES)}")

print(f"\n📊 Hedef Değişken Dağılımı:")
print(model_df[TARGET].value_counts())
print(f"Mortalite Oranı: {model_df[TARGET].mean()*100:.2f}%")

print("\n" + "=" * 70)
print("BÖLÜM 7: DEEP LEARNING MODELİ")
print("=" * 70)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                            roc_auc_score, confusion_matrix, f1_score)

X = model_df[ALL_FEATURES].values.astype(np.float32)
y = model_df[TARGET].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train seti: {len(X_train)} örnek")
print(f"✓ Test seti: {len(X_test)} örnek")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)])
print(f"✓ Pozitif sınıf ağırlığı: {pos_weight.item():.2f}")

class AdvancedMortalityPredictor(nn.Module):
    def __init__(self, input_size):
        super(AdvancedMortalityPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = AdvancedMortalityPredictor(input_size=len(ALL_FEATURES))
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

print(f"\n📊 Model Mimarisi:")
print(model)
print(f"\nToplam parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")

print("\n🚀 Model Eğitimi Başlıyor...")
epochs = 100
best_auc = 0
train_losses = []
val_aucs = []

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

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_proba = torch.sigmoid(val_outputs).numpy()
        val_auc = roc_auc_score(y_test, val_proba)
        val_aucs.append(val_auc)

    scheduler.step(val_auc)

    if val_auc > best_auc:
        best_auc = val_auc
        best_model_state = model.state_dict().copy()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

model.load_state_dict(best_model_state)

print("\n" + "=" * 70)
print("BÖLÜM 8: MODEL DEĞERLENDİRME")
print("=" * 70)

model.eval()
with torch.no_grad():
    y_pred_proba = torch.sigmoid(model(X_test_tensor)).numpy()
    y_pred = (y_pred_proba > 0.5).astype(int)

print(f"\n📈 Test Sonuçları:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Negative:  {cm[0,0]:3d}  |  False Positive: {cm[0,1]:3d}")
print(f"  False Negative: {cm[1,0]:3d}  |  True Positive:  {cm[1,1]:3d}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))

print("\n" + "=" * 70)
print("BÖLÜM 9: ÖZELLİK ÖNEMİ ANALİZİ")
print("=" * 70)

print("\n🔍 Özellik önemi hesaplanıyor...")

first_layer_weights = model.network[0].weight.data.abs().mean(dim=0).numpy()

importance_df = pd.DataFrame({
    'Feature': ALL_FEATURES,
    'Importance': first_layer_weights
}).sort_values('Importance', ascending=False)

print("\n🏆 En Önemli 15 Özellik (Ağırlık Bazlı):")
print(importance_df.head(15).to_string(index=False))

print("\n" + "=" * 70)
print("BÖLÜM 10: GÖRSELLEŞTİRME")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
axes[0, 0].set_title('Eğitim Kaybı')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

axes[0, 1].plot(val_aucs, label='Validation AUC', color='green')
axes[0, 1].axhline(y=best_auc, color='r', linestyle='--', label=f'Best AUC: {best_auc:.4f}')
axes[0, 1].set_title('Validation ROC-AUC')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Survived', 'Died'],
            yticklabels=['Survived', 'Died'])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

top_features = importance_df.head(10)
axes[1, 1].barh(range(len(top_features)), top_features['Importance'].values)
axes[1, 1].set_yticks(range(len(top_features)))
axes[1, 1].set_yticklabels(top_features['Feature'].values)
axes[1, 1].set_title('Top 10 Özellik Önemi')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('mimic_advanced_results.png', dpi=150)
print("✓ Grafikler 'mimic_advanced_results.png' olarak kaydedildi")
plt.show()

print("\n" + "=" * 70)
print("ÖZET")
print("=" * 70)

print(f"""
📊 VERİ:
   - {len(model_df)} hasta kaydı
   - {len(ALL_FEATURES)} özellik kullanıldı
   - Demografik: {len(DEMOGRAPHIC_FEATURES)}
   - Lab Sonuçları: {len(LAB_FEATURES)}
   - Teşhis: {len(DIAGNOSIS_FEATURES)}

🎯 MODEL PERFORMANSI:
   - Best ROC-AUC: {best_auc:.4f}
   - Test Accuracy: {accuracy_score(y_test, y_pred):.4f}
   - F1 Score: {f1_score(y_test, y_pred):.4f}

🏆 EN ÖNEMLİ ÖZELLİKLER:
{importance_df.head(5).to_string(index=False)}

💡 GELİŞTİRME ÖNERİLERİ:
   1. Daha fazla lab testi eklenebilir
   2. Vital bulgular (CHARTEVENTS) eklenebilir
   3. LSTM/Transformer ile zaman serisi analizi yapılabilir
   4. Cross-validation eklenebilir
   5. Hyperparameter tuning yapılabilir
""")

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'features': ALL_FEATURES,
    'best_auc': best_auc
}, 'mortality_model.pth')
print("✓ Model 'mortality_model.pth' olarak kaydedildi")

model_df.to_csv('final_model_data.csv', index=False)
print("✓ Veri 'final_model_data.csv' olarak kaydedildi")
