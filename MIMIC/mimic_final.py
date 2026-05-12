import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, classification_report,
                            roc_auc_score, confusion_matrix, f1_score,
                            precision_score, recall_score)

DATA_PATH = "mimic-iii-clinical-database-demo-1.4/"

print("=" * 70)
print("MIMIC-III FINAL DEEP LEARNING MODELİ")
print("=" * 70)

print("\n📊 Tablolar yükleniyor...")

patients = pd.read_csv(DATA_PATH + "PATIENTS.csv")
admissions = pd.read_csv(DATA_PATH + "ADMISSIONS.csv")
icustays = pd.read_csv(DATA_PATH + "ICUSTAYS.csv")
labevents = pd.read_csv(DATA_PATH + "LABEVENTS.csv")
diagnoses = pd.read_csv(DATA_PATH + "DIAGNOSES_ICD.csv")
chartevents = pd.read_csv(DATA_PATH + "CHARTEVENTS.csv")
d_labitems = pd.read_csv(DATA_PATH + "D_LABITEMS.csv")
d_items = pd.read_csv(DATA_PATH + "D_ITEMS.csv")

for df in [patients, admissions, icustays, labevents, diagnoses,
           chartevents, d_labitems, d_items]:
    df.columns = df.columns.str.upper()

print(f"✓ PATIENTS: {len(patients)} hasta")
print(f"✓ ADMISSIONS: {len(admissions)} yatış")
print(f"✓ ICUSTAYS: {len(icustays)} ICU kaydı")
print(f"✓ LABEVENTS: {len(labevents)} lab sonucu")
print(f"✓ CHARTEVENTS: {len(chartevents)} vital bulgu")
print(f"✓ DIAGNOSES: {len(diagnoses)} teşhis")

print("\n" + "=" * 70)
print("BÖLÜM 2: TEMEL VERİ SETİ")
print("=" * 70)

base_data = patients.merge(admissions, on='SUBJECT_ID', how='inner')
base_data = base_data.merge(icustays, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

for col in ['DOB', 'ADMITTIME', 'DISCHTIME', 'INTIME', 'OUTTIME']:
    base_data[col] = pd.to_datetime(base_data[col], errors='coerce')

base_data['AGE'] = base_data['ADMITTIME'].dt.year - base_data['DOB'].dt.year
base_data.loc[base_data['AGE'] > 100, 'AGE'] = 90
base_data.loc[base_data['AGE'] < 0, 'AGE'] = 0

base_data['GENDER_NUM'] = (base_data['GENDER'] == 'M').astype(int)
base_data['LOS_ICU'] = (base_data['OUTTIME'] - base_data['INTIME']).dt.total_seconds() / 86400
base_data['LOS_HOSPITAL'] = (base_data['DISCHTIME'] - base_data['ADMITTIME']).dt.total_seconds() / 86400
base_data['IS_EMERGENCY'] = (base_data['ADMISSION_TYPE'] == 'EMERGENCY').astype(int)

print(f"✓ Temel veri seti: {base_data.shape}")

print("\n" + "=" * 70)
print("BÖLÜM 3: LAB ÖZELLİKLERİ")
print("=" * 70)

IMPORTANT_LABS = {
    50912: 'CREATININE',
    50971: 'POTASSIUM',
    50983: 'SODIUM',
    50882: 'BICARBONATE',
    51221: 'HEMATOCRIT',
    51222: 'HEMOGLOBIN',
    51265: 'PLATELET',
    51301: 'WBC',
    50931: 'GLUCOSE',
    51006: 'BUN',
    50813: 'LACTATE',
}

lab_filtered = labevents[labevents['ITEMID'].isin(IMPORTANT_LABS.keys())].copy()
lab_filtered['LAB_NAME'] = lab_filtered['ITEMID'].map(IMPORTANT_LABS)

print(f"✓ {len(IMPORTANT_LABS)} önemli lab testi seçildi")

def calc_lab_stats(group):
    result = {}
    for lab in IMPORTANT_LABS.values():
        data = group[group['LAB_NAME'] == lab]['VALUENUM'].dropna()
        if len(data) > 0:
            result[f'{lab}_MEAN'] = data.mean()
            result[f'{lab}_MIN'] = data.min()
            result[f'{lab}_MAX'] = data.max()
        else:
            result[f'{lab}_MEAN'] = np.nan
            result[f'{lab}_MIN'] = np.nan
            result[f'{lab}_MAX'] = np.nan
    return pd.Series(result)

lab_features = lab_filtered.groupby('HADM_ID').apply(calc_lab_stats).reset_index()
print(f"✓ Lab özellikleri: {lab_features.shape}")

print("\n" + "=" * 70)
print("BÖLÜM 4: VİTAL BULGULAR")
print("=" * 70)

VITAL_SIGNS = {
    211: 'HEART_RATE', 220045: 'HEART_RATE',
    51: 'SBP', 220050: 'SBP', 220179: 'SBP',
    8368: 'DBP', 220051: 'DBP', 220180: 'DBP',
    52: 'MAP', 220052: 'MAP', 220181: 'MAP',
    618: 'RESP_RATE', 220210: 'RESP_RATE',
    646: 'SPO2', 220277: 'SPO2',
    223761: 'TEMPERATURE', 678: 'TEMPERATURE',
    198: 'GCS', 220739: 'GCS',
}

chart_filtered = chartevents[chartevents['ITEMID'].isin(VITAL_SIGNS.keys())].copy()
chart_filtered['VITAL_NAME'] = chart_filtered['ITEMID'].map(VITAL_SIGNS)

print(f"✓ Filtrelenmiş vital kayıtları: {len(chart_filtered)}")

def calc_vital_stats(group):
    result = {}
    for vital in set(VITAL_SIGNS.values()):
        data = group[group['VITAL_NAME'] == vital]['VALUENUM'].dropna()
        if len(data) > 0:
            result[f'{vital}_MEAN'] = data.mean()
            result[f'{vital}_MIN'] = data.min()
            result[f'{vital}_MAX'] = data.max()
        else:
            result[f'{vital}_MEAN'] = np.nan
            result[f'{vital}_MIN'] = np.nan
            result[f'{vital}_MAX'] = np.nan
    return pd.Series(result)

if len(chart_filtered) > 0:
    vital_features = chart_filtered.groupby('HADM_ID').apply(calc_vital_stats).reset_index()
    print(f"✓ Vital özellikleri: {vital_features.shape}")
else:
    chart_filtered = chartevents[chartevents['ITEMID'].isin(VITAL_SIGNS.keys())].copy()
    chart_filtered['VITAL_NAME'] = chart_filtered['ITEMID'].map(VITAL_SIGNS)
    vital_features = chart_filtered.groupby('ICUSTAY_ID').apply(calc_vital_stats).reset_index()
    print(f"✓ Vital özellikleri (ICUSTAY bazlı): {vital_features.shape}")

print("\n" + "=" * 70)
print("BÖLÜM 5: TEŞHİS ÖZELLİKLERİ")
print("=" * 70)

diagnoses['ICD9_GROUP'] = diagnoses['ICD9_CODE'].astype(str).str[:3]

def calc_diag_features(group):
    result = {
        'TOTAL_DIAGNOSES': len(group),
        'UNIQUE_DIAGNOSES': group['ICD9_CODE'].nunique(),
    }
    important = {
        'HAS_HEART_FAILURE': '428',
        'HAS_KIDNEY_FAILURE': '584',
        'HAS_RESPIRATORY_FAILURE': '518',
        'HAS_SEPSIS': '038',
        'HAS_DIABETES': '250',
        'HAS_HYPERTENSION': '401',
    }
    for name, code in important.items():
        result[name] = int(code in group['ICD9_GROUP'].values)
    return pd.Series(result)

diag_features = diagnoses.groupby('HADM_ID').apply(calc_diag_features).reset_index()
print(f"✓ Teşhis özellikleri: {diag_features.shape}")

print("\n" + "=" * 70)
print("BÖLÜM 6: VERİ BİRLEŞTİRME")
print("=" * 70)

final_data = base_data.merge(lab_features, on='HADM_ID', how='left')
final_data = final_data.merge(diag_features, on='HADM_ID', how='left')

if 'HADM_ID' in vital_features.columns:
    final_data = final_data.merge(vital_features, on='HADM_ID', how='left')
elif 'ICUSTAY_ID' in vital_features.columns:
    final_data = final_data.merge(vital_features, on='ICUSTAY_ID', how='left')

print(f"✓ Birleştirilmiş veri: {final_data.shape}")

TARGET = 'HOSPITAL_EXPIRE_FLAG'

DEMO_FEATURES = ['AGE', 'GENDER_NUM', 'LOS_ICU', 'IS_EMERGENCY']

LAB_FEATURES = [col for col in final_data.columns if any(
    lab in col for lab in ['CREATININE', 'POTASSIUM', 'SODIUM', 'BICARBONATE',
                           'HEMATOCRIT', 'HEMOGLOBIN', 'PLATELET', 'WBC',
                           'GLUCOSE', 'BUN', 'LACTATE']
) and ('_MEAN' in col or '_MIN' in col or '_MAX' in col)]

VITAL_FEATURES = [col for col in final_data.columns if any(
    vital in col for vital in ['HEART_RATE', 'SBP', 'DBP', 'MAP',
                               'RESP_RATE', 'SPO2', 'TEMPERATURE', 'GCS']
) and ('_MEAN' in col or '_MIN' in col or '_MAX' in col)]

DIAG_FEATURES = ['TOTAL_DIAGNOSES', 'UNIQUE_DIAGNOSES', 'HAS_HEART_FAILURE',
                 'HAS_KIDNEY_FAILURE', 'HAS_RESPIRATORY_FAILURE',
                 'HAS_SEPSIS', 'HAS_DIABETES', 'HAS_HYPERTENSION']

ALL_FEATURES = DEMO_FEATURES + LAB_FEATURES + VITAL_FEATURES + DIAG_FEATURES
ALL_FEATURES = [f for f in ALL_FEATURES if f in final_data.columns]

print(f"\n📊 Özellik Özeti:")
print(f"  - Demografik: {len([f for f in DEMO_FEATURES if f in ALL_FEATURES])}")
print(f"  - Lab: {len([f for f in LAB_FEATURES if f in ALL_FEATURES])}")
print(f"  - Vital: {len([f for f in VITAL_FEATURES if f in ALL_FEATURES])}")
print(f"  - Teşhis: {len([f for f in DIAG_FEATURES if f in ALL_FEATURES])}")
print(f"  - TOPLAM: {len(ALL_FEATURES)}")

print("\n" + "=" * 70)
print("BÖLÜM 7: VERİ HAZIRLAMA")
print("=" * 70)

model_df = final_data[ALL_FEATURES + [TARGET]].copy()

for col in ALL_FEATURES:
    if model_df[col].isnull().any():
        median_val = model_df[col].median()
        model_df[col] = model_df[col].fillna(median_val if pd.notna(median_val) else 0)

model_df = model_df.dropna(subset=[TARGET])

print(f"✓ Model verisi: {model_df.shape}")
print(f"\n📊 Hedef Dağılımı:")
print(model_df[TARGET].value_counts())

print("\n" + "=" * 70)
print("BÖLÜM 8: ÖZELLİK SEÇİMİ")
print("=" * 70)

X = model_df[ALL_FEATURES].values.astype(np.float32)
y = model_df[TARGET].values.astype(np.float32)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

k_features = min(20, len(ALL_FEATURES), len(model_df) // 5)
print(f"✓ En iyi {k_features} özellik seçiliyor...")

selector = SelectKBest(f_classif, k=k_features)
X_selected = selector.fit_transform(X, y)

selected_mask = selector.get_support()
SELECTED_FEATURES = [f for f, s in zip(ALL_FEATURES, selected_mask) if s]

print(f"\n🏆 Seçilen Özellikler:")
for i, feat in enumerate(SELECTED_FEATURES, 1):
    print(f"  {i}. {feat}")

print("\n" + "=" * 70)
print("BÖLÜM 9: MODEL")
print("=" * 70)

class SimpleMortalityModel(nn.Module):
    """Basit ama etkili model - overfitting'i önlemek için"""
    def __init__(self, input_size):
        super(SimpleMortalityModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

print("\n" + "=" * 70)
print("BÖLÜM 10: 5-FOLD CROSS-VALIDATION")
print("=" * 70)

N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 15

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = {
    'accuracy': [], 'auc': [], 'f1': [],
    'precision': [], 'recall': []
}

all_y_true = []
all_y_pred = []
all_y_proba = []

print(f"\n🚀 {N_FOLDS}-Fold Cross-Validation başlıyor...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
    print(f"\n{'='*40}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'='*40}")

    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleMortalityModel(input_size=k_features)

    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(sum(y_train), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(model(X_val_t)).numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_pred_proba)
    except:
        auc = 0.5
    f1 = f1_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)

    fold_results['accuracy'].append(acc)
    fold_results['auc'].append(auc)
    fold_results['f1'].append(f1)
    fold_results['precision'].append(prec)
    fold_results['recall'].append(rec)

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_pred_proba)

    print(f"\n  📈 Fold {fold} Sonuçları:")
    print(f"     Accuracy:  {acc:.4f}")
    print(f"     ROC-AUC:   {auc:.4f}")
    print(f"     F1 Score:  {f1:.4f}")
    print(f"     Precision: {prec:.4f}")
    print(f"     Recall:    {rec:.4f}")

print("\n" + "=" * 70)
print("BÖLÜM 11: CROSS-VALIDATION SONUÇLARI")
print("=" * 70)

print("\n📊 5-Fold CV Ortalamaları (± Std):")
print(f"  Accuracy:  {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
print(f"  ROC-AUC:   {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
print(f"  F1 Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
print(f"  Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
print(f"  Recall:    {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")

print("\n📊 Genel Confusion Matrix (Tüm Fold'lar):")
cm = confusion_matrix(all_y_true, all_y_pred)
print(f"  True Negative:  {cm[0,0]:3d}  |  False Positive: {cm[0,1]:3d}")
print(f"  False Negative: {cm[1,0]:3d}  |  True Positive:  {cm[1,1]:3d}")

print("\n📋 Genel Classification Report:")
print(classification_report(all_y_true, all_y_pred,
                          target_names=['Survived', 'Died'],
                          zero_division=0))

print("\n" + "=" * 70)
print("BÖLÜM 12: FİNAL MODEL EĞİTİMİ")
print("=" * 70)

print("\n🎯 Tüm veri ile final model eğitiliyor...")

final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_selected)

X_final_t = torch.FloatTensor(X_final_scaled)
y_final_t = torch.FloatTensor(y).unsqueeze(1)

final_ds = TensorDataset(X_final_t, y_final_t)
final_loader = DataLoader(final_ds, batch_size=BATCH_SIZE, shuffle=True)

final_model = SimpleMortalityModel(input_size=k_features)
pos_weight = torch.tensor([(len(y) - sum(y)) / max(sum(y), 1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

for epoch in range(EPOCHS):
    final_model.train()
    for batch_X, batch_y in final_loader:
        optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

print("✓ Final model eğitildi")

print("\n" + "=" * 70)
print("BÖLÜM 13: GÖRSELLEŞTİRME")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
means = [np.mean(fold_results['accuracy']), np.mean(fold_results['auc']),
         np.mean(fold_results['f1']), np.mean(fold_results['precision']),
         np.mean(fold_results['recall'])]
stds = [np.std(fold_results['accuracy']), np.std(fold_results['auc']),
        np.std(fold_results['f1']), np.std(fold_results['precision']),
        np.std(fold_results['recall'])]

x_pos = np.arange(len(metrics))
axes[0, 0].bar(x_pos, means, yerr=stds, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(metrics)
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('5-Fold CV Sonuçları (Mean ± Std)')
axes[0, 0].set_ylim(0, 1)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Survived', 'Died'],
            yticklabels=['Survived', 'Died'])
axes[0, 1].set_title('Confusion Matrix (Tüm Fold\'lar)')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

folds = range(1, N_FOLDS + 1)
axes[1, 0].bar(folds, fold_results['auc'], color='coral', edgecolor='black')
axes[1, 0].axhline(y=np.mean(fold_results['auc']), color='red', linestyle='--',
                   label=f'Mean: {np.mean(fold_results["auc"]):.3f}')
axes[1, 0].set_xlabel('Fold')
axes[1, 0].set_ylabel('ROC-AUC')
axes[1, 0].set_title('Her Fold için ROC-AUC')
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1)

selector_scores = selector.scores_
feature_importance = pd.DataFrame({
    'Feature': ALL_FEATURES,
    'Score': selector_scores
}).sort_values('Score', ascending=False).head(10)

axes[1, 1].barh(range(len(feature_importance)), feature_importance['Score'].values, color='green')
axes[1, 1].set_yticks(range(len(feature_importance)))
axes[1, 1].set_yticklabels(feature_importance['Feature'].values)
axes[1, 1].set_title('Top 10 Özellik (F-Score)')
axes[1, 1].set_xlabel('F-Score')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('mimic_final_results.png', dpi=150)
print("✓ Grafikler 'mimic_final_results.png' olarak kaydedildi")
plt.show()

print("\n" + "=" * 70)
print("ÖZET")
print("=" * 70)

print(f"""
📊 VERİ:
   - {len(model_df)} örnek
   - {k_features} özellik (seçilmiş)
   - Kaynak: Demografik + Lab + Vital + Teşhis

🎯 5-FOLD CROSS-VALIDATION SONUÇLARI:
   - Accuracy:  {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}
   - ROC-AUC:   {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}
   - F1 Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}
   - Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}
   - Recall:    {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}

🏆 EN ÖNEMLİ ÖZELLİKLER:
{feature_importance[['Feature', 'Score']].head(5).to_string(index=False)}

💡 NOT:
   - Cross-validation ile daha güvenilir sonuçlar elde edildi
   - Demo veri seti küçük olduğu için performans sınırlı
   - Gerçek MIMIC-III (~50.000 hasta) ile çok daha iyi sonuçlar alınabilir
""")

torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler': final_scaler,
    'selector': selector,
    'features': ALL_FEATURES,
    'selected_features': SELECTED_FEATURES,
    'cv_results': fold_results
}, 'final_mortality_model.pth')
print("✓ Model 'final_mortality_model.pth' olarak kaydedildi")

model_df.to_csv('final_model_data.csv', index=False)
print("✓ Veri 'final_model_data.csv' olarak kaydedildi")
