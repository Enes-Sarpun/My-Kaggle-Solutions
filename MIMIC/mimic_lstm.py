import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                            confusion_matrix, classification_report,
                            precision_score, recall_score)

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "mimic-iii-clinical-database-demo-1.4/"

print("=" * 70)
print("MIMIC-III - LSTM ZAMAN SERİSİ ANALİZİ")
print("=" * 70)

print("\n📊 Veriler yükleniyor...")

patients = pd.read_csv(DATA_PATH + "PATIENTS.csv")
admissions = pd.read_csv(DATA_PATH + "ADMISSIONS.csv")
icustays = pd.read_csv(DATA_PATH + "ICUSTAYS.csv")
chartevents = pd.read_csv(DATA_PATH + "CHARTEVENTS.csv")

for df in [patients, admissions, icustays, chartevents]:
    df.columns = df.columns.str.upper()

print(f"✓ PATIENTS: {len(patients)}")
print(f"✓ ADMISSIONS: {len(admissions)}")
print(f"✓ ICUSTAYS: {len(icustays)}")
print(f"✓ CHARTEVENTS: {len(chartevents)}")

print("\n" + "=" * 70)
print("BÖLÜM 2: HEDEF DEĞİŞKEN")
print("=" * 70)

icu_data = icustays.merge(admissions[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG']],
                          on='HADM_ID', how='inner')

icu_data['INTIME'] = pd.to_datetime(icu_data['INTIME'])
icu_data['OUTTIME'] = pd.to_datetime(icu_data['OUTTIME'])

print(f"✓ ICU kayıtları: {len(icu_data)}")
print(f"✓ Mortalite dağılımı:")
print(icu_data['HOSPITAL_EXPIRE_FLAG'].value_counts())

print("\n" + "=" * 70)
print("BÖLÜM 3: ZAMAN SERİSİ HAZIRLAMA")
print("=" * 70)

VITAL_ITEMS = {
    211: 'HR', 220045: 'HR',
    51: 'SBP', 220050: 'SBP', 220179: 'SBP',
    8368: 'DBP', 220051: 'DBP', 220180: 'DBP',
    52: 'MAP', 220052: 'MAP', 220181: 'MAP',
    618: 'RR', 220210: 'RR',
    646: 'SPO2', 220277: 'SPO2',
    223761: 'TEMP', 678: 'TEMP', 676: 'TEMP',
}

print("\n🔄 Vital bulgular filtreleniyor...")
chart_vitals = chartevents[chartevents['ITEMID'].isin(VITAL_ITEMS.keys())].copy()
chart_vitals['VITAL_NAME'] = chart_vitals['ITEMID'].map(VITAL_ITEMS)
chart_vitals['CHARTTIME'] = pd.to_datetime(chart_vitals['CHARTTIME'])

print(f"✓ Filtrelenmiş vital kayıtları: {len(chart_vitals)}")

print("\n📊 Vital bulgu dağılımı:")
print(chart_vitals['VITAL_NAME'].value_counts())

print("\n" + "=" * 70)
print("BÖLÜM 4: SAATLİK PENCERELER")
print("=" * 70)

def create_hourly_sequences(icustay_id, vital_data, icu_info, max_hours=48):
    vitals = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']
    num_vitals = len(vitals)
    sequence = np.full((max_hours, num_vitals), np.nan)
    intime = icu_info['INTIME']

    for _, row in vital_data.iterrows():
        hours_since_admission = (row['CHARTTIME'] - intime).total_seconds() / 3600
        hour_idx = int(hours_since_admission)
        if 0 <= hour_idx < max_hours:
            vital_name = row['VITAL_NAME']
            if vital_name in vitals:
                vital_idx = vitals.index(vital_name)
                value = row['VALUENUM']
                if pd.notna(value) and value > 0:
                    if np.isnan(sequence[hour_idx, vital_idx]):
                        sequence[hour_idx, vital_idx] = value
                    else:
                        sequence[hour_idx, vital_idx] = (sequence[hour_idx, vital_idx] + value) / 2

    return sequence

print("\n🔄 Zaman serileri oluşturuluyor...")

MAX_HOURS = 48
VITALS = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']

sequences = []
labels = []
valid_icustays = []
sequence_lengths = []

for idx, row in icu_data.iterrows():
    icustay_id = row['ICUSTAY_ID']
    stay_vitals = chart_vitals[chart_vitals['ICUSTAY_ID'] == icustay_id]

    if len(stay_vitals) > 10:
        seq = create_hourly_sequences(icustay_id, stay_vitals, row, MAX_HOURS)
        valid_hours = np.sum(~np.isnan(seq).all(axis=1))

        if valid_hours >= 6:
            sequences.append(seq)
            labels.append(row['HOSPITAL_EXPIRE_FLAG'])
            valid_icustays.append(icustay_id)
            sequence_lengths.append(valid_hours)

print(f"✓ Oluşturulan zaman serisi sayısı: {len(sequences)}")
print(f"✓ Ortalama seri uzunluğu: {np.mean(sequence_lengths):.1f} saat")
print(f"✓ Min/Max uzunluk: {np.min(sequence_lengths)}/{np.max(sequence_lengths)} saat")

print("\n" + "=" * 70)
print("BÖLÜM 5: VERİ ÖN İŞLEME")
print("=" * 70)

X = np.array(sequences)
y = np.array(labels)

print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")
print(f"✓ Mortalite oranı: {y.mean()*100:.2f}%")

print("\n🔄 Eksik değerler dolduruluyor...")

def fill_missing_values(X):
    X_filled = X.copy()

    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            series = X_filled[i, :, j]

            mask = np.isnan(series)
            idx = np.where(~mask, np.arange(len(series)), 0)
            np.maximum.accumulate(idx, out=idx)
            series_ffill = series[idx]

            mask = np.isnan(series_ffill)
            if mask.any():
                idx = np.where(~mask, np.arange(len(series_ffill)), len(series_ffill)-1)
                idx = np.minimum.accumulate(idx[::-1])[::-1]
                series_bfill = series_ffill[idx]
                X_filled[i, :, j] = series_bfill
            else:
                X_filled[i, :, j] = series_ffill

    for j in range(X_filled.shape[2]):
        col = X_filled[:, :, j].flatten()
        median_val = np.nanmedian(col)
        if np.isnan(median_val):
            median_val = 0
        X_filled[:, :, j] = np.nan_to_num(X_filled[:, :, j], nan=median_val)

    return X_filled

X_filled = fill_missing_values(X)

print("🔄 Normalizasyon yapılıyor...")

original_shape = X_filled.shape
X_flat = X_filled.reshape(-1, X_filled.shape[-1])

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_flat)
X_normalized = X_normalized.reshape(original_shape)

print(f"✓ Normalize edilmiş veri shape: {X_normalized.shape}")

print("\n" + "=" * 70)
print("BÖLÜM 6: PYTORCH DATASET")
print("=" * 70)

class MIMICSequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

print("\n" + "=" * 70)
print("BÖLÜM 7: LSTM MODEL MİMARİSİ")
print("=" * 70)

class LSTMMortalityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMMortalityPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        context, attention_weights = self.attention_net(lstm_out)
        out = self.fc(context)
        return out, attention_weights

INPUT_SIZE = len(VITALS)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.4

model = LSTMMortalityPredictor(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

print(f"📊 Model Mimarisi:")
print(model)
print(f"\nToplam parametre: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "=" * 70)
print("BÖLÜM 8: 5-FOLD CROSS-VALIDATION")
print("=" * 70)

N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
PATIENCE = 15

fold_results = {'accuracy': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []}
all_y_true = []
all_y_pred = []
all_y_proba = []
all_attention_weights = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"\n🚀 {N_FOLDS}-Fold Cross-Validation başlıyor...")
print(f"   Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_normalized, y), 1):
    print(f"\n{'='*50}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'='*50}")

    X_train, X_val = X_normalized[train_idx], X_normalized[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    len_train = [sequence_lengths[i] for i in train_idx]
    len_val = [sequence_lengths[i] for i in val_idx]

    train_ds = MIMICSequenceDataset(X_train, y_train, len_train)
    val_ds = MIMICSequenceDataset(X_val, y_val, len_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMMortalityPredictor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(sum(y_train), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y, batch_len in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_len in val_loader:
                outputs, _ = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    y_pred_list = []
    y_proba_list = []
    attention_list = []

    with torch.no_grad():
        for batch_X, batch_y, batch_len in val_loader:
            outputs, attn_weights = model(batch_X)
            proba = torch.sigmoid(outputs).squeeze().numpy()
            y_proba_list.extend(proba if proba.ndim > 0 else [proba.item()])
            attention_list.append(attn_weights.numpy())

    y_proba_val = np.array(y_proba_list)
    y_pred_val = (y_proba_val > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred_val)
    try:
        auc = roc_auc_score(y_val, y_proba_val)
    except:
        auc = 0.5
    f1 = f1_score(y_val, y_pred_val, zero_division=0)
    prec = precision_score(y_val, y_pred_val, zero_division=0)
    rec = recall_score(y_val, y_pred_val, zero_division=0)

    fold_results['accuracy'].append(acc)
    fold_results['auc'].append(auc)
    fold_results['f1'].append(f1)
    fold_results['precision'].append(prec)
    fold_results['recall'].append(rec)

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred_val)
    all_y_proba.extend(y_proba_val)

    print(f"\n  📈 Fold {fold} Sonuçları:")
    print(f"     Accuracy:  {acc:.4f}")
    print(f"     ROC-AUC:   {auc:.4f}")
    print(f"     F1 Score:  {f1:.4f}")

print("\n" + "=" * 70)
print("BÖLÜM 9: LSTM CROSS-VALIDATION SONUÇLARI")
print("=" * 70)

print("\n📊 5-Fold CV Ortalamaları (± Std):")
print(f"  Accuracy:  {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
print(f"  ROC-AUC:   {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
print(f"  F1 Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
print(f"  Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
print(f"  Recall:    {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")

print("\n📊 Genel Confusion Matrix:")
cm = confusion_matrix(all_y_true, all_y_pred)
print(f"  True Negative:  {cm[0,0]:3d}  |  False Positive: {cm[0,1]:3d}")
print(f"  False Negative: {cm[1,0]:3d}  |  True Positive:  {cm[1,1]:3d}")

print("\n📋 Classification Report:")
print(classification_report(all_y_true, all_y_pred,
                          target_names=['Survived', 'Died'],
                          zero_division=0))

print("\n" + "=" * 70)
print("BÖLÜM 10: MODEL KARŞILAŞTIRMASI")
print("=" * 70)

feedforward_results = {
    'Accuracy': 0.8526,
    'ROC-AUC': 0.9030,
    'F1 Score': 0.7683,
}

lstm_results = {
    'Accuracy': np.mean(fold_results['accuracy']),
    'ROC-AUC': np.mean(fold_results['auc']),
    'F1 Score': np.mean(fold_results['f1']),
}

print("\n📊 Feedforward NN vs LSTM:")
print(f"{'Metrik':<12} {'Feedforward':>12} {'LSTM':>12} {'Fark':>12}")
print("-" * 50)
for metric in feedforward_results:
    ff_val = feedforward_results[metric]
    lstm_val = lstm_results[metric]
    diff = lstm_val - ff_val
    sign = "+" if diff > 0 else ""
    print(f"{metric:<12} {ff_val:>12.4f} {lstm_val:>12.4f} {sign}{diff:>11.4f}")

print("\n" + "=" * 70)
print("BÖLÜM 11: GÖRSELLEŞTİRME")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = list(feedforward_results.keys())
x = np.arange(len(metrics))
width = 0.35

ff_vals = list(feedforward_results.values())
lstm_vals = list(lstm_results.values())

bars1 = axes[0, 0].bar(x - width/2, ff_vals, width, label='Feedforward NN', color='#3498db')
bars2 = axes[0, 0].bar(x + width/2, lstm_vals, width, label='LSTM', color='#e74c3c')

axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Model Karşılaştırması')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metrics)
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1)

for bar, val in zip(bars1, ff_vals):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, lstm_vals):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

folds = range(1, N_FOLDS + 1)
axes[0, 1].plot(folds, fold_results['accuracy'], 'o-', label='Accuracy', linewidth=2)
axes[0, 1].plot(folds, fold_results['auc'], 's-', label='AUC', linewidth=2)
axes[0, 1].plot(folds, fold_results['f1'], '^-', label='F1', linewidth=2)
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('LSTM: Fold-by-Fold Sonuçlar')
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(True, alpha=0.3)

sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
            xticklabels=['Survived', 'Died'],
            yticklabels=['Survived', 'Died'])
axes[1, 0].set_title('LSTM Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

sample_idx = 0
sample_seq = X[sample_idx]

for i, vital in enumerate(VITALS):
    valid_mask = ~np.isnan(X[sample_idx, :, i])
    if valid_mask.any():
        axes[1, 1].plot(np.where(valid_mask)[0], X[sample_idx, valid_mask, i],
                        label=vital, alpha=0.7)

axes[1, 1].set_xlabel('Saat (ICU girişinden itibaren)')
axes[1, 1].set_ylabel('Değer')
axes[1, 1].set_title(f'Örnek Zaman Serisi (Hasta #{valid_icustays[sample_idx]})')
axes[1, 1].legend(loc='upper right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mimic_lstm_results.png', dpi=150)
print("✓ Grafikler 'mimic_lstm_results.png' olarak kaydedildi")
plt.show()

print("\n" + "=" * 70)
print("ÖZET")
print("=" * 70)

print(f"""
📊 LSTM ZAMAN SERİSİ MODELİ:
   - Girdi: {MAX_HOURS} saatlik vital bulgu serisi
   - Vital sayısı: {len(VITALS)} (HR, SBP, DBP, MAP, RR, SPO2, TEMP)
   - Örnek sayısı: {len(sequences)}
   - Model: Bidirectional LSTM + Attention

🎯 LSTM CV SONUÇLARI:
   - Accuracy:  {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}
   - ROC-AUC:   {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}
   - F1 Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}

📈 FEEDFORWARD NN vs LSTM:
   LSTM modeli zaman içindeki değişimleri yakalayarak
   hastalığın seyrini daha iyi analiz edebilir.

   Demo veri setinde örnek sayısı az olduğundan
   LSTM'in tam potansiyeli görülemeyebilir.
   Gerçek MIMIC-III'te (~50K hasta) LSTM genellikle
   daha iyi performans gösterir.

💡 LSTM AVANTAJLARI:
   1. Temporal patterns yakalar (trend, değişim hızı)
   2. "Hasta kötüleşiyor mu?" sorusuna cevap verir
   3. Eksik verilere daha dayanıklı (sequence learning)
   4. Attention ile kritik zaman noktalarını gösterir
""")

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'vitals': VITALS,
    'max_hours': MAX_HOURS,
    'cv_results': fold_results
}, 'lstm_mortality_model.pth')
print("✓ LSTM modeli 'lstm_mortality_model.pth' olarak kaydedildi")
