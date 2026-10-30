"""Diagnóstico de la Época 2 (Etapa 2a).

Audita:
1. Desfase de distribución (Distribution Shift) entre train (60%), val (20%) y test (20%).
2. Curvas de pérdida época por época para EMB_LSTM_v2 bajo diferentes tasas de aprendizaje.
3. Impacto en el error de prueba (MAE) y época óptima de convergencia.
"""
import os
import re
import json
import glob
import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

_STOP_WORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()

DATASET_DIR = "dataset"
OUT_DIR = "artefactos_exportados/etapa2"
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)

def load_data():
    frames = []
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        df = pd.read_csv(f)
        df["Project"] = os.path.basename(f).replace(".csv", "")
        df["full_text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        df["Effort"] = df["storypoint"].astype(float)
        df = df[df["Effort"] > 0].dropna(subset=["Effort"])
        df["Order"] = df["issuekey"].astype(str).str.extract(r"(\d+)$")[0].astype(float)
        frames.append(df[["Project", "issuekey", "Order", "full_text", "Effort"]])
    data = pd.concat(frames, ignore_index=True)
    return data

def split_temporal(df):
    train, val, test = [], [], []
    for _, g in df.groupby("Project", sort=True):
        g = g.sort_values(["Order", "issuekey"])
        n = len(g)
        a, b = int(n * 0.6), int(n * 0.8)
        train += g.index[:a].tolist()
        val += g.index[a:b].tolist()
        test += g.index[b:].tolist()
    return np.array(train), np.array(val), np.array(test)

def build_model(vocab_size, lr=1e-3, max_len=100, embed_dim=128, lstm_units=64):
    model = keras.Sequential([
        keras.Input(shape=(max_len,), dtype="int64"),
        layers.Embedding(vocab_size, embed_dim, mask_zero=True),
        layers.LSTM(lstm_units),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ], name=f"emb_lstm_lr_{lr}")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.Huber(delta=1.0))
    return model

def main():
    print("=== INICIANDO SUBETAPA 2a: DIAGNÓSTICO DE LA ÉPOCA 2 ===")
    data = load_data()
    print(f"Total de tickets cargados: {len(data)}")

    tr_idx, va_idx, te_idx = split_temporal(data)
    y = data["Effort"].to_numpy(dtype="float32")
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
    
    # 1. Comparativa de distribución estadística (Train vs Val vs Test)
    ks_tr_va = ks_2samp(y_tr, y_va)
    ks_tr_te = ks_2samp(y_tr, y_te)
    
    dist_report = {
        "train": {
            "n": int(len(y_tr)),
            "mean": float(np.mean(y_tr)),
            "median": float(np.median(y_tr)),
            "std": float(np.std(y_tr)),
            "pct_ge_13": float(np.mean(y_tr >= 13) * 100)
        },
        "val": {
            "n": int(len(y_va)),
            "mean": float(np.mean(y_va)),
            "median": float(np.median(y_va)),
            "std": float(np.std(y_va)),
            "pct_ge_13": float(np.mean(y_va >= 13) * 100)
        },
        "test": {
            "n": int(len(y_te)),
            "mean": float(np.mean(y_te)),
            "median": float(np.median(y_te)),
            "std": float(np.std(y_te)),
            "pct_ge_13": float(np.mean(y_te >= 13) * 100)
        },
        "ks_train_vs_val": {"statistic": float(ks_tr_va.statistic), "pvalue": float(ks_tr_va.pvalue)},
        "ks_train_vs_test": {"statistic": float(ks_tr_te.statistic), "pvalue": float(ks_tr_te.pvalue)}
    }
    
    print("\n--- Distribución de Story Points ---")
    print(f"Train: Media={dist_report['train']['mean']:.2f}, Mediana={dist_report['train']['median']:.1f}, % >=13 SP={dist_report['train']['pct_ge_13']:.1f}%")
    print(f"Val:   Media={dist_report['val']['mean']:.2f}, Mediana={dist_report['val']['median']:.1f}, % >=13 SP={dist_report['val']['pct_ge_13']:.1f}%")
    print(f"Test:  Media={dist_report['test']['mean']:.2f}, Mediana={dist_report['test']['median']:.1f}, % >=13 SP={dist_report['test']['pct_ge_13']:.1f}%")
    print(f"Kolmogorov-Smirnov Train vs Val: stat={ks_tr_va.statistic:.4f}, p={ks_tr_va.pvalue:.4e}")
    print(f"Kolmogorov-Smirnov Train vs Test: stat={ks_tr_te.statistic:.4f}, p={ks_tr_te.pvalue:.4e}")

    # 2. Preprocesamiento de texto y tokenización ajustada solo en Train
    print("\nAplicando preprocesamiento de texto...")
    t0 = time.time()
    clean_texts = [preprocess_text(t) for t in data["full_text"]]
    print(f"Texto limpio en {time.time()-t0:.1f} s")
    
    clean_tr = [clean_texts[i] for i in tr_idx]
    clean_va = [clean_texts[i] for i in va_idx]
    clean_te = [clean_texts[i] for i in te_idx]

    vocab_size = 20000
    max_len = 100
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize=None,
        split="whitespace"
    )
    vectorizer.adapt(clean_tr)

    x_tr = vectorizer(np.array(clean_tr)).numpy()
    x_va = vectorizer(np.array(clean_va)).numpy()
    x_te = vectorizer(np.array(clean_te)).numpy()

    z_tr = np.log1p(y_tr)
    z_va = np.log1p(y_va)
    z_te = np.log1p(y_te)

    # 3. Auditoría de curvas con diferentes Learning Rates
    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4]
    lr_results = {}

    for lr in learning_rates:
        print(f"\nEntrenando EMB_LSTM_v2 con lr={lr} (seed=42, max_epochs=20)...")
        keras.utils.set_random_seed(42)
        model = build_model(vocab_size=vocab_size, lr=lr, max_len=max_len)
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0)
        ]
        
        t_start = time.time()
        hist = model.fit(
            x_tr, z_tr,
            validation_data=(x_va, z_va),
            epochs=20,
            batch_size=64,
            callbacks=callbacks,
            verbose=0
        )
        t_elapsed = time.time() - t_start
        
        val_losses = [float(v) for v in hist.history["val_loss"]]
        train_losses = [float(v) for v in hist.history["loss"]]
        best_epoch = int(np.argmin(val_losses) + 1)
        best_val_loss = float(np.min(val_losses))
        
        # Inferencia en Test
        pred_z_te = model.predict(x_te, verbose=0).ravel()
        pred_y_te = np.clip(np.expm1(pred_z_te), 0.0, None)
        test_mae = float(np.mean(np.abs(pred_y_te - y_te)))
        test_rmse = float(np.sqrt(np.mean((pred_y_te - y_te)**2)))
        mae_ge_13 = float(np.mean(np.abs(pred_y_te[y_te >= 13] - y_te[y_te >= 13])))
        
        lr_results[str(lr)] = {
            "best_epoch": best_epoch,
            "total_epochs": len(val_losses),
            "best_val_loss": best_val_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mae_ge_13": mae_ge_13,
            "train_time_s": t_elapsed,
            "history_loss": train_losses,
            "history_val_loss": val_losses
        }
        
        print(f"-> LR {lr}: Mejor época={best_epoch}/{len(val_losses)}, Best Val Loss={best_val_loss:.4f}, Test MAE={test_mae:.3f}, MAE(>=13)={mae_ge_13:.2f}, Tiempo={t_elapsed:.1f}s")
        for ep, (tl, vl) in enumerate(zip(train_losses[:6], val_losses[:6]), start=1):
            star = " *" if ep == best_epoch else ""
            print(f"     Época {ep:02d}: train_loss={tl:.4f}, val_loss={vl:.4f}{star}")

    diagnostico_final = {
        "distribucion": dist_report,
        "sensibilidad_lr": lr_results
    }

    report_path = os.path.join(OUT_DIR, "diagnostico_epoca2.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(diagnostico_final, f, indent=2)

    print(f"\n[OK] Diagnóstico completado y guardado en {report_path}")

if __name__ == "__main__":
    main()

