"""
fase2_lstm_v2.py — Fase II corregida: LSTM sobre secuencias reales + protocolo comparativo.

Diferencias respecto a fase2_lstm.py (v1):
  1. La LSTM recibe secuencias de tokens (TextVectorization + Embedding) en lugar de
     un vector TF-IDF con timesteps=1, por lo que la recurrencia sí modela el orden.
  2. Todo el preprocesamiento (vocabulario TF-IDF / tokenizador) se ajusta SOLO con
     el conjunto de entrenamiento (sin fuga de información hacia la prueba).
  3. El modelo principal predice log1p(SP) con pérdida Huber para atenuar el efecto
     de los outliers (SP hasta 100) que, con MSE, empujan las predicciones a la media.
  4. Semillas fijas y operaciones deterministas (reproducibilidad).
  5. Protocolo de evaluación comparativo:
       - Baselines ingenuos: media, mediana global y mediana por proyecto.
       - TFIDF_LSTM_v1_replica: la arquitectura v1 exacta, pero sin fuga.
       - TFIDF_LSTM_v1_original: el modelo v1 ya entrenado (solo partición aleatoria).
       - EMB_LSTM_mse_raw: ablación (misma red v2, pero MSE sobre SP crudos).
       - EMB_LSTM_v2: modelo propuesto.
     Métricas: MAE, MdAE, RMSE, SA (Standardized Accuracy), sesgo, máximo/p99 de las
     predicciones, MAE por proyecto, error por rango de SP, Wilcoxon y A12.
  6. Dos particiones:
       - random:   80/20 aleatoria con random_state=42 (idéntica a v1 → comparable).
       - temporal: por proyecto, ordenado por issuekey, 60/20/20 (protocolo Deep-SE).

Uso:
    python fase2_lstm_v2.py                       # ambas particiones, todos los modelos
    python fase2_lstm_v2.py --split temporal
    python fase2_lstm_v2.py --quick               # prueba de humo (submuestra, 2 épocas)
    python fase2_lstm_v2.py --skip-tfidf --skip-ablation

Salidas: artefactos_exportados/v2/<split>/ (CSV, figuras, modelo, vocabulario)
         artefactos_exportados/v2/resumen_comparativo.csv
"""

import argparse
import glob
import json
import os
import re
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ==============================================================================
# Configuración
# ==============================================================================

DATASET_DIR = "./dataset"
OUT_ROOT = "artefactos_exportados/v2"
V1_DATA_PATH = "artefactos_exportados/datos_vectorizados.pkl"
V1_MODEL_PATH = "artefactos_exportados/modelo_lstm_fase2.keras"

REFERENCE_MODEL = "EMB_LSTM_v2"
SP_BINS = [0, 2, 5, 8, 13, 20, np.inf]
SP_LABELS = ["1-2", "3-5", "6-8", "9-13", "14-20", "21+"]


def parse_args():
    p = argparse.ArgumentParser(description="Fase II v2 — LSTM secuencial y evaluación comparativa")
    p.add_argument("--split", choices=["random", "temporal", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-len", type=int, default=100, help="Longitud de secuencia (tokens)")
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--skip-tfidf", action="store_true", help="No reentrenar la réplica TF-IDF v1")
    p.add_argument("--skip-ablation", action="store_true", help="No entrenar la ablación MSE/raw")
    p.add_argument("--skip-v1-original", action="store_true", help="No evaluar el modelo v1 guardado")
    p.add_argument("--gpu", action="store_true", help="Usar GPU (Metal). Por defecto CPU determinista")
    p.add_argument("--quick", action="store_true", help="Prueba de humo: 3000 tickets, 2 épocas")
    return p.parse_args()


ARGS = parse_args()

import tensorflow as tf  # noqa: E402  (se importa tras leer argumentos)
import keras  # noqa: E402
from keras import layers  # noqa: E402
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402

if not ARGS.gpu:
    tf.config.set_visible_devices([], "GPU")
keras.utils.set_random_seed(ARGS.seed)
tf.config.experimental.enable_op_determinism()


# ==============================================================================
# BLOQUE 1: Datos y preprocesamiento (réplica del PLN de la Fase I)
# ==============================================================================

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    """Idéntico a model.py / api.py, pero con stopwords y lematizador cacheados."""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)


def load_data(folder_path):
    # Se conserva glob.glob sin ordenar (igual que model.py) para que la partición
    # aleatoria con random_state=42 reproduzca exactamente el conjunto de prueba v1.
    frames = []
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        df = pd.read_csv(file)
        df["Project"] = os.path.basename(file).replace(".csv", "")
        df["full_text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        df["Effort"] = df["storypoint"]
        df = df[df["Effort"] > 0]
        df = df.dropna(subset=["Effort"])
        # Orden cronológico aproximado: número secuencial de la clave (p. ej. MDL-1626)
        df["Order"] = df["issuekey"].astype(str).str.extract(r"(\d+)$")[0].astype(float)
        frames.append(df[["Project", "issuekey", "Order", "full_text", "Effort"]])
    return pd.concat(frames, ignore_index=True)


def split_random(n, seed):
    idx = np.arange(n)
    # random_state fijo en 42 para coincidir con model.py (partición v1)
    train_full, test = train_test_split(idx, test_size=0.2, random_state=42)
    train, val = train_test_split(train_full, test_size=0.2, random_state=seed)
    return train, val, test


def split_temporal(df):
    """Por proyecto: primeros 60% entrenamiento, 20% validación, 20% prueba."""
    train, val, test = [], [], []
    for _, g in df.groupby("Project", sort=True):
        g = g.sort_values(["Order", "issuekey"])
        n = len(g)
        a, b = int(n * 0.6), int(n * 0.8)
        train += g.index[:a].tolist()
        val += g.index[a:b].tolist()
        test += g.index[b:].tolist()
    return np.array(train), np.array(val), np.array(test)


# ==============================================================================
# BLOQUE 2: Modelos
# ==============================================================================


def make_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=ARGS.patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, ARGS.patience // 2),
                          min_lr=1e-5, verbose=1),
    ]


def build_tfidf_lstm_v1(num_features):
    """Arquitectura exacta de fase2_lstm.py (timesteps=1, LSTM relu, MSE)."""
    model = keras.Sequential([
        keras.Input(shape=(1, num_features)),
        layers.LSTM(128, activation="relu", return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="linear"),
    ], name="tfidf_lstm_v1")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def build_emb_lstm(vocab_size, loss):
    """LSTM secuencial: tokens → Embedding (con máscara de padding) → LSTM (tanh)."""
    model = keras.Sequential([
        keras.Input(shape=(ARGS.max_len,), dtype="int64"),
        layers.Embedding(vocab_size, ARGS.embed_dim, mask_zero=True),
        layers.LSTM(ARGS.lstm_units),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ], name="emb_lstm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    return model


def fit(model, x_tr, y_tr, x_va, y_va):
    keras.utils.set_random_seed(ARGS.seed)
    t0 = time.time()
    hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=ARGS.epochs,
                     batch_size=ARGS.batch_size, callbacks=make_callbacks(), verbose=2)
    return hist.history, time.time() - t0


# ==============================================================================
# BLOQUE 3: Métricas y pruebas estadísticas
# ==============================================================================


def mae_random_guessing(y_test, y_train):
    """MAE esperado de adivinar al azar un valor de entrenamiento (P0 de Shepperd & MacDonell)."""
    s = np.sort(np.asarray(y_train, dtype=float))
    c = np.concatenate([[0.0], np.cumsum(s)])
    n = len(s)
    y = np.asarray(y_test, dtype=float)
    k = np.searchsorted(s, y, side="right")
    left = y * k - c[k]
    right = (c[n] - c[k]) - y * (n - k)
    return float(np.mean((left + right) / n))


def global_metrics(y, p, mae_p0):
    err = np.abs(p - y)
    mae = err.mean()
    return {
        "MAE": mae,
        "MdAE": np.median(err),
        "RMSE": np.sqrt(np.mean((p - y) ** 2)),
        "SA_%": (1 - mae / mae_p0) * 100,
        "Sesgo": np.mean(p - y),
        "Pred_max": p.max(),
        "Pred_p99": np.percentile(p, 99),
        "MAE_SP>=13": err[y >= 13].mean() if (y >= 13).any() else np.nan,
    }


def a12(err_ref, err_other):
    """Vargha-Delaney: P(error de referencia < error del otro). >0.5 ⇒ referencia mejor."""
    u = stats.mannwhitneyu(err_other, err_ref, alternative="two-sided").statistic
    return u / (len(err_ref) * len(err_other))


def effect_label(a):
    d = abs(a - 0.5)
    return "despreciable" if d < 0.06 else "pequeño" if d < 0.14 else "mediano" if d < 0.21 else "grande"


# ==============================================================================
# BLOQUE 4: Gráficas
# ==============================================================================


def plot_pred_vs_real(y, preds, path):
    models = [m for m in preds if m.startswith(("TFIDF", "EMB"))]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    for ax, m in zip(axes[0], models):
        ax.scatter(y, preds[m], s=6, alpha=0.3)
        ax.plot([0, 100], [0, 100], "k--", lw=1, label="y = x")
        ax.axhline(13, color="red", ls=":", lw=1, label="13 SP")
        ax.set(title=m, xlabel="SP reales", ylabel="SP predichos", xlim=(0, 100), ylim=(0, 100))
        ax.legend(loc="upper left")
        ax.grid(ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bins(bins_df, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    pivot_mae = bins_df.pivot(index="Rango_SP", columns="Modelo", values="MAE").reindex(SP_LABELS)
    pivot_mae.plot(kind="bar", ax=ax1, width=0.85)
    ax1.set(title="MAE por rango de SP reales", xlabel="Rango de SP", ylabel="MAE")
    ax1.grid(axis="y", ls=":", alpha=0.6)
    ax1.legend(fontsize=8)

    pivot_pred = bins_df.pivot(index="Rango_SP", columns="Modelo", values="Media_pred").reindex(SP_LABELS)
    real = bins_df.drop_duplicates("Rango_SP").set_index("Rango_SP")["Media_real"].reindex(SP_LABELS)
    for m in pivot_pred.columns:
        if m.startswith(("TFIDF", "EMB")):
            ax2.plot(SP_LABELS, pivot_pred[m], marker="o", label=m)
    ax2.plot(SP_LABELS, real, "k--", marker="s", label="Media real")
    ax2.set(title="Predicción media vs. valor real por rango", xlabel="Rango de SP", ylabel="SP")
    ax2.grid(ls=":", alpha=0.6)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_histories(histories, path):
    if not histories:
        return
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4), squeeze=False)
    for ax, (m, h) in zip(axes[0], histories.items()):
        ax.plot(h["loss"], label="Entrenamiento")
        ax.plot(h["val_loss"], label="Validación")
        ax.set(title=m, xlabel="Época", ylabel="Pérdida")
        ax.grid(ls=":", alpha=0.6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ==============================================================================
# BLOQUE 5: Experimento por partición
# ==============================================================================


def run_split(data, split_name):
    print("\n" + "=" * 78)
    print(f"PARTICIÓN: {split_name.upper()}")
    print("=" * 78)
    out_dir = os.path.join(OUT_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    if split_name == "random":
        tr, va, te = split_random(len(data), ARGS.seed)
    else:
        tr, va, te = split_temporal(data)
    print(f"Entrenamiento: {len(tr)} | Validación: {len(va)} | Prueba: {len(te)}")

    text = data["Clean_Story"].to_numpy()
    y = data["Effort"].to_numpy(dtype="float32")
    proj = data["Project"].to_numpy()
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    y_tr_full = np.concatenate([y_tr, y_va])  # para baselines (no ven la prueba)

    preds, histories, train_time = {}, {}, {}

    # --- Baselines ingenuos -----------------------------------------------------
    preds["Baseline_media"] = np.full(len(te), y_tr_full.mean())
    preds["Baseline_mediana"] = np.full(len(te), np.median(y_tr_full))
    med_proj = pd.Series(y_tr_full, index=np.concatenate([proj[tr], proj[va]])).groupby(level=0).median()
    preds["Baseline_mediana_proyecto"] = (
        pd.Series(proj[te]).map(med_proj).fillna(np.median(y_tr_full)).to_numpy())

    # --- Modelo v1 original (ya entrenado; con fuga TF-IDF) -------------------------
    if split_name == "random" and not ARGS.skip_v1_original and not ARGS.quick:
        if os.path.exists(V1_DATA_PATH) and os.path.exists(V1_MODEL_PATH):
            import joblib
            _, X_test_v1, _, y_test_v1 = joblib.load(V1_DATA_PATH)
            if np.array_equal(np.asarray(y_test_v1.index), te):
                m_v1 = keras.models.load_model(V1_MODEL_PATH)
                x = np.asarray(X_test_v1, dtype="float32").reshape(len(te), 1, -1)
                preds["TFIDF_LSTM_v1_original"] = m_v1.predict(x, verbose=0).ravel()
                del X_test_v1, x
                print("Modelo v1 original evaluado sobre el mismo conjunto de prueba.")
            else:
                print("AVISO: el conjunto de prueba v1 no coincide; se omite v1 original.")
        else:
            print("AVISO: no se encontraron artefactos v1; se omite v1 original.")

    # --- Réplica v1 sin fuga (TF-IDF ajustado solo con entrenamiento) ---------------
    if not ARGS.skip_tfidf:
        print("\n[TFIDF_LSTM_v1_replica] TF-IDF ajustado solo con entrenamiento")
        tfidf = TfidfVectorizer(max_features=1000)
        x_tr = tfidf.fit_transform(text[tr]).toarray().astype("float32")
        nf = x_tr.shape[1]
        x_va = tfidf.transform(text[va]).toarray().astype("float32").reshape(-1, 1, nf)
        x_te = tfidf.transform(text[te]).toarray().astype("float32").reshape(-1, 1, nf)
        x_tr = x_tr.reshape(-1, 1, nf)
        model = build_tfidf_lstm_v1(nf)
        histories["TFIDF_LSTM_v1_replica"], train_time["TFIDF_LSTM_v1_replica"] = fit(
            model, x_tr, y_tr, x_va, y_va)
        preds["TFIDF_LSTM_v1_replica"] = model.predict(x_te, verbose=0).ravel()
        del x_tr, x_va, x_te, model

    # --- Tokenización secuencial (ajustada solo con entrenamiento) -------------------
    vectorizer = layers.TextVectorization(
        max_tokens=ARGS.vocab_size, output_mode="int", output_sequence_length=ARGS.max_len,
        standardize=None, split="whitespace")
    vectorizer.adapt(text[tr])
    vocab = vectorizer.get_vocabulary()
    s_tr, s_va, s_te = (vectorizer(text[i]).numpy() for i in (tr, va, te))
    n_tokens = np.array([len(t.split()) for t in text[tr]])
    print(f"\nVocabulario: {len(vocab)} tokens | Textos truncados a {ARGS.max_len}: "
          f"{(n_tokens > ARGS.max_len).mean() * 100:.1f}% | Mediana de tokens: {np.median(n_tokens):.0f}")

    # --- Ablación: misma red secuencial, MSE sobre SP crudos ------------------------
    if not ARGS.skip_ablation:
        print("\n[EMB_LSTM_mse_raw] Ablación: Embedding+LSTM, MSE, objetivo crudo")
        model = build_emb_lstm(len(vocab), "mean_squared_error")
        histories["EMB_LSTM_mse_raw"], train_time["EMB_LSTM_mse_raw"] = fit(
            model, s_tr, y_tr, s_va, y_va)
        preds["EMB_LSTM_mse_raw"] = model.predict(s_te, verbose=0).ravel()
        del model

    # --- Modelo propuesto: Embedding+LSTM, Huber sobre log1p(SP) ---------------------
    print(f"\n[{REFERENCE_MODEL}] Embedding+LSTM, Huber, objetivo log1p(SP)")
    model = build_emb_lstm(len(vocab), keras.losses.Huber(delta=1.0))
    histories[REFERENCE_MODEL], train_time[REFERENCE_MODEL] = fit(
        model, s_tr, np.log1p(y_tr), s_va, np.log1p(y_va))
    preds[REFERENCE_MODEL] = np.clip(np.expm1(model.predict(s_te, verbose=0).ravel()), 0, None)

    model.save(os.path.join(out_dir, "modelo_emb_lstm_v2.keras"))
    with open(os.path.join(out_dir, "vocabulario_v2.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(out_dir, "config_v2.json"), "w") as f:
        json.dump({"max_len": ARGS.max_len, "vocab_size": len(vocab), "target": "log1p",
                   "inverse": "expm1, clip >= 0", "loss": "huber(delta=1.0)",
                   "split": split_name, "seed": ARGS.seed}, f, indent=2)

    # ==========================================================================
    # Evaluación
    # ==========================================================================
    mae_p0 = mae_random_guessing(y_te, y_tr_full)

    rows = []
    for m, p in preds.items():
        r = {"Particion": split_name, "Modelo": m, **global_metrics(y_te, p, mae_p0)}
        r["Tiempo_entrenamiento_s"] = train_time.get(m, np.nan)
        r["Epocas"] = len(histories[m]["loss"]) if m in histories else np.nan
        rows.append(r)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(out_dir, "metricas_globales.csv"), index=False)

    pred_df = pd.DataFrame({"Project": proj[te], "issuekey": data["issuekey"].to_numpy()[te],
                            "SP_real": y_te, **{m: p for m, p in preds.items()}})
    pred_df.to_csv(os.path.join(out_dir, "predicciones.csv"), index=False)

    proj_df = (pred_df.melt(id_vars=["Project", "issuekey", "SP_real"], var_name="Modelo",
                            value_name="Pred")
               .assign(AE=lambda d: (d["Pred"] - d["SP_real"]).abs())
               .pivot_table(index="Project", columns="Modelo", values="AE", aggfunc="mean"))
    proj_df.insert(0, "n_prueba", pred_df.groupby("Project").size())
    proj_df.round(3).to_csv(os.path.join(out_dir, "mae_por_proyecto.csv"))

    bins = pd.cut(y_te, bins=SP_BINS, labels=SP_LABELS)
    bin_rows = []
    for m, p in preds.items():
        for lab in SP_LABELS:
            mask = np.asarray(bins == lab)
            if mask.any():
                bin_rows.append({"Modelo": m, "Rango_SP": lab, "n": int(mask.sum()),
                                 "Media_real": y_te[mask].mean(), "Media_pred": p[mask].mean(),
                                 "MAE": np.abs(p[mask] - y_te[mask]).mean(),
                                 "Sesgo": (p[mask] - y_te[mask]).mean()})
    bins_df = pd.DataFrame(bin_rows)
    bins_df.round(3).to_csv(os.path.join(out_dir, "error_por_rango.csv"), index=False)

    err_ref = np.abs(preds[REFERENCE_MODEL] - y_te)
    stat_rows = []
    for m, p in preds.items():
        if m == REFERENCE_MODEL:
            continue
        err = np.abs(p - y_te)
        w = stats.wilcoxon(err_ref, err)
        a = a12(err_ref, err)
        stat_rows.append({"Referencia": REFERENCE_MODEL, "Comparado": m,
                          "MAE_ref": err_ref.mean(), "MAE_comparado": err.mean(),
                          "Wilcoxon_p": w.pvalue, "A12": a, "Efecto": effect_label(a)})
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(os.path.join(out_dir, "pruebas_estadisticas.csv"), index=False)

    plot_pred_vs_real(y_te, preds, os.path.join(out_dir, "fig_pred_vs_real.png"))
    plot_bins(bins_df, os.path.join(out_dir, "fig_error_por_rango.png"))
    plot_histories(histories, os.path.join(out_dir, "fig_curvas_aprendizaje.png"))

    pd.set_option("display.width", 200)
    print(f"\nMAE P0 (adivinanza aleatoria): {mae_p0:.3f}")
    print("\n--- Métricas globales ---")
    print(metrics_df.drop(columns="Particion").round(3).to_string(index=False))
    print("\n--- Pruebas estadísticas (errores absolutos) ---")
    print(stats_df.round(4).to_string(index=False))
    return metrics_df


# ==============================================================================
# Ejecución principal
# ==============================================================================

if __name__ == "__main__":
    if ARGS.quick:
        ARGS.epochs = 2
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | "
          f"Dispositivos: {[d.device_type for d in tf.config.get_visible_devices()]}")

    data = load_data(DATASET_DIR)
    if ARGS.quick:
        data = data.sample(n=3000, random_state=ARGS.seed).reset_index(drop=True)
    print(f"Tickets: {len(data)} | Proyectos: {data['Project'].nunique()}")
    print("Aplicando PLN...")
    data["Clean_Story"] = data["full_text"].apply(preprocess_text)

    splits = ["random", "temporal"] if ARGS.split == "both" else [ARGS.split]
    if ARGS.quick:
        OUT_ROOT = os.path.join(OUT_ROOT, "_quick")
    os.makedirs(OUT_ROOT, exist_ok=True)

    all_metrics = pd.concat([run_split(data, s) for s in splits], ignore_index=True)
    all_metrics.round(4).to_csv(os.path.join(OUT_ROOT, "resumen_comparativo.csv"), index=False)
    print(f"\nResumen guardado en {OUT_ROOT}/resumen_comparativo.csv")
