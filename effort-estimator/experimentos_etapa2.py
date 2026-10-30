"""Diseño Factorial Completo y Modelo Condicionado por Proyecto (Etapa 2b).

Factores:
- Preprocesamiento: [Estándar (PLN clásico), Conservador (con dígitos y símbolos técnicos)]
- Escala de Objetivo: [Crudo (Story Points reales), Log1p (log(1+y))]
- Función de Pérdida: [MSE, Huber (delta=1.0)]
- Semillas: 5 semillas aleatorias por tratamiento [42, 101, 202, 303, 404]

Modelo Adicional:
- EMB_LSTM_project: Red recurrente bimodal condicionada por embedding de proyecto.
"""
import os
import re
import glob
import time
import argparse
import numpy as np
import pandas as pd
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

_STOP_WORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()

DATASET_DIR = "dataset"
OUT_DIR = "artefactos_exportados/etapa2"
os.makedirs(OUT_DIR, exist_ok=True)

FIB_SEQ = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89], dtype=float)

def snap_to_fibonacci(preds):
    preds = np.asarray(preds)
    # Distancia absoluta a cada elemento de Fibonacci
    idx = np.abs(preds[:, None] - FIB_SEQ[None, :]).argmin(axis=1)
    return FIB_SEQ[idx]

def preprocess_standard(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)

def preprocess_conservative(text):
    # Preserva letras, dígitos, puntos entre números (e.g. 3.12), y guiones entre palabras (e.g. utf-8, bug-fix)
    text = str(text).lower()
    text = re.sub(r"(?<=\d)\.(?=\d)", "DOT", text)  # protege puntos en versiones
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = text.replace("DOT", ".")
    tokens = [t for t in text.split() if len(t) > 1 or t.isalnum()]
    return " ".join(tokens)

def load_dataset():
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

def build_text_model(vocab_size, max_len, loss_type, lr=5e-4):
    loss_fn = "mse" if loss_type.lower() == "mse" else keras.losses.Huber(delta=1.0)
    model = keras.Sequential([
        keras.Input(shape=(max_len,), dtype="int64"),
        layers.Embedding(vocab_size, 128, mask_zero=True),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ], name="emb_lstm_factorial")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss_fn)
    return model

def build_conditioned_model(vocab_size, num_projects, max_len, loss_type, lr=5e-4):
    loss_fn = "mse" if loss_type.lower() == "mse" else keras.losses.Huber(delta=1.0)
    
    # Rama de Texto
    text_input = keras.Input(shape=(max_len,), dtype="int64", name="text_input")
    emb_text = layers.Embedding(vocab_size, 128, mask_zero=True)(text_input)
    lstm_out = layers.LSTM(64)(emb_text)
    lstm_drop = layers.Dropout(0.3)(lstm_out)
    
    # Rama de Proyecto (Embedding categórico)
    proj_input = keras.Input(shape=(1,), dtype="int32", name="proj_input")
    emb_proj = layers.Embedding(num_projects, 16)(proj_input)
    proj_flat = layers.Flatten()(emb_proj)
    proj_dense = layers.Dense(16, activation="relu")(proj_flat)
    
    # Fusión
    merged = layers.Concatenate()([lstm_drop, proj_dense])
    dense1 = layers.Dense(32, activation="relu")(merged)
    out = layers.Dense(1, activation="linear")(dense1)
    
    model = keras.Model(inputs=[text_input, proj_input], outputs=out, name="emb_lstm_project_conditioned")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss_fn)
    return model

def run_experiments(seeds=[42, 101, 202, 303, 404], lr=5e-4, epochs=20):
    data = load_dataset()
    tr_idx, va_idx, te_idx = split_temporal(data)
    y_raw = data["Effort"].to_numpy(dtype="float32")
    y_tr, y_va, y_te = y_raw[tr_idx], y_raw[va_idx], y_raw[te_idx]
    
    # Mapeo de proyectos a enteros
    project_names = sorted(data["Project"].unique())
    proj2id = {p: i for i, p in enumerate(project_names)}
    data["project_id"] = data["Project"].map(proj2id).to_numpy(dtype="int32")
    p_tr = data["project_id"].iloc[tr_idx].to_numpy()
    p_va = data["project_id"].iloc[va_idx].to_numpy()
    p_te = data["project_id"].iloc[te_idx].to_numpy()

    print("Preprocesando corpus bajo modalidades: Estándar y Conservadora...")
    text_clean = {
        "standard": [preprocess_standard(t) for t in data["full_text"]],
        "conservative": [preprocess_conservative(t) for t in data["full_text"]]
    }

    # Vectorizadores ajustados en Train
    vectorizers = {}
    vocab_size = 20000
    max_len = 100
    for mode in ["standard", "conservative"]:
        vec = layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=max_len,
            standardize=None,
            split="whitespace"
        )
        vec.adapt([text_clean[mode][i] for i in tr_idx])
        vectorizers[mode] = {
            "x_tr": vec(np.array([text_clean[mode][i] for i in tr_idx])).numpy(),
            "x_va": vec(np.array([text_clean[mode][i] for i in va_idx])).numpy(),
            "x_te": vec(np.array([text_clean[mode][i] for i in te_idx])).numpy()
        }

    # Matriz Factorial 2x2x2
    # Preprocessing: standard, conservative
    # Target: raw, log1p
    # Loss: mse, huber
    grid = []
    for prep in ["standard", "conservative"]:
        for tgt in ["raw", "log1p"]:
            for loss in ["mse", "huber"]:
                grid.append({
                    "model_name": f"LSTM_{prep}_{tgt}_{loss}",
                    "prep": prep,
                    "target": tgt,
                    "loss": loss,
                    "conditioned": False
                })
                
    # Agregar modelo condicionado (con mejor prep y loss)
    grid.append({
        "model_name": "LSTM_conditioned_standard_log1p_huber",
        "prep": "standard",
        "target": "log1p",
        "loss": "huber",
        "conditioned": True
    })
    grid.append({
        "model_name": "LSTM_conditioned_conservative_log1p_huber",
        "prep": "conservative",
        "target": "log1p",
        "loss": "huber",
        "conditioned": True
    })

    print(f"\nIniciando evaluación de {len(grid)} configuraciones a través de {len(seeds)} semillas...")
    all_runs = []

    for cfg in grid:
        c_name = cfg["model_name"]
        print(f"\n=======================================================")
        print(f"Configuración: {c_name} (Prep={cfg['prep']}, Target={cfg['target']}, Loss={cfg['loss']}, Cond={cfg['conditioned']})")
        print(f"=======================================================")
        
        # Objetivos
        if cfg["target"] == "log1p":
            t_tr, t_va, t_te = np.log1p(y_tr), np.log1p(y_va), np.log1p(y_te)
        else:
            t_tr, t_va, t_te = y_tr, y_va, y_te

        x_tr = vectorizers[cfg["prep"]]["x_tr"]
        x_va = vectorizers[cfg["prep"]]["x_va"]
        x_te = vectorizers[cfg["prep"]]["x_te"]

        for seed in seeds:
            keras.utils.set_random_seed(seed)
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0)
            ]

            t0 = time.time()
            if cfg["conditioned"]:
                model = build_conditioned_model(vocab_size, len(project_names), max_len, cfg["loss"], lr=lr)
                hist = model.fit(
                    [x_tr, p_tr], t_tr,
                    validation_data=([x_va, p_va], t_va),
                    epochs=epochs,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=0
                )
                pred_out = model.predict([x_te, p_te], verbose=0).ravel()
            else:
                model = build_text_model(vocab_size, max_len, cfg["loss"], lr=lr)
                hist = model.fit(
                    x_tr, t_tr,
                    validation_data=(x_va, t_va),
                    epochs=epochs,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=0
                )
                pred_out = model.predict(x_te, verbose=0).ravel()
            train_time = time.time() - t0

            # Transformación inversa si aplica
            if cfg["target"] == "log1p":
                pred_y = np.clip(np.expm1(pred_out), 0.0, None)
            else:
                pred_y = np.clip(pred_out, 0.0, None)

            # Métricas
            mae = float(np.mean(np.abs(pred_y - y_te)))
            mdae = float(np.median(np.abs(pred_y - y_te)))
            rmse = float(np.sqrt(np.mean((pred_y - y_te)**2)))
            mae_ge_13 = float(np.mean(np.abs(pred_y[y_te >= 13] - y_te[y_te >= 13])))
            
            # Fibonacci snapped
            fib_pred = snap_to_fibonacci(pred_y)
            mae_fib = float(np.mean(np.abs(fib_pred - y_te)))

            best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
            total_epochs = len(hist.history["val_loss"])

            run_record = {
                "config": c_name,
                "preprocessing": cfg["prep"],
                "target": cfg["target"],
                "loss": cfg["loss"],
                "conditioned": cfg["conditioned"],
                "seed": seed,
                "best_epoch": best_epoch,
                "total_epochs": total_epochs,
                "mae": mae,
                "mdae": mdae,
                "rmse": rmse,
                "mae_ge_13": mae_ge_13,
                "mae_fib": mae_fib,
                "train_time_s": train_time
            }
            all_runs.append(run_record)
            print(f"  [Seed {seed}] Best Ep: {best_epoch}/{total_epochs} | MAE: {mae:.3f} | MdAE: {mdae:.2f} | MAE(>=13): {mae_ge_13:.2f} | Fib MAE: {mae_fib:.3f} ({train_time:.1f}s)")

    # Guardar corridas individuales
    df_runs = pd.DataFrame(all_runs)
    runs_path = os.path.join(OUT_DIR, "runs_factoriales.csv")
    df_runs.to_csv(runs_path, index=False)

    # Resumen agrupado (Media ± Desviación Estándar)
    summary = df_runs.groupby(["config", "preprocessing", "target", "loss", "conditioned"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mdae_mean=("mdae", "mean"),
        mdae_std=("mdae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_ge_13_mean=("mae_ge_13", "mean"),
        mae_ge_13_std=("mae_ge_13", "std"),
        mae_fib_mean=("mae_fib", "mean"),
        mae_fib_std=("mae_fib", "std"),
        best_epoch_mean=("best_epoch", "mean")
    ).reset_index()

    summary_path = os.path.join(OUT_DIR, "resumen_factoriales.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=======================================================")
    print("RESUMEN FACTORIAL AGREGADO (5 SEMILLAS)")
    print("=======================================================")
    for _, r in summary.iterrows():
        print(f"{r['config']:<42} | MAE: {r['mae_mean']:.3f} ± {r['mae_std']:.3f} | >=13: {r['mae_ge_13_mean']:.2f} ± {r['mae_ge_13_std']:.2f} | Fib: {r['mae_fib_mean']:.3f} ± {r['mae_fib_std']:.3f} | BestEp: {r['best_epoch_mean']:.1f}")

    print(f"\n[OK] Resultados guardados en {runs_path} y {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    args = parser.parse_args()
    run_experiments(lr=args.lr, epochs=args.epochs)

