#!/usr/bin/env python3
# IVC_discovered_vCANN_train_fixed.py

import os, re, glob, json, shutil, csv, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# =================== USER SETTINGS ===================
DATA_ROOT   = "data"          # data/<age>/*.csv
OUT_ROOT    = "runs_ivcann"   # runs per age
SEED        = 42

# Training
EPOCHS_MAX  = 4000
BATCH       = 32
LR_BASE     = 1e-1            # base LR (scheduler warm-up + cosine decay)
CLIPNORM    = 5.0
REG_KIND    = "L2"            # "L1" or "L2"
REG_PEN     = 0.0

# Fast dev
FAST_DEV          = True
MAX_POINTS_FILE   = 300
MAX_POINTS_AGE    = 6000
FAST_EPOCHS_MAX   = 1200

# Cleaning / resume
CLEAN_AT_START    = True
CLEAN_ALL         = False
RESUME            = False

# Numerics
LMIN       = 1e-3
LMAX       = 5.0
EXP_CLIP   = 15.0
VOL_EPS    = 1e-6

# Early stopping
ES_PATIENCE = 120 if FAST_DEV else 200
ES_TOL      = 1e-4

# Normalize targets
NORMALIZE_TARGETS = True
# =====================================================
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_frac=0.05):
        super().__init__()
        self.base_lr = tf.convert_to_tensor(base_lr, tf.float32)
        self.total_steps = tf.cast(max(1, int(total_steps)), tf.float32)
        self.warm = tf.cast(max(1, int(warmup_frac * max(1, int(total_steps)))), tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # linear warmup
        warm_ratio = tf.minimum(1.0, step / self.warm)
        lr = self.base_lr * warm_ratio
        # cosine decay after warmup
        prog = tf.clip_by_value((step - self.warm) / tf.maximum(1.0, self.total_steps - self.warm), 0.0, 1.0)
        lr = lr * 0.5 * (1.0 + tf.cos(prog * tf.constant(np.pi, tf.float32)))
        return lr

def set_seeds(seed: int):
    tf.keras.backend.clear_session()
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def find_ages(root):
    ages = [os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    ages.sort(); return ages

def parse_csvs_for_age(age_dir):
    pd_samples, fl_samples = [], []
    for csvp in glob.glob(os.path.join(age_dir, "*.csv")):
        fname = os.path.basename(csvp)
        m = re.match(r"(?P<spec>.+)[_.](?P<kind>pd|fl)(?P<lvl>\d+)\.csv$", fname, flags=re.IGNORECASE)
        if not m: continue
        spec = m.group("spec"); kind = m.group("kind").lower(); lvl = int(m.group("lvl"))

        df = pd.read_csv(csvp)
        df.columns = [c.strip().lower() for c in df.columns]
        cols = set(df.columns)
        if not {"lambda_theta","lambda_z"}.issubset(cols):
            print(f"[WARN] {csvp}: needs lambda_theta, lambda_z. Found {sorted(cols)}"); continue

        lam_th = df["lambda_theta"].astype(float).to_numpy()
        lam_z  = df["lambda_z"].astype(float).to_numpy()

        if kind == "pd":
            if "sigma_theta_kpa" not in cols:
                print(f"[WARN] {csvp}: missing sigma_theta_kpa"); continue
            sig_th = df["sigma_theta_kpa"].astype(float).to_numpy()
            mask = np.isfinite(lam_th) & np.isfinite(lam_z) & np.isfinite(sig_th) & (lam_th>LMIN) & (lam_z>LMIN)
            n = int(mask.sum())
            if n:
                if FAST_DEV and n > MAX_POINTS_FILE:
                    idx = np.linspace(0, n-1, MAX_POINTS_FILE, dtype=int)
                    lam_th = lam_th[mask][idx]; lam_z = lam_z[mask][idx]; sig_th = sig_th[mask][idx]
                else:
                    lam_th = lam_th[mask]; lam_z = lam_z[mask]; sig_th = sig_th[mask]
                pd_samples.append(dict(specimen=spec, level=lvl, lam_th=lam_th, lam_z=lam_z, sig_th=sig_th, path=csvp))
        else:
            if "sigma_z_kpa" not in cols:
                print(f"[WARN] {csvp}: missing sigma_z_kpa"); continue
            sig_z = df["sigma_z_kpa"].astype(float).to_numpy()
            mask = np.isfinite(lam_th) & np.isfinite(lam_z) & np.isfinite(sig_z) & (lam_th>LMIN) & (lam_z>LMIN)
            n = int(mask.sum())
            if n:
                if FAST_DEV and n > MAX_POINTS_FILE:
                    idx = np.linspace(0, n-1, MAX_POINTS_FILE, dtype=int)
                    lam_th = lam_th[mask][idx]; lam_z = lam_z[mask][idx]; sig_z = sig_z[mask][idx]
                else:
                    lam_th = lam_th[mask]; lam_z = lam_z[mask]; sig_z = sig_z[mask]
                fl_samples.append(dict(specimen=spec, level=lvl, lam_th=lam_th, lam_z=lam_z, sig_z=sig_z, path=csvp))
        print(f"[INFO] Parsed {csvp} for {spec} ({kind}, lvl={lvl}) → N={n}")
    return {"pd": pd_samples, "fl": fl_samples}

def build_dataset(age_blob):
    X_th, y_th, X_z, y_z = [], [], [], []
    for s in age_blob["pd"]:
        mask = np.isfinite(s["lam_th"]) & np.isfinite(s["lam_z"]) & np.isfinite(s["sig_th"])
        if mask.any():
            X_th.append(np.stack([s["lam_th"][mask], s["lam_z"][mask]], axis=1))
            y_th.append(s["sig_th"][mask])
    for s in age_blob["fl"]:
        mask = np.isfinite(s["lam_th"]) & np.isfinite(s["lam_z"]) & np.isfinite(s["sig_z"])
        if mask.any():
            X_z.append(np.stack([s["lam_th"][mask], s["lam_z"][mask]], axis=1))
            y_z.append(s["sig_z"][mask])
    X_th = np.concatenate(X_th, axis=0) if X_th else np.zeros((0,2))
    y_th = np.concatenate(y_th, axis=0) if y_th else np.zeros((0,))
    X_z  = np.concatenate(X_z,  axis=0) if X_z  else np.zeros((0,2))
    y_z  = np.concatenate(y_z,  axis=0) if y_z  else np.zeros((0,))
    if FAST_DEV and X_th.shape[0] > MAX_POINTS_AGE:
        idx = np.linspace(0, X_th.shape[0]-1, MAX_POINTS_AGE, dtype=int)
        X_th, y_th = X_th[idx], y_th[idx]
    if FAST_DEV and X_z.shape[0] > MAX_POINTS_AGE:
        idx = np.linspace(0, X_z.shape[0]-1, MAX_POINTS_AGE, dtype=int)
        X_z, y_z = X_z[idx], y_z[idx]
    return X_th, y_th, X_z, y_z

# -------------- vCANN invariants --------------
@tf.function(reduce_retracing=True)
def vcann_invariants(lam_th, lam_z):
    lam_th = tf.clip_by_value(lam_th, LMIN, LMAX)
    lam_z  = tf.clip_by_value(lam_z,  LMIN, LMAX)
    lam_r  = 1.0 / (lam_th * lam_z + tf.constant(VOL_EPS, tf.float32))

    C11 = lam_th**2; C22 = lam_z**2; C33 = lam_r**2
    Cinv11 = 1.0 / C11; Cinv22 = 1.0 / C22; Cinv33 = 1.0 / C33

    Iiso = (C11 + C22 + C33) / 3.0
    Ith  = C11
    Iz   = C22
    Jiso = (Cinv11 + Cinv22 + Cinv33) / 3.0
    Jth  = Cinv11
    Jz   = Cinv22
    IIIc = C11 * C22 * C33  # ~1

    return Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc, C11, C22, C33, lam_r

def regularizer(kind, pen):
    if pen <= 0: return None
    return keras.regularizers.l2(pen) if kind == "L2" else keras.regularizers.l1(pen)

class PsiNetVCANN(keras.Model):
    """7 branches (Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc) → 4 feats/branch → NonNeg mixer → Ψ."""
    def __init__(self, reg_kind="L2", reg_pen=0.0):
        super().__init__()
        reg = regularizer(reg_kind, reg_pen)
        self.names = ["Iiso","Ith","Iz","Jiso","Jth","Jz","IIIc"]
        self._shift = {name: tf.constant(1.0, dtype=tf.float32) for name in self.names}

        k_small_pos = keras.initializers.RandomUniform(minval=1e-4, maxval=5e-4, seed=SEED+123)
        def pos_init(seed): return keras.initializers.RandomUniform(minval=0.01, maxval=0.05, seed=SEED + seed)

        def branch(seed_base: int):
            return [
                keras.layers.Dense(1, use_bias=False, kernel_initializer=k_small_pos,   kernel_regularizer=reg),  # lin
                keras.layers.Dense(1, use_bias=False, kernel_initializer=pos_init(seed_base+1), kernel_regularizer=reg,
                                   kernel_constraint=keras.constraints.NonNeg()),                                # expm1
                keras.layers.Dense(1, use_bias=False, kernel_initializer=k_small_pos,   kernel_regularizer=reg),  # quad lin
                keras.layers.Dense(1, use_bias=False, kernel_initializer=pos_init(seed_base+2), kernel_regularizer=reg,
                                   kernel_constraint=keras.constraints.NonNeg()),                                # quad expm1
            ]
        self.br = {name: branch(10 + 5*i) for i, name in enumerate(self.names)}

        self.mixer = keras.layers.Dense(
            1, use_bias=False,
            kernel_initializer=keras.initializers.RandomUniform(minval=0.01, maxval=0.05, seed=SEED+999),
            kernel_constraint=keras.constraints.NonNeg(),
            kernel_regularizer=reg
        )

    @staticmethod
    def _safe_expm1(z):
        z = tf.clip_by_value(z, -EXP_CLIP, EXP_CLIP)
        return tf.math.expm1(z)

    def _apply_branch(self, x, layers):
        t1 = layers[0](x)
        t2 = self._safe_expm1(layers[1](x))
        x2 = tf.square(x)
        t3 = layers[2](x2)
        t4 = self._safe_expm1(layers[3](x2))
        return tf.concat([t1, t2, t3, t4], axis=1)

    def call(self, inv_tuple, training=False):
        Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc = inv_tuple
        xs = {
            "Iiso": Iiso - self._shift["Iiso"],
            "Ith":  Ith  - self._shift["Ith"],
            "Iz":   Iz   - self._shift["Iz"],
            "Jiso": Jiso - self._shift["Jiso"],
            "Jth":  Jth  - self._shift["Jth"],
            "Jz":   Jz   - self._shift["Jz"],
            "IIIc": IIIc - self._shift["IIIc"],
        }
        feats = [self._apply_branch(xs[name], self.br[name]) for name in self.names]
        feats = tf.concat(feats, axis=1)  # 7 × 4 = 28
        psi = self.mixer(feats)
        return tf.squeeze(psi, axis=1)

# -------------- Stress helper --------------
def principal_stresses_from_dpsi(C11, C22, C33, dpsi_dC11, dpsi_dC22, dpsi_dC33):
    if dpsi_dC11 is None: dpsi_dC11 = tf.zeros_like(C11)
    if dpsi_dC22 is None: dpsi_dC22 = tf.zeros_like(C22)
    if dpsi_dC33 is None: dpsi_dC33 = tf.zeros_like(C33)
    p = 2.0 * C33 * dpsi_dC33
    sig_th = -p + 2.0 * C11 * dpsi_dC11
    sig_z  = -p + 2.0 * C22 * dpsi_dC22
    return sig_th, sig_z

# -------------- Dataset maker --------------
def _make_joint_dataset(X_th, y_th, X_z, y_z, batch):
    rows = X_th.shape[0] + X_z.shape[0]
    if rows == 0: return None
    X_list, ytheta_list, yz_list = [], [], []
    if X_th.shape[0] > 0:
        X_list.append(X_th); ytheta_list.append(y_th); yz_list.append(np.full_like(y_th, np.nan))
    if X_z.shape[0] > 0:
        X_list.append(X_z);  ytheta_list.append(np.full_like(y_z, np.nan)); yz_list.append(y_z)
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    ytheta = np.concatenate(ytheta_list, axis=0).astype(np.float32)
    yz = np.concatenate(yz_list, axis=0).astype(np.float32)
    if FAST_DEV and X.shape[0] > MAX_POINTS_AGE:
        idx = np.linspace(0, X.shape[0]-1, MAX_POINTS_AGE, dtype=int)
        X, ytheta, yz = X[idx], ytheta[idx], yz[idx]
    ds = tf.data.Dataset.from_tensor_slices((X, ytheta, yz)) \
         .shuffle(min(10000, max(1000, X.shape[0])), seed=SEED) \
         .batch(batch, drop_remainder=True) \
         .prefetch(tf.data.AUTOTUNE)
    return ds, X.shape[0]

# -------------- LR schedule (pure Python callable) --------------
def make_cosine_warmup_schedule(base_lr, steps_total, warmup_frac=0.05):
    warm = max(1, int(warmup_frac * max(1, steps_total)))
    pi = np.pi
    def lr_fn(step):
        step_f = tf.cast(step, tf.float32)
        warm_f = tf.cast(warm, tf.float32)
        total_f = tf.cast(max(1, steps_total), tf.float32)
        warm_ratio = tf.minimum(1.0, step_f / warm_f)
        lr = base_lr * warm_ratio
        progress = tf.clip_by_value((step_f - warm_f) / tf.maximum(1.0, (total_f - warm_f)), 0.0, 1.0)
        lr = lr * 0.5 * (1.0 + tf.cos(progress * tf.constant(pi, tf.float32)))
        return lr
    return lr_fn

# -------------- Train step --------------
@tf.function(jit_compile=False, experimental_relax_shapes=True)
def _compute_joint_grads(model, xb, ytheta, yz, mu_th, sd_th, mu_z, sd_z):
    lam_th = xb[:, 0:1]; lam_z  = xb[:, 1:2]
    with tf.GradientTape() as tape_out:
        with tf.GradientTape() as tape_in:
            tape_in.watch([lam_th, lam_z])
            Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc, C11, C22, C33, _ = vcann_invariants(lam_th, lam_z)
            psi = model((Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc), training=True)
        dpsi_dC11, dpsi_dC22, dpsi_dC33 = tape_in.gradient(
            psi, [C11, C22, C33],
            unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        sig_th_hat, sig_z_hat = principal_stresses_from_dpsi(C11, C22, C33, dpsi_dC11, dpsi_dC22, dpsi_dC33)
        sig_th_hat = tf.squeeze(sig_th_hat, 1)
        sig_z_hat  = tf.squeeze(sig_z_hat,  1)

        m_th = tf.math.is_finite(ytheta)
        m_z  = tf.math.is_finite(yz)

        if NORMALIZE_TARGETS:
            sig_th_hat_n = (tf.boolean_mask(sig_th_hat, m_th) - mu_th) / sd_th if tf.reduce_any(m_th) else 0.0
            sig_z_hat_n  = (tf.boolean_mask(sig_z_hat,  m_z ) - mu_z ) / sd_z  if tf.reduce_any(m_z ) else 0.0
            ytheta_n = (tf.boolean_mask(ytheta, m_th) - mu_th) / sd_th if tf.reduce_any(m_th) else 0.0
            yz_n     = (tf.boolean_mask(yz,     m_z ) - mu_z ) / sd_z  if tf.reduce_any(m_z ) else 0.0

            loss_th = tf.reduce_mean(tf.square(sig_th_hat_n - ytheta_n)) if tf.reduce_any(m_th) else 0.0
            loss_z  = tf.reduce_mean(tf.square(sig_z_hat_n  - yz_n))     if tf.reduce_any(m_z)  else 0.0
        else:
            loss_th = tf.reduce_mean(tf.square(tf.boolean_mask(sig_th_hat, m_th) - tf.boolean_mask(ytheta, m_th))) if tf.reduce_any(m_th) else 0.0
            loss_z  = tf.reduce_mean(tf.square(tf.boolean_mask(sig_z_hat,  m_z ) - tf.boolean_mask(yz,     m_z )))  if tf.reduce_any(m_z ) else 0.0

        n_th = tf.reduce_sum(tf.cast(m_th, tf.float32))
        n_z  = tf.reduce_sum(tf.cast(m_z,  tf.float32))
        n_tot = tf.maximum(n_th + n_z, 1.0)
        w_th = tf.where(n_th > 0, n_tot / (2.0 * n_th), 0.0)
        w_z  = tf.where(n_z  > 0, n_tot / (2.0 * n_z),  0.0)
        loss = w_th * loss_th + w_z * loss_z

    grads = tape_out.gradient(loss, model.trainable_variables)
    return loss, grads, loss_th, loss_z

def _num_steps(num_rows, batch):
    return max(1, num_rows // batch)  # drop_remainder=True

def train_joint(model, X_th, y_th, X_z, y_z, epochs, batch, lr_base, mu_th, sd_th, mu_z, sd_z):
    ds, num_rows = _make_joint_dataset(X_th, y_th, X_z, y_z, batch)
    if ds is None:
        print("[SKIP] No data for joint training"); return

    steps_per_epoch = _num_steps(num_rows, batch)
    total_steps = steps_per_epoch * (FAST_EPOCHS_MAX if FAST_DEV else epochs)

    lr_schedule = WarmupCosine(LR_BASE, total_steps, warmup_frac=0.05)
    opt = keras.optimizers.legacy.Adam(learning_rate=lr_schedule, clipnorm=CLIPNORM)
    _ = opt.iterations  

    best = float("inf"); stale = 0
    E = FAST_EPOCHS_MAX if FAST_DEV else epochs

    for ep in range(1, E+1):
        losses, losses_th, losses_z = [], [], []
        for xb, ytheta, yz in ds:
            loss, grads, lth, lz = _compute_joint_grads(model, xb, ytheta, yz, mu_th, sd_th, mu_z, sd_z)

            # Ensure we ALWAYS advance iterations (and thus the LR schedule)
            pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if not pairs:  # extremely unlikely; fallback: no-op update on 1 var
                z = tf.zeros_like(model.trainable_variables[0])
                pairs = [(z, model.trainable_variables[0])]
            opt.apply_gradients(pairs)

            losses.append(float(loss))
            losses_th.append(float(lth) if np.isfinite(lth) else np.nan)
            losses_z.append(float(lz)  if np.isfinite(lz)  else np.nan)

        avg  = float(np.nanmean(losses)) if losses else np.nan
        avgT = float(np.nanmean(losses_th)) if losses_th else np.nan
        avgZ = float(np.nanmean(losses_z))  if losses_z  else np.nan

        if (ep % 20) == 0 or ep <= 5:
            curr_lr = float(lr_schedule(opt.iterations).numpy())
            print(f"[joint] epoch {ep:4d}  loss={avg:9.5f}  (θ={avgT:9.5f}, z={avgZ:9.5f})  lr={curr_lr:.2e}")

        if avg + ES_TOL < best: best = avg; stale = 0
        else:
            stale += 1
            if stale >= ES_PATIENCE:
                print(f"[joint] early stop at epoch {ep} (best={best:.5f})")
                break

# -------------- Cleaning / IO --------------
def _rm_path(p: str) -> bool:
    try:
        if os.path.isdir(p) and not os.path.islink(p): shutil.rmtree(p)
        else: os.remove(p)
        return True
    except FileNotFoundError: return False
    except PermissionError as e: print(f"[CLEAN][WARN] {e}"); return False

def clean_outdir(outdir: str, delete_all: bool = False):
    outdir_abs = os.path.abspath(outdir)
    os.makedirs(outdir_abs, exist_ok=True)
    if delete_all:
        if outdir_abs in ("/","") or len(outdir_abs) < 5:
            raise RuntimeError(f"[CLEAN] Refusing suspicious path: {outdir_abs}")
        print(f"[CLEAN] Removing entire directory: {outdir_abs}")
        shutil.rmtree(outdir_abs, ignore_errors=True)
        os.makedirs(outdir_abs, exist_ok=True); return
    patterns = ["Psi_ckpt*", "*ckpt*", "checkpoint", "events*", "logs",
                "fit_*.png", "Psi_weights.txt", "Psi_mixer_features.csv", "meta.json"]
    removed = 0
    for pat in patterns:
        for p in glob.glob(os.path.join(outdir_abs, pat)):
            if _rm_path(p):
                removed += 1; print(f"[CLEAN] Removed: {p}")
    print(f"[CLEAN] Removed {removed} items from {outdir_abs}")

def quick_scatter(x, y, yhat, xlabel, ylabel, title, out_png):
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=10, label="data", alpha=0.6)
    plt.scatter(x, yhat, s=8, label="fit")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def save_weights(model, outdir):
    os.makedirs(outdir, exist_ok=True)
    def _scalar(var): return float(var.numpy().reshape(-1)[0])
    names4 = ["lin", "exp", "quad_lin", "quad_exp"]

    branch_rows = []
    for inv_name in model.names:
        layers = model.br[inv_name]
        for feat_name, layer in zip(names4, layers):
            branch_rows.append((inv_name, feat_name, _scalar(layer.kernel)))

    mixer_w = model.mixer.kernel.numpy().reshape(-1)  # length 28
    feature_order = []
    for inv_name in model.names:
        feature_order += [f"{inv_name}_{n}" for n in names4]

    txt_path = os.path.join(outdir, "Psi_weights.txt")
    with open(txt_path, "w") as f:
        f.write("Ψ-Net weights (vCANN invariants)\n")
        f.write("================================\n\n")
        f.write("Invariant reference shifts (all 1.0 at C=I):\n")
        for inv_name in model.names:
            f.write(f"  shift_{inv_name} = 1.0\n")
        f.write("\nBranch weights (pre-activation scalars):\n")
        last_inv = None
        for inv, feat, w in branch_rows:
            if inv != last_inv: f.write(f"\n[{inv}]\n"); last_inv = inv
            f.write(f"  {feat:9s}: {w:+.6e}\n")
        f.write("\nMixer weights (feature → Ψ):\n")
        for name, w in zip(feature_order, mixer_w):
            f.write(f"  {name:14s} -> {w:+.6e}\n")

    csv_path = os.path.join(outdir, "Psi_mixer_features.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf); writer.writerow(["feature","weight"])
        for name, w in zip(feature_order, mixer_w):
            writer.writerow([name, f"{w:.8e}"])

    ckpt = tf.train.Checkpoint(psinet=model)
    ckpt.write(os.path.join(outdir, "Psi_ckpt"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"reg": REG_KIND, "pen": REG_PEN, "arch": "vCANN_invariants"}, f, indent=2)
    print(f"[WEIGHTS] Saved:\n  - {txt_path}\n  - {csv_path}")

# -------------- Eval helpers --------------
def predict_sigma_theta(model, X):
    lam_th = tf.convert_to_tensor(X[:,0:1], dtype=tf.float32)
    lam_z  = tf.convert_to_tensor(X[:,1:2], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([lam_th, lam_z])
        Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc, C11, C22, C33, _ = vcann_invariants(lam_th, lam_z)
        psi = model((Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc), training=False)
    dpsi_dC11, dpsi_dC22, dpsi_dC33 = tape.gradient(
        psi, [C11, C22, C33],
        unconnected_gradients=tf.UnconnectedGradients.ZERO
    )
    sig_th, _ = principal_stresses_from_dpsi(C11, C22, C33, dpsi_dC11, dpsi_dC22, dpsi_dC33)
    return tf.squeeze(sig_th, 1).numpy()

def predict_sigma_z(model, X):
    lam_th = tf.convert_to_tensor(X[:,0:1], dtype=tf.float32)
    lam_z  = tf.convert_to_tensor(X[:,1:2], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([lam_th, lam_z])
        Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc, C11, C22, C33, _ = vcann_invariants(lam_th, lam_z)
        psi = model((Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc), training=False)
    dpsi_dC11, dpsi_dC22, dpsi_dC33 = tape.gradient(
        psi, [C11, C22, C33],
        unconnected_gradients=tf.UnconnectedGradients.ZERO
    )
    _, sig_z = principal_stresses_from_dpsi(C11, C22, C33, dpsi_dC11, dpsi_dC22, dpsi_dC33)
    return tf.squeeze(sig_z, 1).numpy()

# -------------- Main --------------
def main():
    set_seeds(SEED)
    ages = find_ages(DATA_ROOT)
    print(f"Found ages: {ages}")
    if not ages:
        print(f"[ERROR] No ages found under {DATA_ROOT}"); return

    for age in ages:
        age_dir = os.path.join(DATA_ROOT, age)
        print(f"Processing age: {age} ({age_dir})")
        blob = parse_csvs_for_age(age_dir)
        X_th, y_th, X_z, y_z = build_dataset(blob)

        outdir = os.path.join(OUT_ROOT, age)
        if CLEAN_ALL: clean_outdir(outdir, delete_all=True)
        elif CLEAN_AT_START: clean_outdir(outdir, delete_all=False)
        else: os.makedirs(outdir, exist_ok=True)

        if X_th.shape[0] == 0 and X_z.shape[0] == 0:
            print(f"[SKIP] {age}: no usable data."); continue

        # Target normalization stats
        eps = 1e-8
        mu_th = float(np.nanmean(y_th)) if y_th.size else 0.0
        sd_th = float(np.nanstd(y_th) + eps) if y_th.size else 1.0
        mu_z  = float(np.nanmean(y_z))  if y_z.size  else 0.0
        sd_z  = float(np.nanstd(y_z)  + eps) if y_z.size  else 1.0

        # Strategy
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        strategy = tf.distribute.get_strategy()
        print(f"[STRATEGY] Using DefaultStrategy (GPUs: {num_gpus})")

        with strategy.scope():
            model = PsiNetVCANN(REG_KIND, REG_PEN)

            # build vars at identity
            _lam_th = tf.ones((1,1), dtype=tf.float32)
            _lam_z  = tf.ones((1,1), dtype=tf.float32)
            Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc, *_ = vcann_invariants(_lam_th, _lam_z)
            _ = model((Iiso, Ith, Iz, Jiso, Jth, Jz, IIIc), training=False)

            ckpt_path = os.path.join(outdir, "Psi_ckpt")
            if RESUME and tf.io.gfile.exists(ckpt_path + ".index"):
                print(f"[RESUME] Loading checkpoint from {ckpt_path}")
                tf.train.Checkpoint(psinet=model).restore(ckpt_path).expect_partial()
            else:
                print("[RESET] Starting from random initialization (no checkpoint restore).")

            nT, nZ = X_th.shape[0], X_z.shape[0]
            print(f"==> {age}: joint training (Pd+Fl): Nθ={nT}  Nz={nZ}")
            train_joint(model, X_th, y_th, X_z, y_z, EPOCHS_MAX, BATCH, LR_BASE,
                        tf.constant(mu_th, tf.float32), tf.constant(sd_th, tf.float32),
                        tf.constant(mu_z,  tf.float32), tf.constant(sd_z,  tf.float32))

            tf.train.Checkpoint(psinet=model).write(ckpt_path)

        # Plots
        if X_th.shape[0] > 0:
            sig_hat = predict_sigma_theta(model, X_th)
            quick_scatter(X_th[:,0], y_th, sig_hat,
                          r"$\lambda_\theta$", r"$\sigma_\theta$ (kPa)",
                          f"{age} – Pd (all)", os.path.join(outdir, "fit_pd_theta_vs_lambda_theta.png"))
        if X_z.shape[0] > 0:
            sig_hat = predict_sigma_z(model, X_z)
            quick_scatter(X_z[:,1], y_z, sig_hat,
                          r"$\lambda_z$", r"$\sigma_z$ (kPa)",
                          f"{age} – Fl (all)", os.path.join(outdir, "fit_fl_sigma_z_vs_lambda_z.png"))

        save_weights(model, outdir)
        print(f"[DONE] Saved to {outdir}")

if __name__ == "__main__":
    main()
