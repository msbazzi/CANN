#!/usr/bin/env python3
# IVCANN_discovered.py
# Discover a strain-energy density function (Ψ) for juvenile lamb IVC
# Folder layout:
#   data/<age>/<specimen>_(pd|fl)<level>.csv
# Examples:
#   data/3weeks/Lucid_fl10.csv
#   data/8weeks/Scout_pd95.csv
#
# CSV columns expected (from your MATLAB exporter):
#   lambda_theta, sigma_theta_kPa, lambda_z, sigma_z_kPa


import os, re, glob, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # before importing tensorflow if you want to suppress INFO/WARN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ========= EDIT HERE (main parameters) ====================================
DATA_ROOT = "data"            # root folder with age subfolders
OUT_ROOT  = "runs_ivcann"     # outputs per age go here
EPOCHS_TH = 4000              # epochs for θ (Pd) fit
EPOCHS_Z  = 4000              # epochs for z (Fl) fit
BATCH     = 64                # batch size
LR        = 5e-3              # learning rate
REG_KIND  = "L2"              # "L1" or "L2"
REG_PEN   = 0.0               # regularization strength
SEED      = 42                # reproducibility (ish)
LMIN = 1e-3      # min stretch
LMAX = 5.0       # max stretch (very conservative)
EXP_CLIP = 15.0  # cap exp pre-activations to avoid overflow
# ==== FAST DEV TOGGLES (local quick tests) ===================================
FAST_DEV = True          # set False for full training
MAX_POINTS_PER_FILE = 300  # subsample from each CSV (per test/specimen)
MAX_POINTS_PER_AGE  = 6000 # cap total points per age (after stacking)
FAST_EPOCHS_TH      = 400  # override EPOCHS_TH when FAST_DEV
FAST_EPOCHS_Z       = 400  # override EPOCHS_Z  when FAST_DEV

# --- Normalization options ---
NORM_STRESS = True                # normalize stresses per file during training
NORM_METHOD = "quantile"          # "quantile" | "zscore" | "minmax"
QUANT_LO, QUANT_HI = 0.10, 0.90   # used if NORM_METHOD == "quantile"
EPS_SCALE = 1e-6                  # guard against zero scale
# =============================================================================


tf.random.set_seed(SEED)
USE_XLA = os.environ.get("USE_XLA", "0") == "1"   # default: OFF
if USE_XLA:
    try:
        tf.config.optimizer.set_jit(True)
        print("[XLA] Enabled JIT")
    except Exception as e:
        print(f"[XLA] Not available here, disabling: {e}")
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
CSV_REQUIRED = ["lambda_theta","sigma_theta_kPa","lambda_z","sigma_z_kPa"]

def find_ages(root):
    ages = [os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    ages.sort()
    return ages

def parse_csvs_for_age(age_dir):
    """
    Return dict with two lists:
      {
        'pd': [ { specimen, level, lam_th, lam_z, sig_th, path }, ... ],
        'fl': [ { specimen, level, lam_th, lam_z, sig_z,  path }, ... ],
      }
    Accepts either "Specimen_pd105.csv" or "Specimen.pd105.csv".
    Normalizes column names (strip + lower) to avoid KeyError.
    """
    pd_samples, fl_samples = [], []
    for csv in glob.glob(os.path.join(age_dir, "*.csv")):
        fname = os.path.basename(csv)

        # accept underscore or dot separator before 'pd'/'fl'
        m = re.match(r"(?P<spec>.+)[_.](?P<kind>pd|fl)(?P<lvl>\d+)\.csv$", fname, flags=re.IGNORECASE)
        if not m:
            # also accept the underscore-only pattern (for safety)
            m = re.match(r"(?P<spec>.+)_(?P<kind>pd|fl)(?P<lvl>\d+)\.csv$", fname, flags=re.IGNORECASE)
        if not m:
            continue

        spec = m.group("spec")
        kind = m.group("kind").lower()
        lvl  = int(m.group("lvl"))

        # --- read & normalize headers ---
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower() for c in df.columns]
        cols = set(df.columns)

        # required basics (lambda_theta, lambda_z always used)
        req = {"lambda_theta", "lambda_z"}
        missing = req - cols
        if missing:
            print(f"[WARN] {csv}: missing required columns {missing}. Found: {sorted(cols)}")
            continue

        lam_th = df["lambda_theta"].astype(float).to_numpy()
        lam_z  = df["lambda_z"].astype(float).to_numpy()

        if kind == "pd":
            # need sigma_theta_kpa only
            if "sigma_theta_kpa" not in cols:
                print(f"[WARN] {csv}: missing 'sigma_theta_kpa'. Found: {sorted(cols)}")
                continue
            sig_th = df["sigma_theta_kpa"].astype(float).to_numpy()
            mask = np.isfinite(lam_th) & np.isfinite(lam_z) & np.isfinite(sig_th)
            mask &= (lam_th > LMIN) & (lam_z > LMIN)  
            if mask.any():
                if FAST_DEV and mask.sum() > MAX_POINTS_PER_FILE:
                    idx = np.linspace(0, mask.sum()-1, MAX_POINTS_PER_FILE, dtype=int)
                    lam_th_m = lam_th[mask][idx]; lam_z_m = lam_z[mask][idx]; sig_th_m = sig_th[mask][idx]
                else:
                    lam_th_m = lam_th[mask]; lam_z_m = lam_z[mask]; sig_th_m = sig_th[mask]
                # inside kind == "pd" branch, right before pd_samples.append(...)
                ref, scl = _ref_scale(sig_th_m, NORM_METHOD)
                pd_samples.append(dict(
                    specimen=spec, level=lvl,
                    lam_th=lam_th_m, lam_z=lam_z_m,
                    sig_th=sig_th_m, path=csv,
                    th_ref=ref, th_scl=scl
                ))

        else:
            # kind == 'fl': need sigma_z_kpa only
            if "sigma_z_kpa" not in cols:
                print(f"[WARN] {csv}: missing 'sigma_z_kpa'. Found: {sorted(cols)}")
                continue
            sig_z = df["sigma_z_kpa"].astype(float).to_numpy()
            mask = np.isfinite(lam_th) & np.isfinite(lam_z) & np.isfinite(sig_z)
            mask &= (lam_th > LMIN) & (lam_z > LMIN)
            if mask.any():
                if FAST_DEV and mask.sum() > MAX_POINTS_PER_FILE:
                    idx = np.linspace(0, mask.sum()-1, MAX_POINTS_PER_FILE, dtype=int)
                    lam_th_m = lam_th[mask][idx]; lam_z_m = lam_z[mask][idx]; sig_z_m = sig_z[mask][idx]
                else:
                    lam_th_m = lam_th[mask]; lam_z_m = lam_z[mask]; sig_z_m = sig_z[mask]
                ref, scl = _ref_scale(sig_z_m, NORM_METHOD)
                fl_samples.append(dict(
                    specimen=spec, level=lvl,
                    lam_th=lam_th_m, lam_z=lam_z_m,
                    sig_z=sig_z_m, path=csv,
                    z_ref=ref, z_scl=scl
                ))

        print(f"[INFO] Parsed {csv} for {spec} ({kind}, lvl={lvl}) "
              f"→ N={mask.sum()}")

    return {"pd": pd_samples, "fl": fl_samples}

def build_dataset(age_blob):
    """Stack all samples for one age."""
    X_th, y_th = [], []
    X_z,  y_z  = [], []
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

    X_th = np.concatenate(X_th, axis=0) if len(X_th) > 0 else np.zeros((0,2))
    y_th = np.concatenate(y_th, axis=0) if len(y_th) > 0 else np.zeros((0,))
    X_z  = np.concatenate(X_z,  axis=0) if len(X_z)  > 0 else np.zeros((0,2))
    y_z  = np.concatenate(y_z,  axis=0) if len(y_z)  > 0 else np.zeros((0,))

    # Cap totals in fast mode
    if FAST_DEV and X_th.shape[0] > MAX_POINTS_PER_AGE:
        sel = np.linspace(0, X_th.shape[0]-1, MAX_POINTS_PER_AGE, dtype=int)
        X_th = X_th[sel]; y_th = y_th[sel]
    if FAST_DEV and X_z.shape[0] > MAX_POINTS_PER_AGE:
        sel = np.linspace(0, X_z.shape[0]-1, MAX_POINTS_PER_AGE, dtype=int)
        X_z = X_z[sel]; y_z = y_z[sel]

    return X_th, y_th, X_z, y_z

# -----------------------------------------------------------------------------
# Physics: invariants and stresses
# -----------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def invariants_from_stretches(lam_th, lam_z):
    # Clamp stretches to avoid division by zero / huge lam_r
    lam_th = tf.clip_by_value(lam_th, LMIN, LMAX)
    lam_z  = tf.clip_by_value(lam_z,  LMIN, LMAX)
    lam_r = 1.0 / (lam_th * lam_z)
    I1 = lam_th**2 + lam_z**2 + lam_r**2
    I2 = lam_th**2 * lam_z**2 + lam_th**2 * lam_r**2 + lam_z**2 * lam_r**2
    I4th = lam_th**2
    I4z  = lam_z**2
    return I1, I2, I4th, I4z

@tf.function(reduce_retracing=True)
def sigma_theta_from_derivs(lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4th):
    lam_r = 1.0/(lam_th*lam_z)
    return 2.0*( dW_dI1*(lam_th**2 - lam_r**2) + dW_dI2*(lam_z**2*(lam_th**2 - lam_r**2)) + dW_dI4th*(lam_th**2) )

@tf.function(reduce_retracing=True)
def sigma_z_from_derivs(lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4z):
    lam_r = 1.0/(lam_th*lam_z)
    return 2.0*( dW_dI1*(lam_z**2 - lam_r**2) + dW_dI2*(lam_th**2*(lam_z**2 - lam_r**2)) + dW_dI4z*(lam_z**2) )

# -----------------------------------------------------------------------------
# Model: invariant-based Ψ(I1,I2,I4θ,I4z)
# -----------------------------------------------------------------------------
def regularizer(kind, pen):
    if pen <= 0: return None
    return keras.regularizers.l2(pen) if kind == "L2" else keras.regularizers.l1(pen)

class PsiNet(keras.Model):
    """Tiny nonnegative-mixed network for Ψ with invariant-wise branches.

    Fixes the RandomUniform initializer warning by seeding each layer's
    initializer independently (no reused initializer objects).
    """
    def __init__(self, reg_kind="L1", reg_pen=0.0):
        super().__init__()
        reg = regularizer(reg_kind, reg_pen)
        self.shift_I1  = tf.constant(3.0, dtype=tf.float32)
        self.shift_I2  = tf.constant(3.0, dtype=tf.float32)
        self.shift_I4  = tf.constant(1.0, dtype=tf.float32)

        kzer = keras.initializers.Zeros()

        def pos_init(seed):
            # different seed per layer avoids the warning and identical weights
            return keras.initializers.RandomUniform(minval=0.0, maxval=0.1, seed=SEED + seed)

        def branch(seed_base: int):
            # four layers with distinct initializers
            return [
                keras.layers.Dense(
                    1, use_bias=False,
                    kernel_initializer=kzer, kernel_regularizer=reg
                ),  # linear
                keras.layers.Dense(
                    1, use_bias=False,
                    kernel_initializer=pos_init(seed_base + 1), kernel_regularizer=reg,
                    kernel_constraint=keras.constraints.NonNeg()
                ),  # exp
                keras.layers.Dense(
                    1, use_bias=False,
                    kernel_initializer=kzer, kernel_regularizer=reg
                ),  # quadratic linear
                keras.layers.Dense(
                    1, use_bias=False,
                    kernel_initializer=pos_init(seed_base + 2), kernel_regularizer=reg,
                    kernel_constraint=keras.constraints.NonNeg()
                ),  # quadratic exp
                # Note: the exp nonlinearity is applied in call()
            ]

        # give each invariant branch its own seed base
        self.bI1   = branch(10)
        self.bI2   = branch(20)
        self.bI4th = branch(30)
        self.bI4z  = branch(40)

        self.mixer = keras.layers.Dense(
            1, use_bias=False,
            kernel_constraint=keras.constraints.NonNeg(),
            kernel_regularizer=reg
        )

    def call(self, I1, I2, I4th, I4z, training=False):
        I1r = I1 - self.shift_I1
        I2r = I2 - self.shift_I2
        I4thr = I4th - self.shift_I4
        I4zr  = I4z  - self.shift_I4

        def safe_expm1(z):
            z = tf.clip_by_value(z, -EXP_CLIP, EXP_CLIP)
            return tf.math.expm1(z)

        def apply_branch(x, layers):
            t1 = layers[0](x)                         # lin
            t2 = safe_expm1(layers[1](x))            # exp
            x2 = tf.square(x)
            t3 = layers[2](x2)                        # quad lin
            t4 = safe_expm1(layers[3](x2))           # quad exp
            return tf.concat([t1, t2, t3, t4], axis=1)

        feat = tf.concat([
            apply_branch(I1r,  self.bI1),
            apply_branch(I2r,  self.bI2),
            apply_branch(I4thr,self.bI4th),
            apply_branch(I4zr, self.bI4z),
        ], axis=1)

        psi = self.mixer(feat)
        return tf.squeeze(psi, axis=1)

# -----------------------------------------------------------------------------
# Training (custom loops so we can take dΨ/dI with GradientTape)
# -----------------------------------------------------------------------------
def _make_dataset(X, y, batch):
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.astype(np.float32))
    ).shuffle(min(10000, max(1000, X.shape[0])), seed=SEED)\
     .batch(batch, drop_remainder=True)\
     .cache()\
     .prefetch(tf.data.AUTOTUNE)
    return ds

def _make_joint_dataset(X_th, y_th, X_z, y_z, batch, meta_th=None, meta_z=None):
    """
    Build a single dataset with NaN targets for the missing label and
    per-row ref/scale vectors for BOTH heads (placeholders where missing).
    """
    X_list, ytheta_list, yz_list = [], [], []
    rth_list, sth_list, rz_list, sz_list = [], [], [], []

    if X_th.shape[0] > 0:
        # rows coming from Pd
        X_list.append(X_th)
        ytheta_list.append(y_th)
        yz_list.append(np.full_like(y_th, np.nan))

        # theta refs/scales for these rows (real or placeholders)
        if meta_th is not None:
            rth_list.append(meta_th["ref"])
            sth_list.append(meta_th["scl"])
        else:
            rth_list.append(np.zeros_like(y_th))
            sth_list.append(np.ones_like(y_th))

        # z refs/scales placeholders for these Pd rows
        rz_list.append(np.zeros_like(y_th))
        sz_list.append(np.ones_like(y_th))

    if X_z.shape[0] > 0:
        # rows coming from Fl
        X_list.append(X_z)
        ytheta_list.append(np.full_like(y_z, np.nan))
        yz_list.append(y_z)

        # z refs/scales for these rows (real or placeholders)
        if meta_z is not None:
            rz_list.append(meta_z["ref"])
            sz_list.append(meta_z["scl"])
        else:
            rz_list.append(np.zeros_like(y_z))
            sz_list.append(np.ones_like(y_z))

        # theta refs/scales placeholders for these Fl rows
        rth_list.append(np.zeros_like(y_z))
        sth_list.append(np.ones_like(y_z))

    if not X_list:
        return None

    X      = np.concatenate(X_list,      axis=0).astype(np.float32)
    ytheta = np.concatenate(ytheta_list, axis=0).astype(np.float32)
    yz     = np.concatenate(yz_list,     axis=0).astype(np.float32)
    rth    = np.concatenate(rth_list,    axis=0).astype(np.float32)
    sth    = np.concatenate(sth_list,    axis=0).astype(np.float32)
    rz     = np.concatenate(rz_list,     axis=0).astype(np.float32)
    sz     = np.concatenate(sz_list,     axis=0).astype(np.float32)

    # optional cap in fast mode (keep all vectors in sync)
    if FAST_DEV and X.shape[0] > MAX_POINTS_PER_AGE:
        sel = np.linspace(0, X.shape[0] - 1, MAX_POINTS_PER_AGE, dtype=int)
        X, ytheta, yz, rth, sth, rz, sz = X[sel], ytheta[sel], yz[sel], rth[sel], sth[sel], rz[sel], sz[sel]

    ds = tf.data.Dataset.from_tensor_slices((X, ytheta, yz, rth, sth, rz, sz)) \
         .shuffle(min(10000, max(1000, X.shape[0])), seed=SEED) \
         .batch(BATCH, drop_remainder=True) \
         .cache() \
         .prefetch(tf.data.AUTOTUNE)
    return ds



@tf.function(jit_compile=False, experimental_relax_shapes=True)
def _compute_joint_grads(model, xb, ytheta, yz, rth, sth, rz, sz):
    lam_th = xb[:, 0:1]
    lam_z  = xb[:, 1:2]

    with tf.GradientTape() as tape_out:
        I1, I2, I4th, I4z = invariants_from_stretches(lam_th, lam_z)
        with tf.GradientTape() as tape_in:
            tape_in.watch([I1, I2, I4th, I4z])
            psi = model(I1, I2, I4th, I4z, training=True)

        dW_dI1, dW_dI2, dW_dI4th, dW_dI4z = tape_in.gradient(psi, [I1, I2, I4th, I4z])
        sig_th_hat = tf.squeeze(sigma_theta_from_derivs(lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4th), 1)
        sig_z_hat  = tf.squeeze(sigma_z_from_derivs(   lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4z),  1)

        m_th = tf.math.is_finite(ytheta)
        m_z  = tf.math.is_finite(yz)

        if NORM_STRESS:
            # normalized residuals per file: (ŷ - ref)/scale  vs  (y - ref)/scale
            th_hat_n = (tf.boolean_mask(sig_th_hat, m_th) - tf.boolean_mask(rth, m_th)) / tf.boolean_mask(sth, m_th)
            th_trg_n = (tf.boolean_mask(ytheta,   m_th) - tf.boolean_mask(rth, m_th)) / tf.boolean_mask(sth, m_th)
            loss_th = tf.reduce_mean(tf.square(th_hat_n - th_trg_n)) if tf.reduce_any(m_th) else 0.0

            z_hat_n  = (tf.boolean_mask(sig_z_hat, m_z) - tf.boolean_mask(rz, m_z)) / tf.boolean_mask(sz, m_z)
            z_trg_n  = (tf.boolean_mask(yz,       m_z) - tf.boolean_mask(rz, m_z)) / tf.boolean_mask(sz, m_z)
            loss_z = tf.reduce_mean(tf.square(z_hat_n - z_trg_n)) if tf.reduce_any(m_z) else 0.0
        else:
            loss_th = tf.reduce_mean(tf.square(tf.boolean_mask(sig_th_hat, m_th) - tf.boolean_mask(ytheta, m_th))) if tf.reduce_any(m_th) else 0.0
            loss_z  = tf.reduce_mean(tf.square(tf.boolean_mask(sig_z_hat,  m_z)  - tf.boolean_mask(yz,     m_z)))  if tf.reduce_any(m_z)  else 0.0

        # balance
        n_th = tf.reduce_sum(tf.cast(m_th, tf.float32))
        n_z  = tf.reduce_sum(tf.cast(m_z,  tf.float32))
        n_tot = tf.maximum(n_th + n_z, 1.0)
        w_th = tf.where(n_th > 0, n_tot / (2.0 * n_th), 0.0)
        w_z  = tf.where(n_z  > 0, n_tot / (2.0 * n_z),  0.0)
        loss = w_th * loss_th + w_z * loss_z

    grads = tape_out.gradient(loss, model.trainable_variables)
    return loss, grads, loss_th, loss_z


def train_joint(model, X_th, y_th, X_z, y_z, epochs, batch, lr, meta_th=None, meta_z=None):
    ds = _make_joint_dataset(X_th, y_th, X_z, y_z, batch, meta_th=meta_th, meta_z=meta_z)
    if ds is None:
        print("[SKIP] No data for joint training"); 
        return
    opt = keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=5.0)
    _ = opt.iterations
    best = float('inf'); patience = 12 if FAST_DEV else 20; tol = 1e-4; stale = 0
    E = (FAST_EPOCHS_TH if FAST_DEV else epochs)
    for ep in range(1, E+1):
        losses, losses_th, losses_z = [], [], []
        for xb, ytheta, yz, rth, sth, rz, sz in ds:
            loss, grads, lth, lz = _compute_joint_grads(model, xb, ytheta, yz, rth, sth, rz, sz)
            pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if pairs: opt.apply_gradients(pairs)
            losses.append(float(loss)); losses_th.append(float(lth)); losses_z.append(float(lz))
        avg = float(np.mean(losses)) if losses else np.nan
        if (ep % 20) == 0 or ep <= 5:
            print(f"[joint] epoch {ep}  loss={avg:.5f}  (θ={np.mean(losses_th):.5f}, z={np.mean(losses_z):.5f})")
        if avg + tol < best: best = avg; stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[joint] early stop at epoch {ep} (best={best:.5f})")
                break


def _ref_scale(y, method="quantile"):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0
    if method == "quantile":
        lo = np.quantile(y, QUANT_LO); hi = np.quantile(y, QUANT_HI)
        ref = lo; scale = max(hi - lo, EPS_SCALE)
    elif method == "zscore":
        ref = float(np.mean(y)); scale = max(float(np.std(y)), EPS_SCALE)
    elif method == "minmax":
        ref = float(np.min(y)); scale = max(float(np.max(y) - ref), EPS_SCALE)
    else:
        ref, scale = 0.0, 1.0
    return float(ref), float(scale)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def quick_scatter(x, y, yhat, xlabel, ylabel, title, out_png):
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=10, label="data", alpha=0.6)
    plt.scatter(x, yhat, s=8, label="fit")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def save_weights(model, outdir):
    import csv
    os.makedirs(outdir, exist_ok=True)

    # --- small helpers ------------------------------------------------------
    def _scalar(var):
        v = var.numpy()
        return float(v.reshape(-1)[0])

    def _branch_coeffs(branch_layers):
        # return [a_lin, a_exp, a_quadlin, a_quadexp] from the 4 Dense(1) kernels
        return [_scalar(L.kernel) for L in branch_layers]

    names4 = ["lin", "exp", "quad_lin", "quad_exp"]
    branches = [
        ("I1",      model.bI1,   float(model.shift_I1.numpy())),
        ("I2",      model.bI2,   float(model.shift_I2.numpy())),
        ("I4theta", model.bI4th, float(model.shift_I4.numpy())),
        ("I4z",     model.bI4z,  float(model.shift_I4.numpy())),
    ]

    # Mixer maps 16 features -> 1
    mixer = model.mixer.kernel.numpy().reshape(-1)
    feature_order = (
        [f"I1_{n}" for n in names4] +
        [f"I2_{n}" for n in names4] +
        [f"I4theta_{n}" for n in names4] +
        [f"I4z_{n}" for n in names4]
    )

    # Collect branch scalars (pre-activations) and compose with mixer
    branch_rows = []  # for Psi_weights.txt (invariant, feature, weight)
    inv_params  = {}  # for suggested closed form
    mix_idx = 0
    for inv_name, br, shift in branches:
        a_lin, a_exp, a_qlin, a_qexp = _branch_coeffs(br)
        # mixer weights for this invariant's four features
        b_lin, b_exp, b_qlin, b_qexp = mixer[mix_idx:mix_idx+4]
        mix_idx += 4

        # rows for human readable table
        for feat_name, a in zip(names4, [a_lin, a_exp, a_qlin, a_qexp]):
            branch_rows.append((inv_name, feat_name, a))

        # Compose final Ψ contribution coefficients:
        #  x := (I - shift)
        #  Ψ_inv(x) = α*x + β*(exp(γ*x)-1) + δ*x^2 + ε*(exp(ζ*x^2)-1)
        α = b_lin  * a_lin
        β = b_exp
        γ = a_exp
        δ = b_qlin * a_qlin
        ε = b_qexp
        ζ = a_qexp

        inv_params[inv_name] = dict(shift=shift, alpha=α, beta=β, gamma=γ,
                                    delta=δ, eps=ε, zeta=ζ)

    # ---------- Write detailed weights (as before) --------------------------
    txt_path = os.path.join(outdir, "Psi_weights.txt")
    with open(txt_path, "w") as f:
        f.write("Ψ-Net weights (by invariant/feature)\n")
        f.write("====================================\n\n")
        f.write("Invariant reference shifts used inside model:\n")
        for inv_name, _, shift in branches:
            f.write(f"  shift_{inv_name} = {shift}\n")
        f.write("\nBranch weights (pre-activation scalars):\n")
        last = None
        for inv, feat, w in branch_rows:
            if inv != last:
                f.write(f"\n[{inv}]\n"); last = inv
            f.write(f"  {feat:9s}: {w:+.6e}\n")
        f.write("\nMixer weights (feature -> Ψ):\n")
        for name, w in zip(feature_order, mixer):
            f.write(f"  {name:14s} -> {w:+.6e}\n")

    csv_path = os.path.join(outdir, "Psi_mixer_features.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["feature", "weight"])
        for name, w in zip(feature_order, mixer):
            writer.writerow([name, f"{w:.8e}"])

    # ---------- Suggested closed-form Ψ export ------------------------------
    suggest_txt = os.path.join(outdir, "Psi_suggested_form.txt")
    with open(suggest_txt, "w") as f:
        f.write("Suggested strain-energy density Ψ(I1, I2, I4θ, I4z)\n")
        f.write("===================================================\n\n")
        f.write("Let x1 = I1 - s1,  x2 = I2 - s2,  xθ = I4θ - s4,  xz = I4z - s4\n")
        f.write("with s1 = %.6g, s2 = %.6g, s4 = %.6g.\n\n" % (
            inv_params["I1"]["shift"], inv_params["I2"]["shift"], inv_params["I4theta"]["shift"]))
        f.write("Then\n")
        f.write("  Ψ = Ψ1(x1) + Ψ2(x2) + Ψθ(xθ) + Ψz(xz)\n")
        f.write("where each term has the form\n")
        f.write("  Ψ•(x) = α• x + β• (e^{γ• x} - 1) + δ• x^2 + ε• (e^{ζ• x^2} - 1)\n\n")

        for key, label in [("I1","Ψ1"), ("I2","Ψ2"), ("I4theta","Ψθ"), ("I4z","Ψz")]:
            p = inv_params[key]
            f.write(f"{label}(x) with shift s={p['shift']:.6g}:\n")
            f.write("  α=%+.6e,  β=%+.6e,  γ=%+.6e,  δ=%+.6e,  ε=%+.6e,  ζ=%+.6e\n\n" %
                    (p["alpha"], p["beta"], p["gamma"], p["delta"], p["eps"], p["zeta"]))

        f.write("LaTeX-ish one-liner:\n")
        def term(inv, sym):
            p = inv_params[inv]
            return (f"{p['alpha']:+.6e}({sym}-{p['shift']:.6g})"
                    f" + {p['beta']:+.6e}(e^{{{p['gamma']:.6e}({sym}-{p['shift']:.6g})}}-1)"
                    f" + {p['delta']:+.6e}({sym}-{p['shift']:.6g})^2"
                    f" + {p['eps']:+.6e}(e^{{{p['zeta']:.6e}({sym}-{p['shift']:.6g})^2}}-1)")
        f.write("  Ψ = " + " + ".join([
            term("I1","I1"), term("I2","I2"), term("I4theta","I4θ"), term("I4z","I4z")
        ]) + "\n")

    # A tiny NumPy implementation you can import/use directly
    suggest_py = os.path.join(outdir, "Psi_suggested_form.py")
    with open(suggest_py, "w") as g:
        g.write("# Auto-generated Ψ from trained PsiNet\n")
        g.write("import numpy as np\n\n")
        for inv, p in inv_params.items():
            g.write(f"# {inv}\n")
            for k, v in p.items():
                g.write(f"{inv}_{k} = {v:.16e}\n")
            g.write("\n")
        g.write("""def psi(I1, I2, I4th, I4z):
    I1 = np.asarray(I1, dtype=float)
    I2 = np.asarray(I2, dtype=float)
    I4th = np.asarray(I4th, dtype=float)
    I4z  = np.asarray(I4z,  dtype=float)
    x1  = I1 - I1_shift
    x2  = I2 - I2_shift
    xth = I4theta_shift  # place-holder; we store shift as I4theta_shift
""")
        # write correctly named shifts
        g.seek(0, os.SEEK_END)
        # build the body with current params
        p1  = inv_params["I1"];  p2  = inv_params["I2"]
        pth = inv_params["I4theta"]; pz = inv_params["I4z"]
        # Fix variable names for readability
        g.write("    xth = I4th - I4theta_shift\n")
        g.write("    xz  = I4z  - I4z_shift\n\n")
        def py_term(var, p):
            return (f"({p['alpha']:.16e})*{var} + "
                    f"({p['beta']:.16e})*np.expm1(({p['gamma']:.16e})*{var}) + "
                    f"({p['delta']:.16e})*({var}**2) + "
                    f"({p['eps']:.16e})*np.expm1(({p['zeta']:.16e})*({var}**2))")
        g.write("    return (" + " + ".join([
            py_term("x1", p1), py_term("x2", p2), py_term("xth", pth), py_term("xz", pz)
        ]) + ")\n")

    # ---------- Keep checkpoint + meta --------------------------------------
    ckpt = tf.train.Checkpoint(psinet=model)
    ckpt.write(os.path.join(outdir, "Psi_ckpt"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"reg": REG_KIND, "pen": REG_PEN}, f, indent=2)

    print(f"[WEIGHTS] Saved:\n"
          f"  - {txt_path}\n  - {csv_path}\n"
          f"  - {suggest_txt}\n  - {suggest_py}")


def parse_cli_or_defaults():
    # Keep VS Code defaults unless CLUSTER_RUN=1
    if os.environ.get("CLUSTER_RUN", "0") != "1":
        return dict(
            data_root=DATA_ROOT, out_root=OUT_ROOT,
            epochs_th=EPOCHS_TH, epochs_z=EPOCHS_Z,
            batch=BATCH, lr=LR, reg=REG_KIND, pen=REG_PEN, seed=SEED,
            fast=FAST_DEV
        )
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=DATA_ROOT)
    p.add_argument('--out_root',  default=OUT_ROOT)
    p.add_argument('--epochs_th', type=int, default=EPOCHS_TH)
    p.add_argument('--epochs_z',  type=int, default=EPOCHS_Z)
    p.add_argument('--batch',     type=int, default=BATCH)
    p.add_argument('--lr',        type=float, default=LR)
    p.add_argument('--reg',       choices=['L1','L2'], default=REG_KIND)
    p.add_argument('--pen',       type=float, default=REG_PEN)
    p.add_argument('--seed',      type=int, default=SEED)
    p.add_argument('--fast',      action='store_true')
    return vars(p.parse_args())

# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------
def main():

    cfg = parse_cli_or_defaults()
    global DATA_ROOT, OUT_ROOT, EPOCHS_TH, EPOCHS_Z, BATCH, LR, REG_KIND, REG_PEN, SEED, FAST_DEV
    DATA_ROOT = cfg['data_root']; OUT_ROOT = cfg['out_root']
    EPOCHS_TH = cfg['epochs_th']; EPOCHS_Z = cfg['epochs_z']
    BATCH = cfg['batch']; LR = cfg['lr']
    REG_KIND = cfg['reg']; REG_PEN = cfg['pen']; SEED = cfg['seed']
    FAST_DEV = cfg['fast']
    tf.random.set_seed(SEED); np.random.seed(SEED)
    ages = find_ages(DATA_ROOT)
    print(f"Found ages: {ages}")
    if not ages:
        print(f"[ERROR] No ages found under {DATA_ROOT}. Expected e.g. data/3weeks")
        return

    for age in ages:
        age_dir = os.path.join(DATA_ROOT, age)
        print(f"Processing age: {age} ({age_dir})")
        blob = parse_csvs_for_age(age_dir)
        X_th, y_th, X_z, y_z = build_dataset(blob)

        outdir = os.path.join(OUT_ROOT, age)
        os.makedirs(outdir, exist_ok=True)

        if X_th.shape[0] == 0 and X_z.shape[0] == 0:
            print(f"[SKIP] {age}: no usable data.")
            continue

        # ==== B) Multi-GPU / single-GPU strategy scope ====
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        use_mirrored = (os.environ.get("USE_MIRRORED", "0") == "1") and (num_gpus >= 2)
        strategy = tf.distribute.MirroredStrategy() if use_mirrored else tf.distribute.get_strategy()
        print(f"[STRATEGY] Using {'MirroredStrategy' if use_mirrored else 'DefaultStrategy'} (GPUs: {num_gpus})")
        with strategy.scope():
            # Build ONE shared Psi model
            model = PsiNet(REG_KIND, REG_PEN)
            # Force variable creation
            _lam_th = tf.ones((1,1), dtype=tf.float32)
            _lam_z  = tf.ones((1,1), dtype=tf.float32)
            I1, I2, I4th, I4z = invariants_from_stretches(_lam_th, _lam_z)
            _ = model(I1, I2, I4th, I4z, training=False)

            # ==== C) Resume from checkpoint if available ====
            ckpt_path = os.path.join(outdir, "Psi_ckpt")
            ckpt = tf.train.Checkpoint(psinet=model)
            if tf.io.gfile.exists(ckpt_path + ".index"):
                print(f"[RESUME] Loading checkpoint from {ckpt_path}")
                ckpt.restore(ckpt_path).expect_partial()
            # Build meta vectors that align with X_th / X_z stacking order
            meta_th = None; meta_z = None
            if len(blob["pd"]) > 0:
                refs = []; scls = []
                for s in blob["pd"]:
                    mask = np.isfinite(s["lam_th"]) & np.isfinite(s["lam_z"]) & np.isfinite(s["sig_th"])
                    n = int(np.sum(mask))
                    refs.append(np.full(n, s["th_ref"], dtype=np.float32))
                    scls.append(np.full(n, s["th_scl"], dtype=np.float32))
                if refs:
                    meta_th = {"ref": np.concatenate(refs, axis=0), "scl": np.concatenate(scls, axis=0)}

            if len(blob["fl"]) > 0:
                refs = []; scls = []
                for s in blob["fl"]:
                    mask = np.isfinite(s["lam_th"]) & np.isfinite(s["lam_z"]) & np.isfinite(s["sig_z"])
                    n = int(np.sum(mask))
                    refs.append(np.full(n, s["z_ref"], dtype=np.float32))
                    scls.append(np.full(n, s["z_scl"], dtype=np.float32))
                if refs:
                    meta_z = {"ref": np.concatenate(refs, axis=0), "scl": np.concatenate(scls, axis=0)}

            print(f"==> {age}: joint training (Pd+Fl): Nθ={X_th.shape[0]}  Nz={X_z.shape[0]}")
            train_joint(model, X_th, y_th, X_z, y_z, max(EPOCHS_TH, EPOCHS_Z), BATCH, LR,
                        meta_th=meta_th, meta_z=meta_z)
            # After training, extra collapsed Pd plot
            if len(blob["pd"]) > 0:
                plt.figure(figsize=(6,5))
                for s in blob["pd"]:
                    lam_th = s["lam_th"]; lam_z = s["lam_z"]; y = s["sig_th"]
                    r, sc = s["th_ref"], s["th_scl"]
                    # model prediction on this file
                    lam_th_tf = tf.convert_to_tensor(lam_th[:,None], tf.float32)
                    lam_z_tf  = tf.convert_to_tensor(lam_z[:,None],  tf.float32)
                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch([lam_th_tf, lam_z_tf])
                        I1,I2,I4th,I4z = invariants_from_stretches(lam_th_tf, lam_z_tf)
                        psi = model(I1,I2,I4th,I4z, training=False)
                    dW_dI1 = tape.gradient(psi, I1); dW_dI2 = tape.gradient(psi, I2); dW_dI4th = tape.gradient(psi, I4th)
                    yhat = sigma_theta_from_derivs(lam_th_tf, lam_z_tf, dW_dI1, dW_dI2, dW_dI4th)[:,0].numpy()
                    # normalized
                    yN    = (y    - r)/sc
                    yhatN = (yhat - r)/sc
                    plt.scatter(lam_th, yN, s=8, alpha=0.35)
                    plt.scatter(lam_th, yhatN, s=6)
                plt.xlabel(r"$\lambda_\theta$"); plt.ylabel(r"normalized $\sigma_\theta$")
                plt.title(f"{age} – Pd (collapsed by per-file quantiles)")
                plt.tight_layout(); plt.savefig(os.path.join(outdir, "fit_pd_collapsed.png"), dpi=180); plt.close()

            # Save final checkpoint
            ckpt.write(ckpt_path)

        # Evaluate / plots
        if X_th.shape[0] > 0:
            lam_th = tf.convert_to_tensor(X_th[:,0:1], dtype=tf.float32)
            lam_z  = tf.convert_to_tensor(X_th[:,1:2], dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([lam_th, lam_z])
                I1, I2, I4th, I4z = invariants_from_stretches(lam_th, lam_z)
                psi = model(I1, I2, I4th, I4z, training=False)
            dW_dI1   = tape.gradient(psi, I1)
            dW_dI2   = tape.gradient(psi, I2)
            dW_dI4th = tape.gradient(psi, I4th)
            sig_hat  = sigma_theta_from_derivs(lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4th)[:,0].numpy()
            quick_scatter(X_th[:,0], y_th, sig_hat, r"$\lambda_\theta$", r"$\sigma_\theta$ (kPa)",
                          f"{age} – Pd (all)", os.path.join(outdir, "fit_pd_theta_vs_lambda_theta.png"))

        if X_z.shape[0] > 0:
            lam_th = tf.convert_to_tensor(X_z[:,0:1], dtype=tf.float32)
            lam_z  = tf.convert_to_tensor(X_z[:,1:2], dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([lam_th, lam_z])
                I1, I2, I4th, I4z = invariants_from_stretches(lam_th, lam_z)
                psi = model(I1, I2, I4th, I4z, training=False)
            dW_dI1  = tape.gradient(psi, I1)
            dW_dI2  = tape.gradient(psi, I2)
            dW_dI4z = tape.gradient(psi, I4z)
            sig_hat = sigma_z_from_derivs(lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4z)[:,0].numpy()
            quick_scatter(X_z[:,1], y_z, sig_hat, r"$\lambda_z$", r"$\sigma_z$ (kPa)",
                          f"{age} – Fl (all)", os.path.join(outdir, "fit_fl_sigma_z_vs_lambda_z.png"))

        save_weights(model, outdir)
        print(f"[DONE] Saved to {outdir}")

if __name__ == "__main__":
    main()
