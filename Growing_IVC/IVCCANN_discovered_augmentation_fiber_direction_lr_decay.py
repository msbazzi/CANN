#!/usr/bin/env python3
# IVCANN_discovered_augment.py
# Discover a strain-energy density function (Ψ) for juvenile lamb IVC
# Folder layout:
#   data/<age>/<specimen>_(pd|fl)<level>.csv
# CSV columns expected:
#   lambda_theta, sigma_theta_kPa, lambda_z, sigma_z_kPa

import os, re, glob, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
BATCH     = 32                # batch size
LR        = 4e-3              # learning rate (safer for aug)
REG_KIND  = "L2"              # "L1" or "L2"
REG_PEN   = 0.2              # regularization strength
SEED      = 42                # reproducibility (ish)

# Physics/num stability
LMIN = 1e-5      # min stretch
LMAX = 10.0       # max stretch (very conservative)
EXP_CLIP = 15.0  # cap exp pre-activations to avoid overflow

# ==== FAST DEV TOGGLES (local quick tests) ===================================
FAST_DEV = True               # set False for full training
MAX_POINTS_PER_FILE = 300     # subsample from each CSV (per test/specimen)
MAX_POINTS_PER_AGE  = 6000    # cap total points per age (after stacking)
FAST_EPOCHS_TH      = 400     # override EPOCHS_TH when FAST_DEV
FAST_EPOCHS_Z       = 400     # override EPOCHS_Z  when FAST_DEV

# --- Augmentation (offline, before building tf.data) ---
AUG_ON         = False     # turn augmentation on/off
AUG_DUP        = 5        # how many jittered duplicates to add (0 = none)
AUG_MIXUP      = True     # add a mixup copy of the dataset
AUG_ALPHA      = 0.2      # beta(alpha, alpha) for mixup
AUG_JITTER_STD = 0.01     # gaussian jitter on (lambda_theta, lambda_z)
AUG_LMIN       = 0.6      # clamp augmented lambdas to this min
AUG_LMAX       = 2.0      # clamp augmented lambdas to this max

# --- Oriented-fiber option ---
USE_FIBER_ANGLE = True      # turn on the new invariant I4phi
FIBER_INIT_DEG  = 45.0      # starting guess for φ (degrees, 0..90)
LAMBDA_PHI_REG  = 0.0       # e.g. 1e-3 to softly bias φ toward FIBER_INIT_DEG

# --- LR schedule (optional) ---
USE_LR_DECAY   = True      # turn on/off LR schedule
DECAY_KIND     = "exp"     # "exp" | "cosine" | "piecewise"
DECAY_EPOCHS   = 5        # decay period in epochs (for exp/cosine)
DECAY_RATE     = 0.5       # multiplier each period (exp only, e.g., 0.5 halves LR)
PIECEWISE_MILESTONES = [20, 40]   # epochs at which to step down (piecewise)
PIECEWISE_VALUES     = [1.0, 0.3, 0.1]  # LR factors vs base (len = len(milestones)+1)
MIN_LR        = 1e-5       # clamp LR to this minimum


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
    """
    pd_samples, fl_samples = [], []
    for csv in glob.glob(os.path.join(age_dir, "*.csv")):
        fname = os.path.basename(csv)
        m = re.match(r"(?P<spec>.+)[_.](?P<kind>pd|fl)(?P<lvl>\d+)\.csv$", fname, flags=re.IGNORECASE)
        if not m:
            m = re.match(r"(?P<spec>.+)_(?P<kind>pd|fl)(?P<lvl>\d+)\.csv$", fname, flags=re.IGNORECASE)
        if not m:
            continue

        spec = m.group("spec"); kind = m.group("kind").lower(); lvl = int(m.group("lvl"))

        df = pd.read_csv(csv)
        df.columns = [c.strip().lower() for c in df.columns]
        cols = set(df.columns)

        req = {"lambda_theta", "lambda_z"}
        missing = req - cols
        if missing:
            print(f"[WARN] {csv}: missing required columns {missing}. Found: {sorted(cols)}")
            continue

        lam_th = df["lambda_theta"].astype(float).to_numpy()
        lam_z  = df["lambda_z"].astype(float).to_numpy()

        if kind == "pd":
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
                pd_samples.append(dict(
                    specimen=spec, level=lvl,
                    lam_th=lam_th_m, lam_z=lam_z_m,
                    sig_th=sig_th_m, path=csv
                ))
        else:
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
                fl_samples.append(dict(
                    specimen=spec, level=lvl,
                    lam_th=lam_th_m, lam_z=lam_z_m,
                    sig_z=sig_z_m, path=csv
                ))

        print(f"[INFO] Parsed {csv} for {spec} ({kind}, lvl={lvl}) → N={mask.sum()}")

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
# Augmentation helpers (offline, before tf.data)
# -----------------------------------------------------------------------------
def _augment_arrays(X, y,
                    dup=AUG_DUP,
                    jitter_std=AUG_JITTER_STD,
                    mixup=AUG_MIXUP,
                    alpha=AUG_ALPHA,
                    lmin=AUG_LMIN,
                    lmax=AUG_LMAX,
                    seed=SEED):
    if X.shape[0] == 0:
        return X, y

    rng = np.random.default_rng(seed)
    X_out = [X.astype(np.float32)]
    y_out = [y.astype(np.float32)]

    # small Gaussian jitter on lambda's
    for _ in range(max(0, int(dup))):
        j = rng.normal(0.0, jitter_std, X.shape).astype(np.float32)
        Xj = X.astype(np.float32) + j
        Xj[:, 0] = np.clip(Xj[:, 0], lmin, lmax)
        Xj[:, 1] = np.clip(Xj[:, 1], lmin, lmax)
        X_out.append(Xj); y_out.append(y.astype(np.float32))

    # mixup within the same modality
    if mixup and X.shape[0] >= 2:
        n = X.shape[0]
        i1 = rng.integers(0, n, size=n)
        i2 = rng.integers(0, n, size=n)
        lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
        Xm = lam[:, None] * X[i1].astype(np.float32) + (1.0 - lam)[:, None] * X[i2].astype(np.float32)
        ym = lam * y[i1].astype(np.float32) + (1.0 - lam) * y[i2].astype(np.float32)
        Xm[:, 0] = np.clip(Xm[:, 0], lmin, lmax)
        Xm[:, 1] = np.clip(Xm[:, 1], lmin, lmax)
        X_out.append(Xm); y_out.append(ym)

    X_aug = np.concatenate(X_out, axis=0).astype(np.float32)
    y_aug = np.concatenate(y_out, axis=0).astype(np.float32)
    return X_aug, y_aug

def maybe_augment(X_th, y_th, X_z, y_z):
    if not AUG_ON:
        return X_th, y_th, X_z, y_z
    if X_th.shape[0] > 0:
        X_th, y_th = _augment_arrays(X_th, y_th)
    if X_z.shape[0] > 0:
        X_z,  y_z  = _augment_arrays(X_z,  y_z)
    return X_th, y_th, X_z, y_z

# -----------------------------------------------------------------------------
# Physics: invariants and stresses
# -----------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def invariants_from_stretches(lam_th, lam_z):
    lam_th = tf.clip_by_value(lam_th, LMIN, LMAX)
    lam_z  = tf.clip_by_value(lam_z,  LMIN, LMAX)
    lam_r = 1.0 / (lam_th * lam_z)
    I1 = lam_th**2 + lam_z**2 + lam_r**2
    I2 = lam_th**2 * lam_z**2 + lam_th**2 * lam_r**2 + lam_z**2 * lam_r**2
    I4th = lam_th**2
    I4z  = lam_z**2
    return I1, I2, I4th, I4z
@tf.function(reduce_retracing=True)
# replace the old I4_phi/I8 helpers with these 4FF invariants
@tf.function(reduce_retracing=True)
def I4_fam_a(lam_th, lam_z, phi):
    # family A: ±φ around circumferential
    c = tf.cos(phi); s = tf.sin(phi)
    return lam_th**2 * c*c + lam_z**2 * s*s

@tf.function(reduce_retracing=True)
def I4_fam_b(lam_th, lam_z, phi):
    # family B: ±(90°-φ) around axial (orthogonal to family A)
    c = tf.cos(phi); s = tf.sin(phi)
    return lam_th**2 * s*s + lam_z**2 * c*c

@tf.function(reduce_retracing=True)
def I8_phi_from_stretches(lam_th, lam_z, phi):
    # for symmetric ±φ families (no shear deformation), cross-invariant
    return lam_th**2 * tf.cos(phi)**2 - lam_z**2 * tf.sin(phi)**2

@tf.function(reduce_retracing=True)
def sigma_theta_from_derivs(lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4th):
    lam_th = tf.clip_by_value(lam_th, LMIN, LMAX)
    lam_z  = tf.clip_by_value(lam_z,  LMIN, LMAX)
    lam_r = 1.0 / (lam_th * lam_z + 1e-8)
    return 2.0 * (
        dW_dI1 * (lam_th**2 - lam_r**2) +
        dW_dI2 * (lam_z**2 * (lam_th**2 - lam_r**2)) +
        dW_dI4th * (lam_th**2)
    )

@tf.function(reduce_retracing=True)
def sigma_z_from_derivs(lam_th, lam_z, dW_dI1, dW_dI2, dW_dI4z):
    lam_th = tf.clip_by_value(lam_th, LMIN, LMAX)
    lam_z  = tf.clip_by_value(lam_z,  LMIN, LMAX)
    lam_r = 1.0 / (lam_th * lam_z + 1e-8)
    return 2.0 * (
        dW_dI1 * (lam_z**2 - lam_r**2) +
        dW_dI2 * (lam_th**2 * (lam_z**2 - lam_r**2)) +
        dW_dI4z * (lam_z**2)
    )

# -----------------------------------------------------------------------------
# Model: invariant-based Ψ(I1,I2,I4θ,I4z)
# -----------------------------------------------------------------------------
def regularizer(kind, pen):
    if pen <= 0: return None
    return keras.regularizers.l2(pen) if kind == "L2" else keras.regularizers.l1(pen)

class PsiNet(keras.Model):
    def __init__(self, reg_kind="L1", reg_pen=0.0):
        super().__init__()
        reg = regularizer(reg_kind, reg_pen)
        self.shift_I1 = tf.constant(3.0, tf.float32)
        self.shift_I2 = tf.constant(3.0, tf.float32)
        self.shift_I4 = tf.constant(1.0, tf.float32)

        kzer = keras.initializers.Zeros()
        def pos_init(seed): return keras.initializers.RandomUniform(0.0, 0.1, seed=SEED+seed)
        def branch(seed):
            return [
                keras.layers.Dense(1, use_bias=False, kernel_initializer=kzer,          kernel_regularizer=reg),
                keras.layers.Dense(1, use_bias=False, kernel_initializer=pos_init(seed+1), kernel_regularizer=reg, kernel_constraint=keras.constraints.NonNeg()),
                keras.layers.Dense(1, use_bias=False, kernel_initializer=kzer,          kernel_regularizer=reg),
                keras.layers.Dense(1, use_bias=False, kernel_initializer=pos_init(seed+2), kernel_regularizer=reg, kernel_constraint=keras.constraints.NonNeg()),
            ]

        self.bI1   = branch(10)
        self.bI2   = branch(20)
        self.bI4th = branch(30)
        self.bI4z  = branch(40)

        # 4-fiber family: two extra branches for I4a (±φ) and I4b (±(90°−φ))
        self.bI4a  = branch(50)
        self.bI4b  = branch(60)

        # trainable fiber angle φ ∈ (0, π/2)
        phi0 = np.deg2rad(FIBER_INIT_DEG)
        u0 = np.log(phi0/(0.5*np.pi - phi0))
        self.phi_u = tf.Variable(u0, dtype=tf.float32, trainable=True, name="phi_u")

        self.mixer = keras.layers.Dense(1, use_bias=False,
                                        kernel_constraint=keras.constraints.NonNeg(),
                                        kernel_regularizer=reg)

    def phi(self):
        return (0.5*np.pi) * tf.math.sigmoid(self.phi_u)

    def call(self, I1, I2, I4th, I4z, I4a, I4b, training=False):
        def safe_expm1(z):
            z = tf.clip_by_value(z, -EXP_CLIP, EXP_CLIP)
            return tf.math.expm1(z)

        def apply_branch(x, layers):
            # zero-slope features at x=0
            f1 = layers[0](x) * x                            # ~ w * x^2
            z2 = layers[1](x); f2 = safe_expm1(z2) - z2      # exp(w x)-1 - w x
            x2 = tf.square(x)
            f3 = layers[2](x2)                               # w * x^2
            z4 = layers[3](x2); f4 = safe_expm1(z4)          # exp(w x^2)-1
            return tf.concat([f1, f2, f3, f4], axis=1)

        I1r   = I1   - self.shift_I1
        I2r   = I2   - self.shift_I2
        I4thr = I4th - self.shift_I4
        I4zr  = I4z  - self.shift_I4
        I4ar  = I4a  - self.shift_I4
        I4br  = I4b  - self.shift_I4

        feats = tf.concat([
            apply_branch(I1r,   self.bI1),
            apply_branch(I2r,   self.bI2),
            apply_branch(I4thr, self.bI4th),
            apply_branch(I4zr,  self.bI4z),
            apply_branch(I4ar,  self.bI4a),
            apply_branch(I4br,  self.bI4b),
        ], axis=1)

        psi = self.mixer(feats)
        return tf.squeeze(psi, axis=1)



# -----------------------------------------------------------------------------
# Training (custom loops so we can take dΨ/dI with GradientTape)
# -----------------------------------------------------------------------------
def _make_joint_dataset(X_th, y_th, X_z, y_z, batch):
    X_list, ytheta_list, yz_list = [], [], []
    if X_th.shape[0] > 0:
        X_list.append(X_th); ytheta_list.append(y_th); yz_list.append(np.full_like(y_th, np.nan))
    if X_z.shape[0] > 0:
        X_list.append(X_z);  ytheta_list.append(np.full_like(y_z, np.nan)); yz_list.append(y_z)
    if not X_list: return None

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    ytheta = np.concatenate(ytheta_list, axis=0).astype(np.float32)
    yz = np.concatenate(yz_list, axis=0).astype(np.float32)

    if FAST_DEV and X.shape[0] > MAX_POINTS_PER_AGE:
        sel = np.linspace(0, X.shape[0]-1, MAX_POINTS_PER_AGE, dtype=int)
        X, ytheta, yz = X[sel], ytheta[sel], yz[sel]

    ds = tf.data.Dataset.from_tensor_slices((X, ytheta, yz)) \
         .shuffle(min(10000, max(1000, X.shape[0])), seed=SEED) \
         .batch(BATCH, drop_remainder=True) \
         .cache() \
         .prefetch(tf.data.AUTOTUNE)
    return ds

@tf.function(jit_compile=False, experimental_relax_shapes=True)
@tf.function(jit_compile=False, experimental_relax_shapes=True)
@tf.function(jit_compile=False, experimental_relax_shapes=True)
def _compute_joint_grads(model, xb, ytheta, yz):
    """
    xb:      [batch,2] -> [lambda_theta, lambda_z]
    ytheta:  σθ targets with NaNs where not available
    yz:      σz targets with NaNs where not available

    returns: total_loss, grads(list), loss_th, loss_z
    """
    # stretches
    lam_th = tf.clip_by_value(xb[:, 0:1], LMIN, LMAX)
    lam_z  = tf.clip_by_value(xb[:, 1:2], LMIN, LMAX)
    lam_r  = 1.0 / (lam_th * lam_z + 1e-8)

    phi = model.phi()  # φ ∈ (0, π/2)

    with tf.GradientTape() as tape_out:
        # ---------------- invariants ----------------
        I1, I2, I4th, I4z = invariants_from_stretches(lam_th, lam_z)
        I4a = I4_fam_a(lam_th, lam_z, phi)   # family A: ±φ (around circumferential)
        I4b = I4_fam_b(lam_th, lam_z, phi)   # family B: ±(90°−φ) (around axial)

        # dΨ/dI* via inner tape
        with tf.GradientTape() as tape_in:
            tape_in.watch([I1, I2, I4th, I4z, I4a, I4b])
            psi = model(I1, I2, I4th, I4z, I4a, I4b, training=True)

        dW_dI1, dW_dI2, dW_dI4th, dW_dI4z, dW_dI4a, dW_dI4b = tape_in.gradient(
            psi, [I1, I2, I4th, I4z, I4a, I4b]
        )

        # ---------------- stresses ----------------
        # isotropic parts
        sig_th_iso = 2.0 * (
            dW_dI1 * (lam_th**2 - lam_r**2) +
            dW_dI2 * (lam_z**2  * (lam_th**2 - lam_r**2))
        )
        sig_z_iso  = 2.0 * (
            dW_dI1 * (lam_z**2 - lam_r**2) +
            dW_dI2 * (lam_th**2 * (lam_z**2  - lam_r**2))
        )

        # anisotropic parts for two symmetric families in each set (±) → factor 4
        c2 = tf.cos(phi)**2
        s2 = tf.sin(phi)**2
        sig_th_aniso = 4.0 * ( dW_dI4a * (lam_th**2) * c2 + dW_dI4b * (lam_th**2) * s2 )
        sig_z_aniso  = 4.0 * ( dW_dI4a * (lam_z**2)  * s2 + dW_dI4b * (lam_z**2)  * c2 )

        sig_th_hat = tf.squeeze(sig_th_iso + sig_th_aniso, axis=1)
        sig_z_hat  = tf.squeeze(sig_z_iso  + sig_z_aniso,  axis=1)

        # ---------------- data loss (masked, balanced) ----------------
        m_th = tf.math.is_finite(ytheta)
        m_z  = tf.math.is_finite(yz)

        loss_th = tf.reduce_mean(
            tf.square(tf.boolean_mask(sig_th_hat, m_th) - tf.boolean_mask(ytheta, m_th))
        ) if tf.reduce_any(m_th) else 0.0

        loss_z = tf.reduce_mean(
            tf.square(tf.boolean_mask(sig_z_hat, m_z) - tf.boolean_mask(yz, m_z))
        ) if tf.reduce_any(m_z) else 0.0

        # balance heads by available sample counts
        n_th  = tf.reduce_sum(tf.cast(m_th, tf.float32))
        n_z   = tf.reduce_sum(tf.cast(m_z,  tf.float32))
        n_tot = tf.maximum(n_th + n_z, 1.0)
        w_th  = tf.where(n_th > 0.0, n_tot / (2.0 * n_th), 0.0)
        w_z   = tf.where(n_z  > 0.0, n_tot / (2.0 * n_z),  0.0)

        loss = w_th * loss_th + w_z * loss_z

        # ---------------- optional priors ----------------
        # (a) prior on φ near initial guess
        if LAMBDA_PHI_REG > 0.0:
            phi0 = np.deg2rad(FIBER_INIT_DEG)
            loss += LAMBDA_PHI_REG * tf.square(phi - phi0)

        # (b) stress-free prior near (λθ,λz) ≈ (1,1)
        S0_W  = 1e-3   # tune ~1e-4 .. 1e-2
        S0RAD = 0.03   # ±3% jitter around 1
        if S0_W > 0.0:
            n = tf.shape(lam_th)[0]
            u1 = 1.0 + S0RAD * tf.random.uniform((n,1), -1.0, 1.0, dtype=lam_th.dtype)
            u2 = 1.0 + S0RAD * tf.random.uniform((n,1), -1.0, 1.0, dtype=lam_th.dtype)

            I1r, I2r, I4thr, I4zr = invariants_from_stretches(u1, u2)
            I4ar = I4_fam_a(u1, u2, phi)
            I4br = I4_fam_b(u1, u2, phi)

            with tf.GradientTape() as t_ref:
                t_ref.watch([I1r, I2r, I4thr, I4zr, I4ar, I4br])
                psi_r = model(I1r, I2r, I4thr, I4zr, I4ar, I4br, training=True)

            d1, d2, d4t, d4z, d4a, d4b = t_ref.gradient(
                psi_r, [I1r, I2r, I4thr, I4zr, I4ar, I4br]
            )

            lam_rr = 1.0 / (u1 * u2 + 1e-8)
            sig_th_iso_r = 2.0*( d1*(u1**2 - lam_rr**2) + d2*(u2**2*(u1**2 - lam_rr**2)) )
            sig_z_iso_r  = 2.0*( d1*(u2**2 - lam_rr**2) + d2*(u1**2*(u2**2 - lam_rr**2)) )
            c2r, s2r = c2, s2  # same φ
            sig_th_aniso_r = 4.0*( d4a*(u1**2)*c2r + d4b*(u1**2)*s2r )
            sig_z_aniso_r  = 4.0*( d4a*(u2**2)*s2r + d4b*(u2**2)*c2r )

            s0 = tf.reduce_mean(tf.square(sig_th_iso_r + sig_th_aniso_r)) \
               + tf.reduce_mean(tf.square(sig_z_iso_r  + sig_z_aniso_r))
            loss += S0_W * s0

        # safety checks
        tf.debugging.assert_all_finite(loss,       "NaN/Inf in loss")
        tf.debugging.assert_all_finite(sig_th_hat, "NaN/Inf in sigma_theta_hat")
        tf.debugging.assert_all_finite(sig_z_hat,  "NaN/Inf in sigma_z_hat")

    # backprop to model params (including phi_u via phi)
    grads = tape_out.gradient(loss, model.trainable_variables)
    return loss, grads, loss_th, loss_z


def _save_loss_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ---- CSVs ----
    import pandas as pd
    df_ep = pd.DataFrame({
        "epoch": np.arange(1, len(history["epoch_loss"]) + 1, dtype=int),
        "loss": history["epoch_loss"],
        "loss_theta": history["epoch_loss_th"],
        "loss_z": history["epoch_loss_z"],
    })
    df_ep.to_csv(os.path.join(outdir, "loss_history_epoch.csv"), index=False)

    df_bt = pd.DataFrame({
        "iteration": np.arange(1, len(history["batch_loss"]) + 1, dtype=int),
        "loss": history["batch_loss"],
    })
    df_bt.to_csv(os.path.join(outdir, "loss_history_batch.csv"), index=False)

    # ---- Epoch plot ----
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(df_ep["epoch"], df_ep["loss"], label="total")
    if np.isfinite(df_ep["loss_theta"]).any():
        plt.plot(df_ep["epoch"], df_ep["loss_theta"], label="θ-head")
    if np.isfinite(df_ep["loss_z"]).any():
        plt.plot(df_ep["epoch"], df_ep["loss_z"], label="z-head")
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Training loss (per epoch)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve_epoch.png"), dpi=180)
    plt.close()

    # ---- Iteration (batch) plot ----
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(df_bt["iteration"], df_bt["loss"])
    plt.xlabel("iteration (batch)"); plt.ylabel("MSE loss"); plt.title("Training loss (per iteration)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve_iteration.png"), dpi=180)
    plt.close()

class CappedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Wrap any schedule and clamp it below by MIN_LR."""
    def __init__(self, base_schedule, min_lr):
        self.base = base_schedule
        self.min_lr = float(min_lr)
    def __call__(self, step):
        return tf.maximum(self.base(step), self.min_lr)
    def get_config(self):
        return {"base": self.base, "min_lr": self.min_lr}

def train_joint(model, X_th, y_th, X_z, y_z, epochs, batch, lr, outdir):
    ds = _make_joint_dataset(X_th, y_th, X_z, y_z, batch)
    if ds is None:
        print("[SKIP] No data for joint training")
        return

    # Steps per epoch (number of batches)
    steps_per_epoch = int(tf.data.experimental.cardinality(ds).numpy())
    if steps_per_epoch < 1:
        steps_per_epoch = 1

    # Build a schedule
    if USE_LR_DECAY:
        if DECAY_KIND == "exp":
            decay_steps = max(1, DECAY_EPOCHS * steps_per_epoch)
            base_sched = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=decay_steps,
                decay_rate=DECAY_RATE,
                staircase=True
            )
        elif DECAY_KIND == "cosine":
            decay_steps = max(1, DECAY_EPOCHS * steps_per_epoch)
            base_sched = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=lr,
                first_decay_steps=decay_steps
            )
        elif DECAY_KIND == "piecewise":
            # Convert epoch milestones to step milestones
            boundaries = [m * steps_per_epoch for m in PIECEWISE_MILESTONES]
            values = [lr * f for f in PIECEWISE_VALUES]
            base_sched = keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=boundaries, values=values
            )
        else:
            base_sched = lr
        lr_or_sched = CappedSchedule(base_sched, MIN_LR) if hasattr(base_sched, "__call__") else base_sched
    else:
        lr_or_sched = lr

    opt = keras.optimizers.legacy.Adam(learning_rate=lr_or_sched, clipnorm=5.0)
    _ = opt.iterations  # create optimizer variables

    history = {
        "batch_loss": [],
        "epoch_loss": [],
        "epoch_loss_th": [],
        "epoch_loss_z": [],
    }

    best = float('inf'); patience = 12 if FAST_DEV else 20; tol = 1e-4; stale = 0
    E = (FAST_EPOCHS_TH if FAST_DEV else epochs)

    for ep in range(1, E + 1):
        losses, losses_th, losses_z = [], [], []
        for xb, ytheta, yz in ds:
            loss, grads, lth, lz = _compute_joint_grads(model, xb, ytheta, yz)
            pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if pairs:
                opt.apply_gradients(pairs)

            # record per-iteration (batch) loss
            history["batch_loss"].append(float(loss))
            losses.append(float(loss))
            losses_th.append(float(lth))
            losses_z.append(float(lz))

        # epoch summaries
        avg  = float(np.mean(losses)) if losses else np.nan
        avgT = float(np.mean([v for v in losses_th if np.isfinite(v)])) if losses_th else np.nan
        avgZ = float(np.mean([v for v in losses_z  if np.isfinite(v)]))  if losses_z  else np.nan

        history["epoch_loss"].append(avg)
        history["epoch_loss_th"].append(avgT)
        history["epoch_loss_z"].append(avgZ)

        if (ep % 20) == 0 or ep <= 5:
            print(f"[joint] epoch {ep}  loss={avg:.5f}  (θ={avgT:.5f}, z={avgZ:.5f})")

        if avg + tol < best:
            best = avg; stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[joint] early stop at epoch {ep} (best={best:.5f})")
                break

    # Save curves at the end
    _save_loss_curves(history, outdir)

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
    os.makedirs(outdir, exist_ok=True)

    # --- helpers ---
    def _scalar(var):
        v = var.numpy()
        return float(v.reshape(-1)[0])

    # Branch list (include I4phi if the model has it)
    names4 = ["lin", "exp", "quad_lin", "quad_exp"]
    pretty = {"I1": "I1", "I2": "I2", "I4theta": "I4θ", "I4z": "I4z", "I4phi": "I4φ"}

    branches = [
        ("I1",      model.bI1,   model.shift_I1),
        ("I2",      model.bI2,   model.shift_I2),
        ("I4theta", model.bI4th, model.shift_I4),
        ("I4z",     model.bI4z,  model.shift_I4),
        ("I4a",     model.bI4a,  model.shift_I4),
        ("I4b",     model.bI4b,  model.shift_I4),
    ]
    if hasattr(model, "bI4phi"):  # optional oriented-fiber branch
        branches.append(("I4phi", model.bI4phi, model.shift_I4))

    # Collect branch scalars (each Dense is (1x1), applied to x or x^2 before/inside expm1)
    # Also build the feature order to align with the mixer kernel
    branch_rows = []  # list of (inv_key, feat_name, w_branch)
    feature_order = []
    for inv_key, br, _shift in branches:
        for feat_name, layer in zip(names4, br):
            w = _scalar(layer.kernel)
            branch_rows.append((inv_key, feat_name, w))
            feature_order.append(f"{inv_key}_{feat_name}")

    # Mixer maps concatenated 4*len(branches) features -> 1 Ψ
    mixer_w = model.mixer.kernel.numpy().reshape(-1)  # length should be 4 * len(branches)
    F_expected = 4 * len(branches)
    if mixer_w.shape[0] != F_expected:
        print(f"[WARN] mixer weight length {mixer_w.shape[0]} ≠ expected {F_expected}. "
              f"Proceeding with min length.")
        L = min(mixer_w.shape[0], F_expected)
        mixer_w = mixer_w[:L]
        feature_order = feature_order[:L]
        # also trim branch_rows for consistency in the text dump
        branch_rows = branch_rows[:L]

    # ------------ human-readable weight dump ------------
    txt_path = os.path.join(outdir, "Psi_weights.txt")
    with open(txt_path, "w") as f:
        f.write("Ψ-Net weights (by invariant/feature)\n")
        f.write("====================================\n\n")
        f.write("Invariant reference shifts used inside model:\n")
        f.write(f"  shift_I1 = {float(model.shift_I1.numpy())}\n")
        f.write(f"  shift_I2 = {float(model.shift_I2.numpy())}\n")
        f.write(f"  shift_I4 = {float(model.shift_I4.numpy())}\n")

        # Optional learned fiber angle (if present)
        phi_deg = None
        if hasattr(model, "phi"):
            try:
                phi_deg = float(model.phi().numpy()) * 180.0 / np.pi
                f.write(f"  (learned) phi  = {phi_deg:.3f} deg\n")
            except Exception:
                pass
        f.write("\n")

        f.write("Branch weights (pre-activation scalars):\n")
        last_inv = None
        for inv, feat, w in branch_rows:
            if inv != last_inv:
                f.write(f"\n[{pretty.get(inv, inv)}]\n")
                last_inv = inv
            f.write(f"  {feat:9s}: {w:+.6e}\n")

        f.write("\nMixer weights (feature → Ψ):\n")
        for name, w in zip(feature_order, mixer_w):
            f.write(f"  {name:14s} -> {w:+.6e}\n")

    # Also export mixer weights as CSV (feature, weight)
    import csv
    csv_path = os.path.join(outdir, "Psi_mixer_features.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["feature", "weight"])
        for name, w in zip(feature_order, mixer_w):
            writer.writerow([name, f"{w:.8e}"])

    # ------------ suggested Ψ form ------------
    # We print a readable expression showing how Ψ is built from the features.
    sug_path = os.path.join(outdir, "Psi_suggested.txt")
    with open(sug_path, "w") as f:
        f.write("Suggested strain-energy density Ψ (model structure)\n")
        f.write("====================================================\n\n")
        f.write("Define shifted invariants:\n")
        f.write("  x1 = I1 - shift_I1\n")
        f.write("  x2 = I2 - shift_I2\n")
        f.write("  xθ = I4θ - shift_I4\n")
        f.write("  xz = I4z - shift_I4\n")
        f.write("  x_a = I4±φ        - shift_I4\n")
        f.write("  x_b = I4±(90°−φ)  - shift_I4\n")

        f.write("Per-invariant feature maps used by the network:\n")
        f.write("  lin(x)       = w_lin * x\n")
        f.write("  exp(x)       = exp(w_exp * x) - 1\n")
        f.write("  quad_lin(x)  = w_quad_lin * x^2\n")
        f.write("  quad_exp(x)  = exp(w_quad_exp * x^2) - 1\n\n")

        f.write("The network computes\n")
        f.write("  Ψ(I1, I2, I4θ, I4z, I4±φ, I4±(90°−φ)) =\n")

        f.write("      Σ_over_invariants  Σ_over_features  [ m(feature) * feature_value ]\n")
        f.write("where m(feature) is the mixer weight for that feature and each feature_value\n")
        f.write("is one of the four maps above applied to the corresponding shifted invariant.\n\n")

        # Provide a fully expanded, numeric mapping line-by-line
        f.write("Expanded sum with numeric coefficients (one term per feature):\n")
        idx = 0
        for inv_key, br, shift in branches:
            inv_sym = pretty.get(inv_key, inv_key)
            for feat_name, layer in zip(names4, br):
                if idx >= len(mixer_w):  # safety
                    break
                w_branch = _scalar(layer.kernel)    # inside the feature
                m_mix    = float(mixer_w[idx])      # mixer weight
                if inv_key == "I1":
                    xsym = "x1"
                elif inv_key == "I2":
                    xsym = "x2"
                elif inv_key == "I4theta":
                    xsym = "xθ"
                elif inv_key == "I4z":
                    xsym = "xz"
                elif inv_key == "I4a":
                    xsym = "x_a"
                elif inv_key == "I4b":
                    xsym = "x_b"
                else:
                    xsym = inv_key  # fallback


                if feat_name == "lin":
                    term = f"{m_mix:+.6e} * ({w_branch:+.6e} * {xsym})"
                elif feat_name == "exp":
                    term = f"{m_mix:+.6e} * (exp({w_branch:+.6e} * {xsym}) - 1)"
                elif feat_name == "quad_lin":
                    term = f"{m_mix:+.6e} * ({w_branch:+.6e} * {xsym}^2)"
                else:  # quad_exp
                    term = f"{m_mix:+.6e} * (exp({w_branch:+.6e} * {xsym}^2) - 1)"

                f.write(f"  [{inv_sym:4s} • {feat_name:9s}]  {term}\n")
                idx += 1

        if phi_deg is not None:
            f.write("\nLearned fiber angle:\n")
            f.write(f"  φ ≈ {phi_deg:.3f} degrees\n")

        f.write("\nNotes:\n")
        f.write("• The overall form is convex-combinational through nonnegative mixer weights (if constrained),\n")
        f.write("  blending linear and exponential feature responses of each invariant.\n")
        f.write("• Any physical scaling/units are inherited from the training data.\n")

    # ------------ checkpoint + meta ------------
    ckpt = tf.train.Checkpoint(psinet=model)
    ckpt.write(os.path.join(outdir, "Psi_ckpt"))

    meta = {"reg": REG_KIND, "pen": REG_PEN}
    if 'phi_deg' in locals() and phi_deg is not None:
        meta["phi_deg"] = phi_deg
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[WEIGHTS] Saved readable weights to:")
    print(f"  - {txt_path}")
    print(f"  - {csv_path}")
    print(f"[FORM]    Saved suggested Ψ form to:")
    print(f"  - {sug_path}")

def parse_cli_or_defaults():
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

def predict_sigma_hat(model, X, mode="theta"):
    """
    mode: "theta" for σ_θ (Pd) or "z" for σ_z (Fl)
    X: array with columns [lambda_theta, lambda_z]
    returns: 1D numpy array of predicted stresses
    """
    lam_th = tf.convert_to_tensor(X[:, 0:1], tf.float32)
    lam_z  = tf.convert_to_tensor(X[:, 1:2], tf.float32)
    phi    = model.phi()

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([lam_th, lam_z])
        I1, I2, I4th, I4z = invariants_from_stretches(lam_th, lam_z)
        I4a = I4_fam_a(lam_th, lam_z, phi)
        I4b = I4_fam_b(lam_th, lam_z, phi)
        psi = model(I1, I2, I4th, I4z, I4a, I4b, training=False)

    dW_dI1, dW_dI2, dW_dI4th, dW_dI4z, dW_dI4a, dW_dI4b = tape.gradient(
        psi, [I1, I2, I4th, I4z, I4a, I4b]
    )
    del tape

    lam_r = 1.0 / (lam_th * lam_z + 1e-8)
    sig_th_iso = 2.0*( dW_dI1*(lam_th**2 - lam_r**2) + dW_dI2*(lam_z**2*(lam_th**2 - lam_r**2)) )
    sig_z_iso  = 2.0*( dW_dI1*(lam_z**2  - lam_r**2) + dW_dI2*(lam_th**2*(lam_z**2  - lam_r**2)) )

    c2 = tf.cos(phi)**2
    s2 = tf.sin(phi)**2
    sig_th_aniso = 4.0 * ( dW_dI4a * (lam_th**2) * c2 + dW_dI4b * (lam_th**2) * s2 )
    sig_z_aniso  = 4.0 * ( dW_dI4a * (lam_z**2)  * s2 + dW_dI4b * (lam_z**2)  * c2 )

    if mode == "theta":
        return (sig_th_iso + sig_th_aniso)[:, 0].numpy()
    else:
        return (sig_z_iso  + sig_z_aniso)[:, 0].numpy()


# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------
def main():
    # ---- tiny helper to version checkpoints by architecture ----
    def feature_count_for(model):
        branches = 4  # I1, I2, I4θ, I4z
        branches += int(hasattr(model, "bI4a"))
        branches += int(hasattr(model, "bI4b"))
        return 4 * branches  # 4 features per branch

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

        # ---- load & build raw datasets ----
        blob = parse_csvs_for_age(age_dir)
        X_th, y_th, X_z, y_z = build_dataset(blob)

        # ---- offline augmentation (if enabled) ----
        X_th, y_th, X_z, y_z = maybe_augment(X_th, y_th, X_z, y_z)
        print(f"[AUG] After augmentation: Nθ={len(y_th)}  Nz={len(y_z)}")

        outdir = os.path.join(OUT_ROOT, age)
        os.makedirs(outdir, exist_ok=True)

        if X_th.shape[0] == 0 and X_z.shape[0] == 0:
            print(f"[SKIP] {age}: no usable data.")
            continue

        # ---- strategy scope (CPU/1GPU by default) ----
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        use_mirrored = (os.environ.get("USE_MIRRORED", "0") == "1") and (num_gpus >= 2)
        strategy = tf.distribute.MirroredStrategy() if use_mirrored else tf.distribute.get_strategy()
        print(f"[STRATEGY] Using {'MirroredStrategy' if use_mirrored else 'DefaultStrategy'} (GPUs: {num_gpus})")

        with strategy.scope():
            # Build model and force variable creation
            model = PsiNet(REG_KIND, REG_PEN)
            _lam_th = tf.ones((1,1), dtype=tf.float32)
            _lam_z  = tf.ones((1,1), dtype=tf.float32)
            I1, I2, I4th, I4z = invariants_from_stretches(_lam_th, _lam_z)
            phi  = model.phi()
            I4a  = I4_fam_a(_lam_th, _lam_z, phi)
            I4b  = I4_fam_b(_lam_th, _lam_z, phi)
            _ = model(I1, I2, I4th, I4z, I4a, I4b, training=False)


            # ---- resume checkpoint (versioned by feature count) ----
            ckpt_feat = feature_count_for(model)   # e.g., 16 or 20
            ckpt_path = os.path.join(outdir, f"Psi_F{ckpt_feat}_ckpt")
            ckpt = tf.train.Checkpoint(psinet=model)

            if tf.io.gfile.exists(ckpt_path + ".index"):
                print(f"[RESUME] Loading checkpoint from {ckpt_path}")
                try:
                    ckpt.restore(ckpt_path).expect_partial()
                except Exception as e:
                    print(f"[RESUME] Shape mismatch while restoring {ckpt_path}; starting fresh. Details: {e}")

            # ---- train ----
            print(f"==> {age}: joint training (Pd+Fl): Nθ={X_th.shape[0]}  Nz={X_z.shape[0]}")
            train_joint(model, X_th, y_th, X_z, y_z, max(EPOCHS_TH, EPOCHS_Z), BATCH, LR, outdir)

            # ---- save checkpoint for this architecture ----
            ckpt.write(ckpt_path)

        # ---- Pd (σ_θ) plot ----
        if X_th.shape[0] > 0:
            sig_hat_th = predict_sigma_hat(model, X_th, mode="theta")
            quick_scatter(
                X_th[:,0], y_th, sig_hat_th,
                r"$\lambda_\theta$", r"$\sigma_\theta$ (kPa)",
                f"{age} – Pd (all)",
                os.path.join(outdir, "fit_pd_theta_vs_lambda_theta.png")
            )

        # ---- Fl (σ_z) plot ----
        if X_z.shape[0] > 0:
            sig_hat_z = predict_sigma_hat(model, X_z, mode="z")
            quick_scatter(
                X_z[:,1], y_z, sig_hat_z,
                r"$\lambda_z$", r"$\sigma_z$ (kPa)",
                f"{age} – Fl (all)",
                os.path.join(outdir, "fit_fl_sigma_z_vs_lambda_z.png")
            )

        # ---- exports (weights, suggested Ψ form, meta) ----
        save_weights(model, outdir)
        print(f"[DONE] Saved to {outdir}")


if __name__ == "__main__":
    main()
