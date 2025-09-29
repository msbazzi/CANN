#!/usr/bin/env python3
# IVCANN_discovered_C_stable.py
# Discover Ψ(C) for juvenile lamb IVC with cross terms and stabilized training.

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
LR        = 5e-3              # base learning rate
REG_KIND  = "L2"              # "L1" or "L2"
REG_PEN   = 1e-4              # light L2 calms features/mixer
SEED      = 42                # reproducibility
LMIN      = 1e-3              # min stretch
LMAX      = 5.0               # max stretch (very conservative)
EXP_CLIP  = 12.0              # cap exp pre-activations to avoid overflow
THETA_LOSS_BOOST = 3.0        # ↑ weight on θ head so it competes with z

# ==== FAST DEV TOGGLES (local quick tests) ===================================
FAST_DEV = True               # set False for full training
MAX_POINTS_PER_FILE = 300     # subsample from each CSV (per test/specimen)
MAX_POINTS_PER_AGE  = 6000    # cap total points per age (after stacking)
FAST_EPOCHS_TH      = 400     # override EPOCHS_TH when FAST_DEV
FAST_EPOCHS_Z       = 400     # override EPOCHS_Z  when FAST_DEV

# (Augmentation hooks reserved; currently not used)
AUG_ON=False; AUG_DUP=2; AUG_MIXUP=True; AUG_ALPHA=0.2; AUG_JITTER_STD=0.01; AUG_LMIN=0.6; AUG_LMAX=2.0

tf.random.set_seed(SEED)
USE_XLA = os.environ.get("USE_XLA", "0") == "1"
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

        req = {"lambda_theta","lambda_z"}
        if req - cols:
            print(f"[WARN] {csv}: missing {req - cols}. Found: {sorted(cols)}"); 
            continue

        lam_th = df["lambda_theta"].astype(float).to_numpy()
        lam_z  = df["lambda_z"].astype(float).to_numpy()

        if kind == "pd":
            if "sigma_theta_kpa" not in cols:
                print(f"[WARN] {csv}: missing 'sigma_theta_kpa'. Found: {sorted(cols)}"); 
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
                pd_samples.append(dict(specimen=spec, level=lvl, lam_th=lam_th_m, lam_z=lam_z_m, sig_th=sig_th_m, path=csv))
        else:
            if "sigma_z_kpa" not in cols:
                print(f"[WARN] {csv}: missing 'sigma_z_kpa'. Found: {sorted(cols)}"); 
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
                fl_samples.append(dict(specimen=spec, level=lvl, lam_th=lam_th_m, lam_z=lam_z_m, sig_z=sig_z_m, path=csv))

        print(f"[INFO] Parsed {csv} for {spec} ({kind}, lvl={lvl}) → N={mask.sum()}")

    return {"pd": pd_samples, "fl": fl_samples}

def build_dataset(age_blob):
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

    X_th = np.concatenate(X_th, axis=0) if X_th else np.zeros((0,2))
    y_th = np.concatenate(y_th, axis=0) if y_th else np.zeros((0,))
    X_z  = np.concatenate(X_z,  axis=0) if X_z  else np.zeros((0,2))
    y_z  = np.concatenate(y_z,  axis=0) if y_z  else np.zeros((0,))

    if FAST_DEV and X_th.shape[0] > MAX_POINTS_PER_AGE:
        sel = np.linspace(0, X_th.shape[0]-1, MAX_POINTS_PER_AGE, dtype=int)
        X_th = X_th[sel]; y_th = y_th[sel]
    if FAST_DEV and X_z.shape[0] > MAX_POINTS_PER_AGE:
        sel = np.linspace(0, X_z.shape[0]-1, MAX_POINTS_PER_AGE, dtype=int)
        X_z = X_z[sel]; y_z = y_z[sel]
    return X_th, y_th, X_z, y_z

# -----------------------------------------------------------------------------
# Physics: C = F^T F and stresses from dΨ/dC with σr=0 (membrane)
# -----------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def C_from_stretches(lam_th, lam_z):
    lam_th = tf.clip_by_value(lam_th, LMIN, LMAX)
    lam_z  = tf.clip_by_value(lam_z,  LMIN, LMAX)
    lam_r  = 1.0 / (lam_th * lam_z)  # incompressible J=1
    C_rr = lam_r**2
    C_tt = lam_th**2
    C_zz = lam_z**2
    return C_rr, C_tt, C_zz, lam_r

@tf.function(reduce_retracing=True)
def sigma_theta_from_dPsi_dC(lam_th, lam_r, dW_dC_rr, dW_dC_tt):
    return 2.0 * ((lam_th**2) * dW_dC_tt - (lam_r**2) * dW_dC_rr)

@tf.function(reduce_retracing=True)
def sigma_z_from_dPsi_dC(lam_z, lam_r, dW_dC_rr, dW_dC_zz):
    return 2.0 * ((lam_z**2)  * dW_dC_zz - (lam_r**2) * dW_dC_rr)

# -----------------------------------------------------------------------------
# Model: Ψ(C_rr, C_tt, C_zz, C_rr*C_tt, C_rr*C_zz, C_tt*C_zz) with richer atoms
# -----------------------------------------------------------------------------
def regularizer(kind, pen):
    if pen <= 0: return None
    return keras.regularizers.l2(pen) if kind == "L2" else keras.regularizers.l1(pen)

class PsiNetWithCross(keras.Model):
    """Ψ(C) with unary branches + pairwise cross branches and 6 atoms per branch."""
    def __init__(self, reg_kind="L1", reg_pen=0.0):
        super().__init__()
        reg = regularizer(reg_kind, reg_pen)
        self.shift_C = tf.constant(1.0, dtype=tf.float32)  # center at identity

        def kzer():
            return keras.initializers.Zeros()
        def pos_init(seed):
            return keras.initializers.RandomUniform(minval=0.0, maxval=0.1, seed=SEED + seed)

        def branch(seed_base: int):
            # 6 atoms: lin, expm1, quad-lin, quad-expm1, cubic-lin, softplus
            return [
                keras.layers.Dense(1, use_bias=False, kernel_initializer=kzer(), kernel_regularizer=reg),               # lin
                keras.layers.Dense(1, use_bias=False, kernel_initializer=pos_init(seed_base+1), kernel_regularizer=reg,
                                   kernel_constraint=keras.constraints.NonNeg()),                                       # exp
                keras.layers.Dense(1, use_bias=False, kernel_initializer=kzer(), kernel_regularizer=reg),               # quad lin
                keras.layers.Dense(1, use_bias=False, kernel_initializer=pos_init(seed_base+2), kernel_regularizer=reg,
                                   kernel_constraint=keras.constraints.NonNeg()),                                       # quad exp
                keras.layers.Dense(1, use_bias=False, kernel_initializer=kzer(), kernel_regularizer=reg),               # cubic lin
                keras.layers.Dense(1, use_bias=False, kernel_initializer=pos_init(seed_base+3), kernel_regularizer=reg,
                                   kernel_constraint=keras.constraints.NonNeg()),                                       # softplus preact
            ]

        # unary branches
        self.bCrr = branch(10)
        self.bCtt = branch(20)
        self.bCzz = branch(30)
        # pairwise branches
        self.bCrrCtt = branch(40)
        self.bCrrCzz = branch(50)
        self.bCttCzz = branch(60)

        # mixer: 6 branches × 6 features = 36 → 1 (shape inferred)
        self.mixer = keras.layers.Dense(
            1, use_bias=False, kernel_constraint=keras.constraints.NonNeg(), kernel_regularizer=reg
        )

    def call(self, C_rr, C_tt, C_zz, training=False):
        # center variables at identity; center pairwise products too
        Crr = C_rr - self.shift_C
        Ctt = C_tt - self.shift_C
        Czz = C_zz - self.shift_C
        CrrCtt = (C_rr * C_tt) - self.shift_C
        CrrCzz = (C_rr * C_zz) - self.shift_C
        CttCzz = (C_tt * C_zz) - self.shift_C

        def safe_expm1(z):
            z = tf.clip_by_value(z, -EXP_CLIP, EXP_CLIP)
            return tf.math.expm1(z)

        def apply_branch(x, layers):
            t1 = layers[0](x)                     # lin
            t2 = safe_expm1(layers[1](x))        # exp
            x2 = tf.square(x)
            t3 = layers[2](x2)                    # quad lin
            t4 = safe_expm1(layers[3](x2))       # quad exp
            x3 = x * x2
            t5 = layers[4](x3)                    # cubic lin
            t6 = tf.nn.softplus(layers[5](x))    # softplus on x
            return tf.concat([t1, t2, t3, t4, t5, t6], axis=1)

        feat = tf.concat([
            apply_branch(Crr,    self.bCrr),
            apply_branch(Ctt,    self.bCtt),
            apply_branch(Czz,    self.bCzz),
            apply_branch(CrrCtt, self.bCrrCtt),
            apply_branch(CrrCzz, self.bCrrCzz),
            apply_branch(CttCzz, self.bCttCzz),
        ], axis=1)

        psi = self.mixer(feat)
        return tf.squeeze(psi, axis=1)

# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def _make_theta_only_dataset(X_th, y_th, batch):
    if X_th.shape[0] == 0:
        return None
    X = X_th.astype(np.float32)
    ytheta = y_th.astype(np.float32)
    yz = np.full_like(ytheta, np.nan, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X, ytheta, yz)) \
         .shuffle(min(10000, max(1000, X.shape[0])), seed=SEED) \
         .batch(BATCH, drop_remainder=True) \
         .cache() \
         .prefetch(tf.data.AUTOTUNE)
    return ds

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

def masked_huber(y_pred, y_true, mask, delta=1.0):
    """Huber loss on masked residuals; safe under @tf.function."""
    def compute():
        r = tf.boolean_mask(y_pred - y_true, mask)
        abs_r = tf.abs(r)
        quad  = tf.minimum(abs_r, delta)
        return tf.reduce_mean(0.5 * tf.square(quad) + delta * (abs_r - quad))
    return tf.cond(tf.reduce_any(mask), compute, lambda: tf.constant(0.0, tf.float32))

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_steps):
        super().__init__()
        self.base_lr = tf.convert_to_tensor(base_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps  = tf.cast(tf.maximum(1, decay_steps), tf.float32)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        step_after = tf.maximum(0.0, step - self.warmup_steps)
        cosine = 0.5 * self.base_lr * (1.0 + tf.cos(np.pi * step_after / self.decay_steps))
        return tf.where(step < self.warmup_steps, warm, cosine)

# Globals for variance normalization (read inside @tf.function)
STD_TH = tf.constant(1.0, tf.float32)
STD_Z  = tf.constant(1.0, tf.float32)

def reference_bc_loss(model, weight=0.1):
    """Penalty to enforce σθ=σz≈0 at λθ=λz=1 (identity)."""
    lam_th = tf.constant([[1.0]], tf.float32)
    lam_z  = tf.constant([[1.0]], tf.float32)
    C_rr, C_tt, C_zz, lam_r = C_from_stretches(lam_th, lam_z)
    with tf.GradientTape() as tape_in:
        tape_in.watch([C_rr, C_tt, C_zz])
        psi = model(C_rr, C_tt, C_zz, training=True)
    dW_dC_rr, dW_dC_tt, dW_dC_zz = tape_in.gradient(psi, [C_rr, C_tt, C_zz])
    sig_th0 = sigma_theta_from_dPsi_dC(lam_th, lam_r, dW_dC_rr, dW_dC_tt)
    sig_z0  = sigma_z_from_dPsi_dC(   lam_z,  lam_r, dW_dC_rr, dW_dC_zz)
    return weight * 0.5 * (tf.square(sig_th0) + tf.square(sig_z0))

@tf.function(jit_compile=False, experimental_relax_shapes=True)
def _compute_joint_grads(model, xb, ytheta, yz):
    lam_th = xb[:, 0:1]; lam_z = xb[:, 1:2]
    with tf.GradientTape() as tape_out:
        C_rr, C_tt, C_zz, lam_r = C_from_stretches(lam_th, lam_z)
        with tf.GradientTape() as tape_in:
            tape_in.watch([C_rr, C_tt, C_zz])
            psi = model(C_rr, C_tt, C_zz, training=True)
        dW_dC_rr, dW_dC_tt, dW_dC_zz = tape_in.gradient(psi, [C_rr, C_tt, C_zz])

        sig_th_hat = sigma_theta_from_dPsi_dC(lam_th, lam_r, dW_dC_rr, dW_dC_tt)
        sig_z_hat  = sigma_z_from_dPsi_dC(   lam_z,  lam_r, dW_dC_rr, dW_dC_zz)
        sig_th_hat = tf.squeeze(sig_th_hat, 1); sig_z_hat = tf.squeeze(sig_z_hat, 1)

        m_th = tf.math.is_finite(ytheta); m_z = tf.math.is_finite(yz)
        loss_th = masked_huber(sig_th_hat, ytheta, m_th, delta=1.0) / (STD_TH**2)
        loss_z  = masked_huber(sig_z_hat,  yz,     m_z,  delta=1.0) / (STD_Z**2)

        # balance by counts, then boost θ head
        n_th  = tf.reduce_sum(tf.cast(m_th, tf.float32))
        n_z   = tf.reduce_sum(tf.cast(m_z,  tf.float32))
        n_tot = tf.maximum(n_th + n_z, 1.0)
        w_th  = tf.where(n_th > 0, n_tot / (2.0 * n_th), 0.0)
        w_z   = tf.where(n_z  > 0, n_tot / (2.0 * n_z),  0.0)

        loss = THETA_LOSS_BOOST * w_th * loss_th + w_z * loss_z

        # reference (identity) stress-free condition
        loss += reference_bc_loss(model, weight=0.1)

    grads = tape_out.gradient(loss, model.trainable_variables)
    return loss, grads, loss_th, loss_z

def train_joint(model, X_th, y_th, X_z, y_z, epochs, batch, lr, std_th=1.0, std_z=1.0):
    # build datasets
    ds_theta = _make_theta_only_dataset(X_th, y_th, batch)
    ds = _make_joint_dataset(X_th, y_th, X_z, y_z, batch)
    if ds is None:
        print("[SKIP] No data for joint training"); 
        return

    # derive schedule from dataset (reiterable)
    steps_per_epoch = max(1, sum(1 for _ in ds))
    total_epochs = (FAST_EPOCHS_TH if FAST_DEV else epochs)
    total_steps  = steps_per_epoch * total_epochs
    warmup_epochs = (2 if FAST_DEV else 5)
    warmup_steps  = warmup_epochs * steps_per_epoch

    # optimizer (created BEFORE warmup so warmup uses it)
    lr_schedule = WarmupCosine(base_lr=lr, warmup_steps=warmup_steps,
                               decay_steps=max(1, total_steps - warmup_steps))
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=3.0)
    _ = opt.iterations

    # θ-only warmup
    theta_warm_epochs = (5 if FAST_DEV else 12)
    if ds_theta is not None and theta_warm_epochs > 0:
        print(f"[warmup θ] {theta_warm_epochs} epochs")
        for _ in range(theta_warm_epochs):
            for xb, ytheta, yz in ds_theta:
                loss, grads, _, _ = _compute_joint_grads(model, xb, ytheta, yz)
                pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
                if pairs:
                    opt.apply_gradients(pairs)

    # main joint training loop
    best = float('inf'); patience = (12 if FAST_DEV else 20); tol = 1e-4; stale = 0
    for ep in range(1, total_epochs + 1):
        losses, losses_th, losses_z = [], [], []
        for xb, ytheta, yz in ds:
            loss, grads, lth, lz = _compute_joint_grads(model, xb, ytheta, yz)
            pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if pairs:
                opt.apply_gradients(pairs)
            losses.append(float(loss)); losses_th.append(float(lth)); losses_z.append(float(lz))

        avg  = float(np.mean(losses)) if losses else np.nan
        avgT = float(np.mean([v for v in losses_th if np.isfinite(v)])) if losses_th else np.nan
        avgZ = float(np.mean([v for v in losses_z  if np.isfinite(v)]))  if losses_z  else np.nan

        if (ep % 20) == 0 or ep <= 5:
            print(f"[joint] epoch {ep}  loss={avg:.5f}  (θ={avgT:.5f}, z={avgZ:.5f})")

        if avg + tol < best:
            best = avg; stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[joint] early stop at epoch {ep} (best={best:.5f})")
                break

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

    def _scalar(var):
        v = var.numpy()
        return float(v.reshape(-1)[0])

    names = ["lin", "exp", "quad_lin", "quad_exp", "cubic_lin", "softplus"]
    branches = [
        ("C_rr",       model.bCrr),
        ("C_tt",       model.bCtt),
        ("C_zz",       model.bCzz),
        ("C_rr*C_tt",  model.bCrrCtt),
        ("C_rr*C_zz",  model.bCrrCzz),
        ("C_tt*C_zz",  model.bCttCzz),
    ]

    # collect branch scalars (each Dense has kernel shape (1,1))
    branch_rows = []
    for nm, br in branches:
        for feat_name, layer in zip(names, br):
            branch_rows.append((nm, feat_name, _scalar(layer.kernel)))

    mixer_w = model.mixer.kernel.numpy().reshape(-1)  # 36 features
    feature_order = (
        [f"C_rr_{n}" for n in names] +
        [f"C_tt_{n}" for n in names] +
        [f"C_zz_{n}" for n in names] +
        [f"C_rr*C_tt_{n}" for n in names] +
        [f"C_rr*C_zz_{n}" for n in names] +
        [f"C_tt*C_zz_{n}" for n in names]
    )

    # --- write text report ---
    txt_path = os.path.join(outdir, "Psi_weights.txt")
    with open(txt_path, "w") as f:
        f.write("Ψ-Net weights (C-based with cross terms, 6 atoms/branch)\n")
        f.write("=======================================================\n\n")
        f.write(f"shift_C = {float(model.shift_C.numpy())}\n\n")
        f.write("Branch weights (pre-activation scalars):\n")
        last_nm = None
        for nm, feat, w in branch_rows:
            if nm != last_nm:
                f.write(f"\n[{nm}]\n"); last_nm = nm
            f.write(f"  {feat:11s}: {w:+.6e}\n")
        f.write("\nMixer weights (feature -> Ψ):\n")
        for name, w in zip(feature_order, mixer_w):
            f.write(f"  {name:18s} -> {w:+.6e}\n")

    # --- write CSV for mixer ---
    import csv
    csv_path = os.path.join(outdir, "Psi_mixer_features.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["feature", "weight"])
        for name, w in zip(feature_order, mixer_w):
            writer.writerow([name, f"{w:.8e}"])

    # checkpoint + meta
    ckpt = tf.train.Checkpoint(psinet=model)
    ckpt.write(os.path.join(outdir, "Psi_ckpt"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({
            "reg": REG_KIND, "pen": REG_PEN,
            "psi_form": "C-based+cross(6-atoms)",
            "theta_loss_boost": THETA_LOSS_BOOST,
            "exp_clip": EXP_CLIP
        }, f, indent=2)

    print(f"[WEIGHTS] Saved readable weights to:\n  - {txt_path}\n  - {csv_path}")

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

# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------
def main():
    cfg = parse_cli_or_defaults()
    global DATA_ROOT, OUT_ROOT, EPOCHS_TH, EPOCHS_Z, BATCH, LR, REG_KIND, REG_PEN, SEED, FAST_DEV, STD_TH, STD_Z
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

        # per-age loss normalization constants (as tf constants for graph use)
        std_th = float(np.std(y_th)) if y_th.size else 1.0
        std_z  = float(np.std(y_z))  if y_z.size  else 1.0
        STD_TH = tf.constant(max(std_th, 1e-6), tf.float32)
        STD_Z  = tf.constant(max(std_z,  1e-6), tf.float32)
        globals()['STD_TH'], globals()['STD_Z'] = STD_TH, STD_Z

        # ==== strategy scope ====
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        use_mirrored = (os.environ.get("USE_MIRRORED", "0") == "1") and (num_gpus >= 2)
        strategy = tf.distribute.MirroredStrategy() if use_mirrored else tf.distribute.get_strategy()
        print(f"[STRATEGY] Using {'MirroredStrategy' if use_mirrored else 'DefaultStrategy'} (GPUs: {num_gpus})")
        with strategy.scope():
            model = PsiNetWithCross(REG_KIND, REG_PEN)
            # Force variable creation
            _lam_th = tf.ones((1,1), dtype=tf.float32); _lam_z = tf.ones((1,1), dtype=tf.float32)
            C_rr, C_tt, C_zz, _ = C_from_stretches(_lam_th, _lam_z)
            _ = model(C_rr, C_tt, C_zz, training=False)

            # resume if available
            ckpt_path = os.path.join(outdir, "Psi_ckpt")
            ckpt = tf.train.Checkpoint(psinet=model)
            if tf.io.gfile.exists(ckpt_path + ".index"):
                print(f"[RESUME] Loading checkpoint from {ckpt_path}")
                ckpt.restore(ckpt_path).expect_partial()

            print(f"==> {age}: joint training (Pd+Fl): Nθ={X_th.shape[0]}  Nz={X_z.shape[0]}")
            train_joint(model, X_th, y_th, X_z, y_z, max(EPOCHS_TH, EPOCHS_Z), BATCH, LR,
                        std_th=std_th, std_z=std_z)

            # save final ckpt
            ckpt.write(ckpt_path)

        # Evaluate / plots
        if X_th.shape[0] > 0:
            lam_th = tf.convert_to_tensor(X_th[:,0:1], dtype=tf.float32)
            lam_z  = tf.convert_to_tensor(X_th[:,1:2], dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([lam_th, lam_z])
                C_rr, C_tt, C_zz, lam_r = C_from_stretches(lam_th, lam_z)
                psi = model(C_rr, C_tt, C_zz, training=False)
            dW_dC_rr = tape.gradient(psi, C_rr)
            dW_dC_tt = tape.gradient(psi, C_tt)
            sig_hat  = sigma_theta_from_dPsi_dC(lam_th, lam_r, dW_dC_rr, dW_dC_tt)[:,0].numpy()
            quick_scatter(X_th[:,0], y_th, sig_hat, r"$\lambda_\theta$", r"$\sigma_\theta$ (kPa)",
                          f"{age} – Pd (all)", os.path.join(outdir, "fit_pd_theta_vs_lambda_theta.png"))

        if X_z.shape[0] > 0:
            lam_th = tf.convert_to_tensor(X_z[:,0:1], dtype=tf.float32)
            lam_z  = tf.convert_to_tensor(X_z[:,1:2], dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([lam_th, lam_z])
                C_rr, C_tt, C_zz, lam_r = C_from_stretches(lam_th, lam_z)
                psi = model(C_rr, C_tt, C_zz, training=False)
            dW_dC_rr = tape.gradient(psi, C_rr)
            dW_dC_zz = tape.gradient(psi, C_zz)
            sig_hat  = sigma_z_from_dPsi_dC(lam_z, lam_r, dW_dC_rr, dW_dC_zz)[:,0].numpy()
            quick_scatter(X_z[:,1], y_z, sig_hat, r"$\lambda_z$", r"$\sigma_z$ (kPa)",
                          f"{age} – Fl (all)", os.path.join(outdir, "fit_fl_sigma_z_vs_lambda_z.png"))

        save_weights(model, outdir)
        print(f"[DONE] Saved to {outdir}")

if __name__ == "__main__":
    main()
