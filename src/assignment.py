import numpy as np
from dataclasses import dataclass
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path("src/figures")


@dataclass
class Parameters:
  # from the assignment
  t_max: float = 25.0
  dt: float = 0.5
  stimulus_time: float = 10.0
  reward_center: float = 20.0
  reward_sigma: float = 1.0
  reward_scale: float = 0.5
  gamma: float = 1.0
  lr: float = 0.2
  memory_span: float = 12.0
  n_trials: int = 201
  representation: str = "tapped"       # "tapped" or "boxcar"
  reward_probability: float = 1.0      # fraction of trials with reward (Q5, Q6)
  seed: int = 42                       # RNG seed for partial reinforcement

  @property
  def n_mem(self) -> int:
    return int(round(self.memory_span / self.dt)) + 1

@dataclass
class State:
  t: "np.ndarray"
  stim: "np.ndarray"
  reward: "np.ndarray"
  V_storage: "np.ndarray"
  dV_storage: "np.ndarray"
  error_storage: "np.ndarray"
  state_matrix: "np.ndarray"
  phi: "np.ndarray"
  w: "np.ndarray"
  rewarded: "np.ndarray | None" = None  # bool array tracking which trials had reward


def initialize_stimulus(t: np.ndarray, t_stim: float, dt: float) -> np.ndarray:
  """
  Create a stimulus signal that is zero everywhere except for a single spike of
  amplitude 1.0 at time t_stim. The stimulus is represented as a 1-D array
  aligned with the time grid t, which is generated using the specified time step dt.
  """
  stim = np.zeros_like(t)
  idx = int(round(t_stim / dt))
  stim[idx] = 1.0
  return stim


def initialize_reward(t: np.ndarray, t_reward: float, sigma: float) -> np.ndarray:
  """
  Create a reward signal that follows a Gaussian distribution centered at
  t_reward with standard deviation sigma. The reward is represented as a
  1-D array aligned with the time grid t.
  """
  return np.exp(-((t - t_reward) ** 2) / (2.0 * sigma**2))


def intitialize_state_matrix(stim: np.ndarray, n_mem: int) -> np.ndarray:
  """Build the stimulus-by-time-delay state matrix S for all time steps.

  Constructs the matrix S used to represent the recent history of the stimulus
  signal. Each row corresponds to an absolute time step t, and each column
  corresponds to a relative delay tau (how many steps ago). The entry S[t, tau]
  records the stimulus level at time (t - tau), i.e. what the stimulus was tau
  steps in the past when viewed from time t.

  For the tapped delay line representation, each entry of S is used
  directly as a feature: phi_{i,tau}(S) = S_{i,tau}. For the boxcar
  representation (Q4), cumulative sums over the tau axis are taken
  instead.

  Boundary handling: at early time steps where t < tau, the stimulus
  history is unavailable (we cannot look before t = 0), so those
  entries remain zero. This is enforced by the inner loop bound
  min(ti, n_mem - 1).

  Args:
      stim: 1-D array of stimulus values at each time step, shape
            (steps,). For Q3, this is a single spike (1.0 at the
            stimulus time, 0.0 elsewhere).
      n_mem: Number of memory slots (columns), equal to
             int(T_mem / dt) + 1. With T_mem = 12 s and dt = 0.5 s
             this gives n_mem = 25, so S is a 1 x 25 matrix per
             time step (for a single stimulus).

  Returns:
      2-D array of shape (steps, n_mem) where steps = len(stim).
      Entry S[t, tau] = stim[t - tau] for tau <= t, and 0 otherwise.
  """
  steps = stim.shape[0]
  S = np.zeros((steps, n_mem), dtype=float)
  for ti in range(steps):
    for tau in range(min(ti, n_mem - 1) + 1):
      S[ti, tau] = stim[ti - tau]
  return S


def initialize_state(params: Parameters) -> State:
  t = np.arange(0, params.t_max + params.dt, params.dt)
  steps = t.shape[0]
  V_storage = np.zeros((params.n_trials, steps), dtype=float)
  dV_storage = np.zeros((params.n_trials, steps), dtype=float)
  error_storage = np.zeros((params.n_trials, steps), dtype=float)
  stim = initialize_stimulus(t, params.stimulus_time, params.dt)
  reward = params.reward_scale * initialize_reward(t, params.reward_center, params.reward_sigma)
  state_matrix = intitialize_state_matrix(stim, params.n_mem)

  # Select feature function based on representation type
  if params.representation == "tapped":
    feat_fn = tapped_features
  elif params.representation == "boxcar":
    feat_fn = boxcar_features
  else:
    raise ValueError(f"Unknown representation: {params.representation}")

  phi = np.zeros((steps, params.n_mem), dtype=float)
  for ti in range(steps):
    phi[ti] = feat_fn(state_matrix[ti])

  w = np.zeros(params.n_mem, dtype=float)

  return State(
    t=t,
    stim=stim,
    reward=reward,
    state_matrix=state_matrix,
    V_storage=V_storage,
    dV_storage=dV_storage,
    error_storage=error_storage,
    phi=phi,
    w=w,
    rewarded=np.zeros(params.n_trials, dtype=bool),
  )


def tapped_features(state_row: np.ndarray) -> np.ndarray:
  """Tapped delay line feature detector: phi_{i,tau}(S) = S_{i,tau}.

  Each feature simply returns the stimulus level at a specific delay.
  This is the identity mapping over the state row — the simplest
  possible feature representation for the state history.

  Args:
      state_row: A single row of the state matrix S, shape (n_mem,).

  Returns:
      A copy of the state row (copy to avoid aliasing issues).
  """
  return state_row.copy()


def boxcar_features(state_row: np.ndarray) -> np.ndarray:
  """Boxcar feature detector: phi^box_{i,tau}(S) = sum_{u=0}^{tau} S_{i,u}.

  Each feature returns the cumulative presence of the stimulus over
  a history window of length tau. This allows generalisation across
  nearby time delays.

  Args:
      state_row: A single row of the state matrix S, shape (n_mem,).

  Returns:
      Cumulative sum along the delay axis.
  """
  return np.cumsum(state_row)


def run_td(params: Parameters | None = None) -> State:
  """Run TD learning with the given parameters.

  Supports tapped delay line (Q3) and boxcar (Q4) representations,
  deterministic reward (Q3, Q4) and partial reinforcement (Q5, Q6).

  Args:
      params: Simulation parameters. Uses Q3 defaults if None.

  Returns:
      State object with V_storage, dV_storage, error_storage populated.
  """
  if params is None:
    params = Parameters()

  rng = np.random.default_rng(params.seed)
  s = initialize_state(params)
  steps = s.t.shape[0]

  for trial in range(params.n_trials):
    # Partial reinforcement: reward present with probability p
    is_rewarded = rng.random() < params.reward_probability
    s.rewarded[trial] = is_rewarded
    reward = s.reward if is_rewarded else np.zeros_like(s.reward)

    v = np.zeros(steps, dtype=float)
    dv = np.zeros(steps, dtype=float)
    delta = np.zeros(steps, dtype=float)
    v[0] = float(s.w @ s.phi[0])

    for ti in range(1, steps):
      v_curr = float(s.w @ s.phi[ti])
      dv[ti] = params.gamma * v_curr - v[ti - 1]
      delta[ti] = reward[ti - 1] + dv[ti]
      s.w += params.lr * delta[ti] * s.phi[ti - 1]
      v[ti] = float(s.w @ s.phi[ti])

    s.V_storage[trial] = v
    s.dV_storage[trial] = dv
    s.error_storage[trial] = delta

  return s

def _setup_style() -> None:
  mpl.rcParams.update({
    "font.family": "monospace",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.4,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
  })


def _save(fig: plt.Figure, subdir: str, name: str) -> None:
  out = FIG_DIR / subdir
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / name, dpi=300, bbox_inches="tight")
  plt.close(fig)


def plot_stim_reward(s: State, subdir: str = "question-3", name: str = "q3a_stim_reward.png") -> None:
  """Plot stimulus and reward as functions of time."""
  fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 3.6),
                           gridspec_kw={"hspace": 0.12})

  axes[0].plot(s.t, s.stim, lw=1.6)
  axes[0].set_ylabel("Stimulus  y(t)")
  axes[0].set_ylim(-0.05, 1.15)
  axes[0].set_yticks([0, 1])
  axes[0].grid(True)

  axes[1].plot(s.t, s.reward, lw=1.6)
  axes[1].set_ylabel("Reward  r(t)")
  axes[1].set_xlabel("Time (s)")
  axes[1].grid(True)

  fig.align_ylabels(axes)
  _save(fig, subdir, name)


def plot_learning_curves(s: State, n_trials: int,
                         subdir: str = "question-3",
                         name: str = "q3b_td_learning.png") -> None:
  """Plot V̂(t), ΔV̂(t), δ(t) for every 10th trial."""
  trial_indices = list(range(0, n_trials, 10))
  n_curves = len(trial_indices)

  cmap = mpl.colormaps["viridis_r"]
  colours = [cmap(i / max(n_curves - 1, 1)) for i in range(n_curves)]

  labels = [r"$\hat{V}(t)$", r"$\Delta\hat{V}(t)$", r"$\delta(t)$"]
  keys = ["V_storage", "dV_storage", "error_storage"]

  fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7.5),
                           gridspec_kw={"hspace": 0.10})

  for ax, key, label in zip(axes, keys, labels):
    data = getattr(s, key)
    for ci, tr in enumerate(trial_indices):
      ax.plot(s.t, data[tr], color=colours[ci], lw=1.0, alpha=0.9)
    ax.set_ylabel(label)
    ax.grid(True)

  axes[-1].set_xlabel("Time (s)")

  sm = mpl.cm.ScalarMappable(
    cmap="viridis_r",
    norm=mpl.colors.Normalize(vmin=1, vmax=n_trials),
  )
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.025, pad=0.02)
  cbar.set_label("Trial", fontfamily="monospace")

  fig.align_ylabels(axes)
  _save(fig, subdir, name)


def dopamine_nonlinearity(x: np.ndarray, alpha: float = 6.0, beta: float = 6.0,
                          x_star: float = 0.27) -> np.ndarray:
  """Piecewise-linear dopamine activation DA(x) from Q5(e).

  Compresses negative values by 1/alpha and saturates large positives
  above x* with slope 1/beta.
  """
  y = np.empty_like(x)
  neg = x < 0.0
  mid = (x >= 0.0) & (x < x_star)
  hi = x >= x_star
  y[neg] = x[neg] / alpha
  y[mid] = x[mid]
  y[hi] = x_star + (x[hi] - x_star) / beta
  return y


def plot_last100_by_type(s: State, subdir: str, name: str,
                         n_last: int = 100) -> None:
  """Q5(a): average V̂, ΔV̂, δ over last n_last trials, split by rewarded/unrewarded/all."""
  t = s.t
  r_mask = s.rewarded[-n_last:]

  labels_keys = [
    (r"$\hat{V}(t)$", "V_storage"),
    (r"$\Delta\hat{V}(t)$", "dV_storage"),
    (r"$\delta(t)$", "error_storage"),
  ]

  fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7.5),
                           gridspec_kw={"hspace": 0.10})

  for ax, (label, key) in zip(axes, labels_keys):
    data = getattr(s, key)[-n_last:]
    ax.plot(t, data.mean(axis=0), color="black", lw=1.8, label="all")
    if r_mask.any():
      ax.plot(t, data[r_mask].mean(axis=0), color="tab:green",
              lw=1.4, ls="--", label="rewarded")
    if (~r_mask).any():
      ax.plot(t, data[~r_mask].mean(axis=0), color="tab:red",
              lw=1.4, ls="--", label="unrewarded")
    ax.set_ylabel(label)
    ax.grid(True)

  axes[0].legend(frameon=False, ncol=3)
  axes[-1].set_xlabel("Time (s)")
  fig.align_ylabels(axes)
  _save(fig, subdir, name)


def plot_da_compare(s: State, subdir: str, name: str,
                    n_last: int = 100) -> None:
  """Q5(e): average δ vs DA(δ) over last n_last trials."""
  t = s.t
  delta_tail = s.error_storage[-n_last:]
  da_tail = dopamine_nonlinearity(delta_tail)

  fig, ax = plt.subplots(1, 1, figsize=(7, 3.8))
  ax.plot(t, delta_tail.mean(axis=0), lw=1.8, label=r"$\delta(t)$")
  ax.plot(t, da_tail.mean(axis=0), lw=1.8, label=r"$\mathrm{DA}(\delta(t))$")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Activity")
  ax.grid(True)
  ax.legend(frameon=False)
  _save(fig, subdir, name)


def plot_da_sweep(t: np.ndarray, da_curves: dict[float, np.ndarray],
                  subdir: str, name: str) -> None:
  """Q6(a): overlay average DA time courses for different reward probabilities."""
  fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
  for p in sorted(da_curves):
    ax.plot(t, da_curves[p], lw=1.8, label=f"p = {p:.2f}")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Mean DA (last 100 trials)")
  ax.grid(True)
  ax.legend(frameon=False, ncol=2)
  _save(fig, subdir, name)


def plot_da_peaks(ps: np.ndarray, peak_stim: np.ndarray,
                  at_reward: np.ndarray, subdir: str, name: str) -> None:
  """Q6(b): DA level at stimulus peak and at reward time vs p."""
  fig, ax = plt.subplots(1, 1, figsize=(7, 4))
  ax.plot(ps, peak_stim, marker="o", lw=1.8, label="Peak near stimulus")
  ax.plot(ps, at_reward, marker="s", lw=1.8, label="At reward time")
  ax.set_xlabel("Reward probability  p")
  ax.set_ylabel("Dopamine level")
  ax.grid(True)
  ax.legend(frameon=False)
  _save(fig, subdir, name)


def main():
  _setup_style()

  if FIG_DIR.exists():
    shutil.rmtree(FIG_DIR)
  FIG_DIR.mkdir(parents=True)

  # Q3: tapped delay line, eps=0.2, N=201
  q3_params = Parameters(representation="tapped", lr=0.2, n_trials=201)
  state_q3 = run_td(q3_params)
  plot_stim_reward(state_q3, subdir="question-3", name="q3a_stim_reward.png")
  plot_learning_curves(state_q3, n_trials=q3_params.n_trials,
                       subdir="question-3", name="q3b_td_learning.png")

  # Q4: boxcar, eps=0.01, N=201
  q4_params = Parameters(representation="boxcar", lr=0.01, n_trials=201)
  state_q4 = run_td(q4_params)
  plot_learning_curves(state_q4, n_trials=q4_params.n_trials,
                       subdir="question-4", name="q4a_boxcar_learning.png")

  # Q5: partial reinforcement, boxcar, eps=0.01, p=0.5, N=1000
  q5_params = Parameters(representation="boxcar", lr=0.01, n_trials=1000,
                          reward_probability=0.5, seed=11)
  state_q5 = run_td(q5_params)
  plot_last100_by_type(state_q5, subdir="question-5", name="q5a_last100_by_type.png")
  plot_da_compare(state_q5, subdir="question-5", name="q5e_da_nonlinearity.png")

  # Q6: reward probability sweep
  dt = 0.5
  t_stim = 10.0
  t_reward = 20.0
  p_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
  stim_idx = int(round(t_stim / dt))
  reward_idx = int(round(t_reward / dt))
  window = 2  # +/- 1 s around stimulus for peak detection

  da_curves: dict[float, np.ndarray] = {}
  peak_stim = np.zeros_like(p_values)
  at_reward = np.zeros_like(p_values)
  t_grid = None

  for i, p in enumerate(p_values):
    q6_params = Parameters(
      representation="boxcar", lr=0.01, n_trials=1000,
      reward_probability=float(p), seed=31 + i,
    )
    state_q6 = run_td(q6_params)
    if t_grid is None:
      t_grid = state_q6.t

    # Apply DA nonlinearity per-trial, then average over last 100
    da = dopamine_nonlinearity(state_q6.error_storage[-100:]).mean(axis=0)
    da_curves[float(p)] = da

    lo = max(0, stim_idx - window)
    hi = min(da.shape[0], stim_idx + window + 1)
    peak_stim[i] = np.max(da[lo:hi])
    at_reward[i] = da[reward_idx]

  plot_da_sweep(t_grid, da_curves, subdir="question-6", name="q6a_da_sweep.png")
  plot_da_peaks(p_values, peak_stim, at_reward,
                subdir="question-6", name="q6b_da_peaks.png")

  print(f"Figures saved to {FIG_DIR.resolve()}")

  


if __name__ == "__main__":
  main()
