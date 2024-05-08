import warnings

import cvxpy as cp
import numpy as np


def iso_regression_linf(x):
  """Unweighted isotonic regression with L-infinity loss"""
  l_max = np.maximum.accumulate(x)
  r_min = np.minimum.accumulate(x[::-1])[::-1]
  return (l_max + r_min) / 2


class PrivateHDEFairPostProcessor:

  def fit(self,
          scores,
          groups,
          alpha=0.0,
          bound=None,
          n_bins=10,
          eps=np.inf,
          rng=None):

    if rng is None:
      rng = np.random.default_rng()
    self.rng_ = rng

    self.alpha_ = alpha
    self.n_groups_ = int(1 + np.max(groups))
    if bound is None:
      warnings.warn(
          "Bound is not set, using min and max of scores, which violates differential privacy"
      )
      bound = (np.min(scores), np.max(scores))
    self.bound_ = bound

    n = len(scores)
    # w = n / len(groups)
    self.n_bins_ = n_bins
    self.bin_width_ = (bound[1] - bound[0]) / n_bins

    # Convert scores to bins (index)
    self.score_to_bin_ = lambda s: np.clip(
        np.floor((s - bound[0]) / self.bin_width_), 0, n_bins - 1).astype(int)
    bins = self.score_to_bin_(scores)

    # Get histogram
    hist = np.empty((self.n_groups_, n_bins), dtype=float)
    for a in range(self.n_groups_):
      mask = groups == a
      hist[a] = np.bincount(bins[mask], minlength=n_bins)
    hist *= 1 / n

    # Add noise to histogram
    noise = rng.laplace(scale=2 / (eps * n), size=(self.n_groups_, n_bins))
    hist += noise

    # Get group weight
    self.w_ = np.clip(hist.sum(axis=1), 1e-6, None)

    # Renormalize histogram
    hist_by_group = hist / self.w_[:, None]
    cumsum_by_group = np.cumsum(hist_by_group,
                                axis=1)  # get partial sums ("cdf")
    for a in range(self.n_groups_):
      cumsum_by_group[a] = iso_regression_linf(
          cumsum_by_group[a])  # perform isotonic regression to get valid cdf
    cumsum_by_group = np.clip(cumsum_by_group, 0, 1)  # clip cdf to [0, 1]
    cumsum_by_group[:, -1] = 1  # set last value of "cdf" to 1

    self.hist_by_group_ = np.diff(cumsum_by_group, prepend=0, axis=1)

    # Get and solve fair post-processing LP
    problem = self.linprog_(self.hist_by_group_, alpha=alpha, w=self.w_)
    # problem.solve(solver=cp.CBC, integerTolerance=1e-8)
    problem.solve(solver=cp.GUROBI)  # ...if you have a Gurobi license

    # Store value and target distributions
    self.score_ = problem.value / self.bin_width_**2
    self.q_by_group_ = problem.var_dict["q"].value

    # Store couplings and optimal transports
    self.gamma_by_group_ = np.clip(
        [problem.var_dict[f'gamma_{a}'].value for a in range(self.n_groups_)],
        0, 1)
    with np.errstate(invalid='ignore'):
      self.g_ = self.gamma_by_group_ / self.gamma_by_group_.sum(axis=-1,
                                                                keepdims=True)
    # Do nothing for unseen values
    for a in range(self.n_groups_):
      for i in range(n_bins):
        if np.isnan(self.g_[a][i][0]):
          self.g_[a][i] = 0
          self.g_[a][i][i] = 1

    return self

  def linprog_(self, hist_by_group, alpha, w):

    alpha = cp.Parameter(value=alpha, name="alpha")
    n_bins = self.n_bins_ or hist_by_group.shape[1]
    n_groups = self.n_groups_ or hist_by_group.shape[0]

    # Variables are the probability mass of the couplings, the barycenter,
    # the output distributions, and slacks
    gamma_by_group = [
        cp.Variable((n_bins, n_bins), name=f"gamma_{a}", nonneg=True)
        for a in range(n_groups)
    ]
    barycenter = cp.Variable(n_bins, name="barycenter", nonneg=True)
    q = cp.Variable((n_groups, n_bins), name="q", nonneg=True)
    slack = cp.Variable((n_groups, n_bins), name="slack", nonneg=True)

    # Get l2 transportation costs
    costs = (np.arange(n_bins, dtype=float)[:, None] - np.arange(n_bins))**2
    cost = cp.sum([
        cp.sum(cp.multiply(gamma_by_group[a], costs)) * w[a]
        for a in range(n_groups)
    ])

    # Build constraints
    constraints = []

    # \sum_{s'} \gamma_{a, s, s'} = p_{a, s} for all a
    for a in range(self.n_groups_):
      constraints.append(cp.sum(gamma_by_group[a], axis=1) == hist_by_group[a])

    # \sum_s \gamma_{a, s, s'} = q_{a, s'}
    for a in range(self.n_groups_):
      constraints.append(cp.sum(gamma_by_group[a], axis=0) == q[a])

    # KS distance
    # | \sum_{s' <= s} (q_{a, s'} - barycenter_{s'}) | <= \xi_{a, s}
    for a in range(self.n_groups_):
      constraints.append(cp.abs(cp.cumsum(q[a] - barycenter)) <= slack[a])
    # \xi_{a, y} <= \alpha / 2
    constraints.append(slack <= alpha / 2)

    # # TV distance
    # # | q_{a, s} - barycenter_{s} | <= \xi_{a, s}
    # for a in range(self.n_groups_):
    #   constraints.append(cp.abs(q[a] - barycenter) <= slack[a])
    # # \sum_{s} \xi_{a, s} <= \alpha / 2
    # constraints.append(cp.sum(slack, axis=1) <= alpha / 2)

    return cp.Problem(cp.Minimize(cost), constraints)

  def predict(self, scores, groups):
    # Convert scores to bins (index)
    bins = self.score_to_bin_(scores)

    # Randomly reassign bins according to the optimal transports
    new_bins = np.empty_like(bins)
    for a in np.unique(groups):
      for i in np.unique(bins[groups == a]):
        mask = (bins == i) & (groups == a)
        new_bins[mask] = self.rng_.choice(self.n_bins_,
                                          size=np.sum(mask),
                                          p=self.g_[a][i])

    return new_bins * self.bin_width_ + self.bound_[0] + self.bin_width_ / 2


class WassersteinBarycenterFairPostProcessor:
  """
  Python reimplementation of https://github.com/lucaoneto/NIPS2020_Fairness
  """

  def fit(self, scores, groups, eps=None, rng=None):

    if rng is None:
      rng = np.random.default_rng()
    self.rng_ = rng

    self.n_groups_ = int(1 + np.max(groups))
    self.w_ = np.bincount(groups, minlength=self.n_groups_) / len(groups)

    if eps is None:
      eps = np.finfo(scores.dtype).eps
    self.eps_ = eps
    jitter = self.rng_.normal(scale=self.eps_, size=len(scores))
    scores = scores + jitter

    self.s0_by_group_ = []
    self.s1_by_group_ = []

    for a in range(self.n_groups_):
      mask = groups == a
      s = scores[mask]

      # Shuffle and split the scores in half
      s = self.rng_.permutation(s)
      s0, s1 = np.array_split(s, 2)

      # Sort and store the scores
      s0 = np.sort(s0)
      s1 = np.sort(s1)
      self.s0_by_group_.append(s0)
      self.s1_by_group_.append(s1)

    return self

  def predict(self, scores, groups):

    jitter = self.rng_.normal(scale=self.eps_, size=len(scores))
    scores = scores + jitter

    s1_sizes = np.array([len(s1) for s1 in self.s1_by_group_])
    new_scores = np.empty_like(scores)

    for a in np.unique(groups):
      mask = groups == a
      s = scores[mask]

      # Get percentile of scores in s0
      k = np.searchsorted(self.s0_by_group_[a], s) / len(self.s0_by_group_[a])

      # Get scores at the same percentile in s1 for all groups
      idx = np.clip(np.ceil(k * s1_sizes[:, None]).astype(int) - 1, 0, None)
      y = [self.s1_by_group_[b][idx[b]] for b in range(self.n_groups_)]

      # Take weighted average
      new_scores[mask] = np.tensordot(self.w_, y, axes=1)

    return new_scores
