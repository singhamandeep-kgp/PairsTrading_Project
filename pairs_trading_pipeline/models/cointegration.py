import os
import glob
from functools import lru_cache
from typing import Optional
import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

@lru_cache(maxsize=None)
def _load_log_price_series(permno: int, prices_dir: str = "GICS_Filtered_Equities_Prices") -> pd.Series:
    pattern = os.path.join(prices_dir, "GICS_*.pkl")
    for path in sorted(glob.glob(pattern)):
        df = pd.read_pickle(path)
        if permno in df.columns:
            series = df[permno].copy().sort_index().astype(float)
            return np.log(series)
    raise KeyError(f"PERMNO {permno} not found in {prices_dir}")

def _align_prices(p1: pd.Series, p2: pd.Series, min_obs: int = 50) -> Optional[pd.DataFrame]:
    joined = pd.concat([p1, p2], axis=1, join="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan)
    joined.columns = ["asset1", "asset2"]
    return joined

def cointegration_from_clusters(
	                            pairs_csv_path: str,
                                prices_dir: str = "GICS_Filtered_Equities_Prices",
	                            significance: float = 0.05,
	                            min_obs: int = 50) -> pd.DataFrame:
	"""
	Compute cointegration stats for all pairs within each OPTICS cluster.
	"""

	pairs_df = pd.read_csv(pairs_csv_path)
	results = []

	# Filter out noise points (-1 labels)
	pairs_df = pairs_df[pairs_df["optics_label"] != -1]

	for label, group in pairs_df.groupby("optics_label"):
		permnos = group["permno"].astype(int).unique()
		# Iterate over all combinations of permnos in the cluster
		for a, b in itertools.combinations(permnos, 2):
			try:
				s1 = _load_log_price_series(a, prices_dir)
				s2 = _load_log_price_series(b, prices_dir)
			except KeyError:
				continue

			aligned = _align_prices(s1, s2, min_obs=min_obs)
			if aligned is None:
				continue

			# Compute cointegration both ways since Engle-Granger is not symmetric and take the lower p-value
			try:
				t_stat_xy, p_value_xy, crit_vals_xy = coint(aligned["asset1"], aligned["asset2"])
				t_stat_yx, p_value_yx, crit_vals_yx = coint(aligned["asset2"], aligned["asset1"])
			except Exception as e:
				print(f"Error during cointegration computation: {e}")
				continue

			if p_value_xy < p_value_yx:
				t_stat, crit_vals = t_stat_xy, crit_vals_xy
			else:
				a, b = b, a  
				t_stat, crit_vals = t_stat_yx, crit_vals_yx

			results.append(
				{
					"permno1": a,
					"permno2": b,
					"cluster": label,
					"t_stat": t_stat,
					"p_value": min(p_value_xy, p_value_yx),
					"crit_1pct": crit_vals[0],
					"crit_5pct": crit_vals[1],
					"crit_10pct": crit_vals[2],
					"cointegrated": min(p_value_xy, p_value_yx) < significance,
					"n_obs": len(aligned),
				}
			)
			
	results = pd.DataFrame(results)
	results.sort_values(by=['p_value'], ascending=True, inplace=True)
	
	return pd.DataFrame(results)

