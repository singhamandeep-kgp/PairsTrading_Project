from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import coint

from . import adapters
from .signals import generate_positions
from .pnl import pair_pnl, portfolio_pnl, equity_curve
from .artifacts import save_artifact


@dataclass
class WindowConfig:
    pca_window: int = 252
    coint_window: int = 504
    z_window: int = 60


@dataclass
class StrategyParams:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: Optional[float] = 4.0
    max_holding_days: Optional[int] = 60


@dataclass
class SelectionParams:
    max_pairs: int = 10
    pval_thresh: float = 0.05
    half_life_min: float = 5.0
    half_life_max: float = 60.0
    min_overlap: int = 200
    k_nn: int = 0  # optional; 0 disables kNN for noise


class RollingPairsBacktestEngine:
    def __init__(
        self,
        sector_id: int | str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        rebalance_freq: str = "MS",
        windows: WindowConfig | None = None,
        strategy: StrategyParams | None = None,
        selection: SelectionParams | None = None,
        artifact_dir: str = "rolling_artifacts",
        prices_dir: str = "GICS_Filtered_Equities_Prices",
        k_loadings: int = 6,
    ) -> None:
        self.sector_id = sector_id
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_freq = rebalance_freq
        self.windows = windows or WindowConfig()
        self.strategy = strategy or StrategyParams()
        self.selection = selection or SelectionParams()
        self.artifact_dir = artifact_dir
        self.prices = adapters.load_sector_prices(sector_id, prices_dir=prices_dir).sort_index()
        self.k_loadings = k_loadings

    def run(self) -> Dict[str, pd.DataFrame | pd.Series]:
        rebal_dates = self._rebalance_dates()
        portfolio_pnl_all: List[pd.Series] = []
        pair_pnl_all: List[pd.DataFrame] = []
        selected_pairs_all: List[pd.DataFrame] = []

        for i, rebal_date in enumerate(rebal_dates):
            next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else self.end_date
            result = self.run_one_rebalance(rebal_date, next_date)
            if result["portfolio_pnl"].empty:
                continue
            portfolio_pnl_all.append(result["portfolio_pnl"])
            pair_pnl_all.append(result["pair_pnl"])
            selected_pairs_df = result["pairs_selected"].copy()
            selected_pairs_df["rebalance_date"] = rebal_date
            selected_pairs_all.append(selected_pairs_df)

        if portfolio_pnl_all:
            portfolio_pnl_series = pd.concat(portfolio_pnl_all).sort_index()
        else:
            portfolio_pnl_series = pd.Series(dtype=float)

        pair_pnl_concat = pd.concat(pair_pnl_all, axis=0).sort_index() if pair_pnl_all else pd.DataFrame()
        selected_pairs_concat = pd.concat(selected_pairs_all, axis=0) if selected_pairs_all else pd.DataFrame()
        equity = equity_curve(portfolio_pnl_series)

        return {
            "pnl": portfolio_pnl_series,
            "equity_curve": equity,
            "selected_pairs_all": selected_pairs_concat,
            "pair_pnl_all": pair_pnl_concat,
        }

    def _rebalance_dates(self) -> List[pd.Timestamp]:
        dates = pd.date_range(self.start_date, self.end_date, freq=self.rebalance_freq)
        # ensure we start at first available trading date on/after start
        trading_dates = self.prices.index
        valid = [trading_dates[trading_dates.get_loc(d, method="bfill")] for d in dates if d <= trading_dates[-1]]
        return [pd.to_datetime(d) for d in valid if d >= self.start_date]

    def run_one_rebalance(self, rebal_date: pd.Timestamp, next_date: pd.Timestamp) -> Dict[str, pd.DataFrame | pd.Series]:
        px_pca = self.prices.loc[:rebal_date].tail(self.windows.pca_window + 1)
        px_coint = self.prices.loc[:rebal_date].tail(self.windows.coint_window)

        if len(px_pca) < self.windows.pca_window or len(px_coint) < self.windows.coint_window:
            return {"portfolio_pnl": pd.Series(dtype=float), "pair_pnl": pd.DataFrame(), "pairs_selected": pd.DataFrame(), "pairs_screened": pd.DataFrame()}

        returns = adapters.scale_returns(px_pca)
        factor_returns, factor_loadings, evr = adapters.pca_on_returns(returns.tail(self.windows.pca_window))
        loadings_use = factor_loadings.iloc[:, : min(self.k_loadings, factor_loadings.shape[1])]
        cluster_labels = adapters.optics_cluster(loadings_use, min_samples=2, max_eps=np.inf, metric="euclidean", scale=True)

        candidate_pairs = self._generate_candidate_pairs(cluster_labels, loadings_use)
        pairs_screened = self._screen_pairs(candidate_pairs, px_coint, rebal_date)
        pairs_selected = self._select_pairs(pairs_screened)

        # Forward trading
        forward_prices = self.prices.loc[rebal_date:next_date]
        pair_pnl_df, portfolio_pnl_series = self._forward_trade(pairs_selected, forward_prices)

        # Artifacts
        save_artifact(loadings_use, self.artifact_dir, self.sector_id, rebal_date, "pca_loadings")
        save_artifact(cluster_labels.to_frame("optics_label"), self.artifact_dir, self.sector_id, rebal_date, "optics_labels")
        save_artifact(candidate_pairs, self.artifact_dir, self.sector_id, rebal_date, "candidate_pairs")
        save_artifact(pairs_screened, self.artifact_dir, self.sector_id, rebal_date, "pairs_screened")
        save_artifact(pairs_selected, self.artifact_dir, self.sector_id, rebal_date, "pairs_selected")
        save_artifact(portfolio_pnl_series.to_frame("pnl"), self.artifact_dir, self.sector_id, rebal_date, "portfolio_pnl_forward")
        if not pair_pnl_df.empty:
            save_artifact(pair_pnl_df, self.artifact_dir, self.sector_id, rebal_date, "pair_pnl_forward")

        return {
            "portfolio_pnl": portfolio_pnl_series,
            "pair_pnl": pair_pnl_df,
            "pairs_selected": pairs_selected,
            "pairs_screened": pairs_screened,
        }

    def _generate_candidate_pairs(self, labels: pd.Series, loadings: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict] = []
        for cluster_id, members in labels.groupby(labels):
            permnos = members.index.tolist()
            if cluster_id == -1:
                continue
            for i in range(len(permnos)):
                for j in range(i + 1, len(permnos)):
                    a, b = permnos[i], permnos[j]
                    vec_a = loadings.loc[a].values
                    vec_b = loadings.loc[b].values
                    dist = float(np.linalg.norm(vec_a - vec_b))
                    records.append({"a": a, "b": b, "cluster_id": cluster_id, "pca_distance": dist})
        return pd.DataFrame(records)

    def _screen_pairs(self, pairs: pd.DataFrame, px_coint: pd.DataFrame, rebal_date: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for _, r in pairs.iterrows():
            a, b = r["a"], r["b"]
            s1 = np.log(px_coint[a].dropna()) if a in px_coint else None
            s2 = np.log(px_coint[b].dropna()) if b in px_coint else None
            if s1 is None or s2 is None:
                continue
            aligned = adapters.align_prices(s1, s2, min_obs=self.selection.min_overlap)
            if aligned is None or len(aligned) < self.selection.min_overlap:
                continue

            try:
                t_xy, p_xy, _ = coint(aligned["asset1"], aligned["asset2"])
                t_yx, p_yx, _ = coint(aligned["asset2"], aligned["asset1"])
            except Exception:
                continue

            if p_xy <= p_yx:
                best_dir = "a_on_b"
                y, x = aligned["asset1"], aligned["asset2"]
            else:
                best_dir = "b_on_a"
                y, x = aligned["asset2"], aligned["asset1"]

            reg = smf.ols("y ~ x", data={"y": y, "x": x}).fit()
            alpha = float(reg.params["Intercept"])
            beta = float(reg.params["x"])

            spread = y - (alpha + beta * x)
            half_life = self._half_life(spread)
            sigma_change = float(spread.diff().dropna().std())

            rows.append(
                {
                    "date_rebal": rebal_date,
                    "a": a,
                    "b": b,
                    "pval_min": float(min(p_xy, p_yx)),
                    "best_direction": best_dir,
                    "beta": beta,
                    "alpha": alpha,
                    "half_life": half_life,
                    "spread_change_sigma": sigma_change,
                    "pca_distance": r.get("pca_distance", np.nan),
                    "cluster_id": r.get("cluster_id", np.nan),
                    "overlap_n": len(aligned),
                }
            )
        return pd.DataFrame(rows)

    def _select_pairs(self, screened: pd.DataFrame) -> pd.DataFrame:
        if screened.empty:
            return screened
        filt = screened[
            (screened["pval_min"] <= self.selection.pval_thresh)
            & (screened["half_life"] >= self.selection.half_life_min)
            & (screened["half_life"] <= self.selection.half_life_max)
            & (screened["overlap_n"] >= self.selection.min_overlap)
        ].copy()
        if filt.empty:
            return filt
        filt["score"] = -np.log10(filt["pval_min"]) - 0.02 * filt["half_life"] - 0.5 * filt["spread_change_sigma"] + 0.1 * (1 / (1 + filt["pca_distance"].replace(0, np.nan)))
        filt.sort_values("score", ascending=False, inplace=True)
        return filt.head(self.selection.max_pairs)

    def _forward_trade(self, pairs_selected: pd.DataFrame, forward_prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pair_pnls = {}
        for _, r in pairs_selected.iterrows():
            a, b, beta = r["a"], r["b"], r["beta"]
            if a not in forward_prices or b not in forward_prices:
                continue
            log_a = np.log(forward_prices[a].dropna())
            log_b = np.log(forward_prices[b].dropna())
            aligned = adapters.align_prices(log_a, log_b, min_obs=self.windows.z_window + 1)
            if aligned is None or len(aligned) <= self.windows.z_window:
                continue
            spread = aligned["asset1"] - beta * aligned["asset2"]
            positions = generate_positions(
                spread,
                z_window=self.windows.z_window,
                entry_z=self.strategy.entry_z,
                exit_z=self.strategy.exit_z,
                stop_z=self.strategy.stop_z,
                max_holding_days=self.strategy.max_holding_days,
            )
            pair_pnls[f"{a}_{b}"] = pair_pnl(spread, positions)
        pair_pnl_df = pd.DataFrame(pair_pnls)
        portfolio_series = portfolio_pnl(pair_pnl_df)
        return pair_pnl_df, portfolio_series

    @staticmethod
    def _half_life(spread: pd.Series) -> float:
        s = spread.dropna()
        if len(s) < 3:
            return np.inf
        df = pd.DataFrame({"s_lag": s.shift(1), "ds": s.diff()}).dropna()
        if len(df) < 3:
            return np.inf
        res = smf.ols("ds ~ s_lag", data=df).fit()
        phi = float(res.params.get("s_lag", np.nan))
        if np.isnan(phi) or phi >= 0:
            return np.inf
        return float(-np.log(2) / phi)
