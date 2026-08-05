"""
Demonstration that the DataAPI works end to end.

    python -m statarb.api.demo

Every call here is one a backtester would make. If this runs, the data layer is
usable; if it doesn't, nothing downstream can be trusted.
"""

from __future__ import annotations

import polars as pl

from ..data.api import DataAPI


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    api = DataAPI()

    rule("1. What is in the database")
    print(api.summary())

    rule("2. Ticker -> sid")
    sid = api.sid_for_ticker("AAPL")
    print(f"AAPL-US -> sid {sid}")

    rule("3. THE TEST: adjusted prices across AAPL's 2020 4-for-1 split")
    raw = api.get_prices([sid], "2020-08-26", "2020-09-02", adjusted="raw",
                         columns=["d", "px_close"]).rename({"px_close": "raw"})
    tot = api.get_prices([sid], "2020-08-26", "2020-09-02", adjusted="total",
                         columns=["d", "px_close"]).rename({"px_close": "total_adj"})
    both = raw.join(tot, on="d").with_columns(
        (pl.col("raw") / pl.col("raw").shift(1) - 1).alias("ret_raw"),
        (pl.col("total_adj") / pl.col("total_adj").shift(1) - 1).alias("ret_adj"),
    )
    print(both)
    split_row = both.filter(pl.col("d") == pl.date(2020, 8, 31))
    print(f"\n  raw return on split day      : {split_row['ret_raw'][0]:+.2%}")
    print(f"  adjusted return on split day : {split_row['ret_adj'][0]:+.2%}")

    rule("4. Returns, computed per security (never across stock boundaries)")
    rets = api.get_returns([sid], "2024-01-02", "2024-01-10")
    print(rets)

    rule("5. POINT-IN-TIME universe: as it stood on 2005-06-01")
    uni = api.get_universe_as_of("2005-06-01")
    dead_now = uni.filter(pl.col("is_dead"))
    print(f"investable on 2005-06-01 : {uni.height:,} securities")
    print(f"  of which now dead      : {dead_now.height:,} "
          f"({dead_now.height / max(uni.height, 1):.1%})")
    print("\n  examples of securities that were investable then and are gone now:")
    print(dead_now.select("sid", "ticker_region", "security_name",
                          "first_px_date", "last_px_date").head(6))
    print("\n  ^ their presence is the proof there is no survivorship bias:")
    print("    a naive 'currently listed' universe would omit every one of them.")

    rule("6. Universe with a liquidity screen (trailing window, no look-ahead)")
    liq = api.get_universe_as_of("2005-06-01", min_price=5.0,
                                 min_adv_usd=1_000_000)
    print(f"after min_price>=$5 and ADV>=$1m : {liq.height:,} securities")
    print(liq.select("ticker_region", "last_price", "adv_usd")
             .sort("adv_usd", descending=True).head(5))

    rule("7. Rebalance dates from the vendor calendar")
    rb = api.get_rebalance_dates("2024-01-01", "2024-06-30")
    print(f"month-end trading days in H1 2024: {rb}")

    rule("8. Wide panel - the shape PCA and clustering want")
    sids = liq.sort("adv_usd", descending=True)["sid"].to_list()[:5]
    panel = api.get_panel(sids, "2005-06-01", "2005-06-10")
    print(panel)

    rule("9. Corporate actions, for explaining an extreme move")
    ca = api.get_corporate_actions([sid], kinds=["SPLIT"])
    print(ca)

    rule("10. Reproducibility: pin a version")
    print(f"current table versions: {api.table_versions()}")
    print("A result computed today can be reproduced exactly with")
    print("    DataAPI(version=<the version recorded alongside that result>)")

    print(f"\n{'=' * 72}\nDataAPI is usable.\n{'=' * 72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
