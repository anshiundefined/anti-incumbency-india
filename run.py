"""
Do Indian voters punish incumbents? A regression discontinuity on close state-assembly elections.

Compare parties (and candidates) that barely won a seat with those that barely lost it.
Near a margin of zero, winning is as good as random, so the jump in the probability of
winning the *next* election at zero is the causal effect of incumbency.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from utils import DATA, HIGHLIGHT, MUTED, PALETTE, get, md_table, savefig, setup, write_results

URL = "https://raw.githubusercontent.com/datameet/india-election-data/master/assembly-elections/assembly.csv"
SRC = "Election Commission of India results, compiled by DataMeet (github.com/datameet/india-election-data)"
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
H = 0.05            # main bandwidth: +/- 5 percentage points of vote-share margin
BANDWIDTHS = [0.02, 0.03, 0.05, 0.075, 0.10]


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^A-Z ]", " ", regex=True).str.split().str.join(" ")


def election_key(raw: str) -> float:
    raw = str(raw).lower()
    y = int(re.search(r"\d{4}", raw).group())
    m = next((v for k, v in MONTHS.items() if k in raw), 6)
    return y + m / 100


def load() -> pd.DataFrame:
    txt = get(URL).content.decode("latin-1")
    d = pd.read_csv(io.StringIO(txt), low_memory=False)
    d = d.dropna(subset=["VOTES", "AC_NAME", "YEAR"])
    d["ekey"] = d["YEAR"].map(election_key)
    d["year"] = d["ekey"].astype(int)
    d["state"] = d["ST_NAME"].str.strip()
    d["ac"] = norm(d["AC_NAME"])
    d["cand"] = norm(d["NAME"])
    d["party"] = d["PARTY"].astype(str).str.strip().str.upper()
    return d


def races(d: pd.DataFrame) -> pd.DataFrame:
    """One row per party (or candidate) per race, with signed margin vs the winner/runner-up."""
    d = d.copy()
    tot = d.groupby(["state", "ekey", "ac"])["VOTES"].transform("sum")
    d["share"] = d["VOTES"] / tot
    d["rank"] = d.groupby(["state", "ekey", "ac"])["VOTES"].rank(ascending=False, method="first")
    top = d[d["rank"] <= 2].copy()
    s1 = top[top["rank"] == 1].set_index(["state", "ekey", "ac"])["share"]
    s2 = top[top["rank"] == 2].set_index(["state", "ekey", "ac"])["share"]
    gap = (s1 - s2).rename("gap")
    top = top.join(gap, on=["state", "ekey", "ac"]).dropna(subset=["gap"])
    top["margin"] = np.where(top["rank"] == 1, top["gap"], -top["gap"])
    top["won"] = (top["rank"] == 1).astype(int)
    return top


def attach_next(top: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Next general election in the same state and constituency (matched on constituency name)."""
    order = d[["state", "ekey"]].drop_duplicates().sort_values(["state", "ekey"])
    order["next_ekey"] = order.groupby("state")["ekey"].shift(-1)
    top = top.merge(order, on=["state", "ekey"]).dropna(subset=["next_ekey"])
    nxt = d[["state", "ekey", "ac", "party", "cand", "VOTES"]].copy()
    nxt["rank"] = nxt.groupby(["state", "ekey", "ac"])["VOTES"].rank(ascending=False, method="first")
    held = nxt[["state", "ekey", "ac"]].drop_duplicates().assign(seat_exists=1)
    winners = nxt[nxt["rank"] == 1].rename(columns={"party": "next_win_party", "cand": "next_win_cand"})
    ran = nxt[["state", "ekey", "ac", "cand"]].drop_duplicates().assign(ran_again=1)
    key = {"ekey": "next_ekey"}
    top = top.merge(held.rename(columns=key), on=["state", "next_ekey", "ac"], how="inner")   # seat still exists
    top = top.merge(winners[["state", "ekey", "ac", "next_win_party", "next_win_cand"]].rename(columns=key),
                    on=["state", "next_ekey", "ac"], how="left")
    top = top.merge(ran.rename(columns=key), on=["state", "next_ekey", "ac", "cand"], how="left")
    top["ran_again"] = top["ran_again"].fillna(0).astype(int)
    top["party_wins_next"] = (top["party"] == top["next_win_party"]).astype(int)
    top["cand_wins_next"] = (top["cand"] == top["next_win_cand"]).astype(int)
    return top


def rd(df: pd.DataFrame, y: str, h: float = H) -> dict:
    """Local linear RD with triangular kernel, separate slopes each side, SEs clustered by constituency."""
    s = df[df["margin"].abs() <= h].copy()
    if len(s) < 60:
        return {"est": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan, "n": len(s), "p": np.nan}
    s["T"] = (s["margin"] > 0).astype(float)
    X = pd.DataFrame({"const": 1.0, "T": s["T"], "m": s["margin"], "Tm": s["T"] * s["margin"]})
    w = 1 - s["margin"].abs() / h
    fit = sm.WLS(s[y], X, weights=w).fit(cov_type="cluster",
                                         cov_kwds={"groups": pd.factorize(s["state"] + s["ac"])[0]})
    b, se = fit.params["T"], fit.bse["T"]
    return {"est": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "n": len(s), "p": fit.pvalues["T"]}


def binned(df: pd.DataFrame, y: str, width: float = 0.01, lim: float = 0.2) -> pd.DataFrame:
    s = df[df["margin"].abs() <= lim].copy()
    s["bin"] = (np.floor(s["margin"] / width) + 0.5) * width
    return s.groupby("bin")[y].agg(["mean", "count"]).reset_index()


def main() -> None:
    setup()
    d = load()
    top = races(d)
    df = attach_next(top, d)
    party = df[df["party"] != "IND"].copy()                  # party incumbency: independents are not a party
    cand = df.copy()
    df[["state", "year", "ekey", "next_ekey", "ac", "AC_TYPE", "party", "cand", "share", "margin", "won",
        "party_wins_next", "cand_wins_next", "ran_again"]].to_csv(DATA / "rd_sample.csv", index=False)
    md = []

    # ---- 1. Main RD plots ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (data, y, title) in zip(axes, [(party, "party_wins_next", "Party wins the seat next time"),
                                           (cand, "cand_wins_next", "Same candidate wins next time")]):
        b = binned(data, y)
        ax.scatter(100 * b["bin"], b["mean"], s=18 + b["count"] / b["count"].max() * 40, color=PALETTE[0], alpha=0.8)
        for side in (-1, 1):
            s = data[(np.sign(data["margin"]) == side) & (data["margin"].abs() <= 0.2)]
            fit = np.polyfit(s["margin"], s[y], 1)
            xs = np.linspace(0, 0.2 * side, 50)
            ax.plot(100 * xs, np.polyval(fit, xs), color=HIGHLIGHT, lw=2.2)
        ax.axvline(0, color="#333", lw=1)
        r = rd(data, y)
        ax.set_title(f"{title}\nRD estimate at 0: {100 * r['est']:+.1f} pp (SE {100 * r['se']:.1f})")
        ax.set_xlabel("Vote-share margin in this election (pp; >0 = won)")
        ax.set_ylabel("Probability")
    fig.suptitle("Barely winning vs barely losing an Indian state assembly seat", fontweight="bold", y=1.03)
    f1 = savefig(fig, "01_rd_plots", SRC)

    # ---- 2. Robustness to bandwidth -------------------------------------------------------------
    rob = []
    for h in BANDWIDTHS:
        for name, data, y in [("Party", party, "party_wins_next"), ("Candidate", cand, "cand_wins_next")]:
            r = rd(data, y, h)
            rob.append({"Outcome": name, "Bandwidth (pp)": 100 * h, "Effect (pp)": 100 * r["est"],
                        "SE (pp)": 100 * r["se"], "p": r["p"], "N": r["n"]})
    rob = pd.DataFrame(rob)

    # placebo: should NOT jump - did the party win this seat in the PREVIOUS election?
    prev = top.merge(d[["state", "ekey"]].drop_duplicates().sort_values(["state", "ekey"])
                     .assign(prev_ekey=lambda x: x.groupby("state")["ekey"].shift(1)), on=["state", "ekey"])
    pw = d[["state", "ekey", "ac", "party", "VOTES"]].copy()
    pw["rank"] = pw.groupby(["state", "ekey", "ac"])["VOTES"].rank(ascending=False, method="first")
    pw = pw[pw["rank"] == 1].rename(columns={"ekey": "prev_ekey", "party": "prev_win_party"})
    prev = prev.merge(pw[["state", "prev_ekey", "ac", "prev_win_party"]], on=["state", "prev_ekey", "ac"], how="inner")
    prev = prev[prev["party"] != "IND"]
    prev["won_before"] = (prev["party"] == prev["prev_win_party"]).astype(int)
    placebo = rd(prev, "won_before")

    # ---- 3. Over time --------------------------------------------------------------------------
    party["decade"] = (party["year"] // 10) * 10
    cand["decade"] = (cand["year"] // 10) * 10
    dec = []
    for dcd in sorted(party["decade"].unique()):
        rp = rd(party[party["decade"] == dcd], "party_wins_next", 0.075)
        rc = rd(cand[cand["decade"] == dcd], "cand_wins_next", 0.075)
        dec.append({"Decade": f"{dcd}s", "Party effect": rp["est"], "Party lo": rp["lo"], "Party hi": rp["hi"],
                    "Candidate effect": rc["est"], "Cand lo": rc["lo"], "Cand hi": rc["hi"], "N (party)": rp["n"]})
    dec = pd.DataFrame(dec).dropna(subset=["Party effect"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dec))
    for off, col, lo, hi, c, lab in [(-0.12, "Party effect", "Party lo", "Party hi", PALETTE[0], "Party"),
                                     (0.12, "Candidate effect", "Cand lo", "Cand hi", PALETTE[1], "Candidate")]:
        ax.errorbar(x + off, 100 * dec[col], yerr=[100 * (dec[col] - dec[lo]), 100 * (dec[hi] - dec[col])],
                    fmt="o", color=c, capsize=4, label=lab, ms=7)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x, dec["Decade"])
    ax.set_ylabel("Incumbency effect on winning next time (pp)")
    ax.set_title("Incumbency advantage by decade (RD, ±7.5 pp bandwidth, 95% CI)")
    ax.legend()
    f2 = savefig(fig, "02_by_decade", SRC)

    # ---- 4. By state (largest samples) ------------------------------------------------------------
    st = []
    for s_, g in party.groupby("state"):
        r = rd(g, "party_wins_next", 0.10)
        if np.isfinite(r["est"]) and r["n"] >= 250:
            st.append({"State": s_, "Effect": r["est"], "lo": r["lo"], "hi": r["hi"], "N": r["n"]})
    st = pd.DataFrame(st).sort_values("Effect")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(st))))
    y = np.arange(len(st))
    ax.errorbar(100 * st["Effect"], y, xerr=[100 * (st["Effect"] - st["lo"]), 100 * (st["hi"] - st["Effect"])],
                fmt="o", color=PALETTE[0], capsize=3)
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y, st["State"])
    ax.set_xlabel("Party incumbency effect (pp; RD, ±10 pp bandwidth, 95% CI)")
    ax.set_title("Where are incumbent parties punished?")
    f3 = savefig(fig, "03_by_state", SRC)

    # ---- Write-up ----------------------------------------------------------------------------------
    rp, rc = rd(party, "party_wins_next"), rd(cand, "cand_wins_next")
    rr = rd(cand, "ran_again")
    rc_cond = rd(cand[cand["ran_again"] == 1], "cand_wins_next")
    md.append("### Headline numbers (±5 pp bandwidth)\n")
    md.append(f"- **Party incumbency effect:** barely winning a seat changes the party's chance of winning it next time by "
              f"**{100 * rp['est']:+.1f} pp** (95% CI {100 * rp['lo']:+.1f} to {100 * rp['hi']:+.1f}; N = {rp['n']:,}).")
    md.append(f"- **Candidate incumbency effect:** **{100 * rc['est']:+.1f} pp** (95% CI {100 * rc['lo']:+.1f} to {100 * rc['hi']:+.1f}).")
    md.append(f"- Barely-winning candidates are **{100 * rr['est']:+.1f} pp** more likely to contest again. Among those who do, "
              f"the effect on winning is **{100 * rc_cond['est']:+.1f} pp** (not causal: who re-runs is selected).")
    md.append(f"- Placebo (winning *this* time should not change whether the party won *last* time): "
              f"**{100 * placebo['est']:+.1f} pp** (p = {placebo['p']:.2f}).")
    md.append(f"- Coverage: {df['state'].nunique()} states/UTs, elections {df['year'].min()}–{df['year'].max()}, "
              f"{len(df):,} party-race observations with a matched next election.\n")
    md.append("### Robustness to bandwidth\n")
    md.append(md_table(rob, "{:.2f}"))
    md.append("\n### By decade (±7.5 pp)\n")
    show = dec[["Decade", "Party effect", "Candidate effect", "N (party)"]].copy()
    show[["Party effect", "Candidate effect"]] *= 100
    md.append(md_table(show, "{:+.1f}"))
    md.append("\n### Figures\n")
    for f, cap in [(f1, "RD plots"), (f2, "Incumbency effect by decade"), (f3, "By state")]:
        md.append(f"**{cap}**\n\n![{cap}]({f})\n")
    write_results("\n".join(md))
    print("done")


if __name__ == "__main__":
    main()
