# in6227_experiment.py
# IN6227 A2: Measured vs Estimated runtime for frequent itemset mining
# - Reads Groceries_dataset.csv (Member_number, Date, itemDescription)
# - Builds datasets by varying T (transactions) and d (unique items)
# - Runs a compact Apriori (no external libs) up to k<=3 and times it
# - Fits brute-force constant C using f(d,T)=T*sum_{k=1..kmax} C(d,k)
# - Saves results CSV and one comparison plot (PNG)

import os
import time
import itertools
import math
from collections import Counter
from math import comb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print("Current working directory:", os.getcwd())



# ---------------------- Config ----------------------
INPUT_CSV = "Groceries_dataset.csv"
MIN_SUPPORT = 0.02
KMAX = 3

# vary T with fixed d, and vary d with fixed T
T_VARIANTS = [1000, 2000, 4000, 8000]     # will be trimmed if dataset smaller
D_VARIANTS = [20, 35, 50, 65]             # will be trimmed by available items

# output files
OUT_CSV = "in6227_results.csv"
OUT_PNG = "in6227_plot.png"
# ----------------------------------------------------


def load_transactions(path: str):
    """Load groceries CSV and group by (Member_number, Date) into baskets."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find '{path}'. Place it in the same folder as this script."
        )
    df = pd.read_csv(path)
    required = {"Member_number", "Date", "itemDescription"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required}, got {df.columns.tolist()}"
        )
    # Robust date parsing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
    grp = (
        df.groupby(["Member_number", "Date"])["itemDescription"]
        .apply(list)
        .reset_index(name="items")
    )
    # ensure strings
    transactions = [list(map(str, t)) for t in grp["items"].tolist()]
    # remove empty
    transactions = [t for t in transactions if len(t) > 0]
    return transactions


def top_k_items(transactions, k):
    cnt = Counter(itertools.chain.from_iterable(transactions))
    return [it for it, _ in cnt.most_common(k)]


def filter_transactions(transactions, keep_items):
    """Keep only items in keep_items; drop empty baskets."""
    keep = set(keep_items)
    tx = [[x for x in t if x in keep] for t in transactions]
    tx = [t for t in tx if len(t) > 0]
    return tx


def apriori_custom(transactions, min_support=0.02, max_k=3):
    """
    Minimal Apriori implementation (up to max_k) for timing and counts.
    Returns:
      L: dict[k] -> {frozenset(items): support_count}
      fis_list: list of (frozenset, support_fraction)
    """
    T = len(transactions)
    tx_sets = [set(t) for t in transactions]

    # L1
    item_counts = Counter(itertools.chain.from_iterable(tx_sets))
    L = {}
    L[1] = {frozenset([i]): c for i, c in item_counts.items() if c / T >= min_support}

    for k in range(2, max_k + 1):
        prev_Lk = L.get(k - 1, {})
        if not prev_Lk:
            break
        prev_itemsets = list(prev_Lk.keys())

        # join
        candidates = set()
        n_prev = len(prev_itemsets)
        for i in range(n_prev):
            for j in range(i + 1, n_prev):
                u = prev_itemsets[i] | prev_itemsets[j]
                if len(u) == k:
                    # prune: all (k-1)-subsets frequent
                    if all((u - frozenset([x])) in prev_Lk for x in u):
                        candidates.add(u)

        # count
        cand_counts = {c: 0 for c in candidates}
        for tx in tx_sets:
            for c in candidates:
                if c.issubset(tx):
                    cand_counts[c] += 1

        L[k] = {c: cnt for c, cnt in cand_counts.items() if cnt / T >= min_support}

    fis_list = []
    for k, d in L.items():
        for itset, cnt in d.items():
            fis_list.append((itset, cnt / T))
    return L, fis_list


def brute_force_model_T_d(T, d, kmax=3):
    """f(d,T) = T * sum_{k=1..kmax} C(d,k)"""
    return T * sum(comb(d, k) for k in range(1, kmax + 1))


def fit_C_single_point(T, d, measured_time, kmax=3):
    fval = brute_force_model_T_d(int(T), int(d), kmax=kmax)
    return measured_time / fval if fval > 0 else np.nan

def _sort_for_group(df):
    if df.iloc[0]["group"] == "vary_T":
        return df.sort_values("T")
    else:
        return df.sort_values("d")

def main():
    # 1) Load data and basic stats
    transactions = load_transactions(INPUT_CSV)
    T_total = len(transactions)
    print(f"Total grouped transactions: {T_total}")

    # anchor items by global frequency (ensure we can reach largest d)
    max_d_needed = max(D_VARIANTS + [50])
    global_top = top_k_items(transactions, max_d_needed)
    if len(global_top) < max_d_needed:
        print(f"[Warn] Only {len(global_top)} unique items found; trimming variants.")
    # Trim T_VARIANTS by available T
    T_vars = [t for t in T_VARIANTS if t <= T_total]
    if not T_vars:
        # very small dataset fallback
        T_vars = [min(500, max(1, T_total // 2)), min(1000, T_total)]
        T_vars = sorted(set([t for t in T_vars if t > 0]))
    # Trim D_VARIANTS by available items
    D_vars = [d for d in D_VARIANTS if d <= len(global_top)]
    if not D_vars:
        D_vars = [min(20, len(global_top))]

    # 2) Build experiment variants
    # A) vary T with fixed d (prefer 50)
    d_fixed = 50 if len(global_top) >= 50 else min(30, len(global_top))
    keep_items_T = global_top[:d_fixed]

    # B) vary d with fixed T (prefer largest T)
    T_fixed = max(T_vars) if len(T_vars) > 0 else min(T_total, 2000)

    results = []

    # 3) Run experiments & time Apriori
    print("\n[Group A] Vary T, fixed d={}".format(d_fixed))
    for T in T_vars:
        tx_subset = transactions[:T]
        tx_filt = filter_transactions(tx_subset, keep_items_T)
        T_eff = len(tx_filt)
        t0 = time.perf_counter()
        L, fis = apriori_custom(tx_filt, min_support=MIN_SUPPORT, max_k=KMAX)
        t1 = time.perf_counter()
        measured = t1 - t0
        results.append({
            "group": "vary_T",
            "dataset": f"T{T}_d{d_fixed}",
            "T": T_eff,
            "d": len(keep_items_T),
            "min_support": MIN_SUPPORT,
            "kmax": KMAX,
            "measured_time_sec": measured,
            "num_itemsets": len(fis),
        })
        print(f"  T={T_eff:>5}, d={len(keep_items_T):>2} -> time={measured:.6f}s, itemsets={len(fis)}")

    print("\n[Group B] Vary d, fixed T={}".format(T_fixed))
    tx_Tfixed = transactions[:T_fixed]
    for d in D_vars:
        keep_items_d = global_top[:d]
        tx_filt = filter_transactions(tx_Tfixed, keep_items_d)
        T_eff = len(tx_filt)
        t0 = time.perf_counter()
        L, fis = apriori_custom(tx_filt, min_support=MIN_SUPPORT, max_k=KMAX)
        t1 = time.perf_counter()
        measured = t1 - t0
        results.append({
            "group": "vary_d",
            "dataset": f"T{T_fixed}_d{d}",
            "T": T_eff,
            "d": len(keep_items_d),
            "min_support": MIN_SUPPORT,
            "kmax": KMAX,
            "measured_time_sec": measured,
            "num_itemsets": len(fis),
        })
        print(f"  T={T_eff:>5}, d={len(keep_items_d):>2} -> time={measured:.6f}s, itemsets={len(fis)}")

    res_df = pd.DataFrame(results)

    # 4) Fit C per group using the first row of each group, then predict all
    def fit_C_group(sub_df):
        first = sub_df.iloc[0]
        return fit_C_single_point(first["T"], first["d"], first["measured_time_sec"], kmax=KMAX)

    C_by_group = {}
    for grp, sub in res_df.groupby("group"):
        C_by_group[grp] = fit_C_group(sub)

    def predict_time(row):
        C = C_by_group.get(row["group"], np.nan)
        fval = brute_force_model_T_d(int(row["T"]), int(row["d"]), kmax=KMAX)
        return C * fval if not np.isnan(C) else np.nan

    res_df["estimated_time_sec"] = res_df.apply(predict_time, axis=1)

    # 5) Save outputs
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to {OUT_CSV}")

    # 6) Plot measured vs estimated
    plt.figure(figsize=(11, 5))
    x = np.arange(len(res_df))
    plt.plot(x, res_df["measured_time_sec"].values, marker="o", label="Measured (Apriori, k<=3)")
    plt.plot(x, res_df["estimated_time_sec"].values, marker="s", label="Estimated (Brute-force, k<=3)")
    plt.xticks(x, res_df["dataset"].tolist(), rotation=45, ha="right")
    plt.xlabel("Dataset (T_d)")
    plt.ylabel("Time (seconds)")
    plt.title("Measured vs Estimated Time for Frequent Itemset Mining")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    for grp, sub in res_df.groupby("group"):
        sub = _sort_for_group(sub)
        plt.figure(figsize=(8, 4))
        # 用数据集标签作为横轴，便于和总图一致
        plt.plot(sub["dataset"], sub["measured_time_sec"], marker="o", label="Measured (Apriori)")
        plt.plot(sub["dataset"], sub["estimated_time_sec"], marker="s", label="Estimated (Brute-force)")
        plt.title(f"Measured vs Estimated Time — {grp}")
        plt.xlabel("Dataset")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        filename = f"in6227_plot_{grp}.png"
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f" Saved plot: {filename}")




    # 7) Print brief summary
    print("\n=== Summary ===")
    print(res_df[["group", "dataset", "T", "d", "measured_time_sec", "estimated_time_sec", "num_itemsets"]])
    print("\nFitted C by group:", C_by_group)


if __name__ == "__main__":
    main()


    # === 追加：分别绘制 vary_T / vary_d 两张独立图 ===
def _sort_for_group(df):
    """让横轴更好看：vary_T 按 T 升序；vary_d 按 d 升序。"""
    sub = df.copy()
    if sub.iloc[0]["group"] == "vary_T":
        return sub.sort_values(["T", "d"])
    else:
        return sub.sort_values(["d", "T"])

