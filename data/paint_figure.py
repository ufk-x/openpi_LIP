"""
绘制 LIBERO 评估结果对比图：RTC True vs False
上子图：成功率（含 95% Wilson score interval）
下子图：平均回合长度（含 95% 置信区间）

文件名格式: eval_results_delay{d}_replan{r}_rtc{True|False}.csv
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# ★ 在此处配置要对比的 (delay, replan) 组合
#   两个数组一一对应：delays[i] 与 replans[i] 构成一组
# ============================================================
DELAYS  = [1, 2, 3, 4, 5]
REPLANS = [5, 5, 5, 5, 5]

# TASK_SUITE = ""
TASK_SUITE = "libero_10"
CSV_DIR = pathlib.Path(__file__).parent / "libero"
FIG_OUT = pathlib.Path(__file__).parent / "libero" / "rtc_comparison.png"


# ---- Wilson score interval ------------------------------------------------
def wilson_ci(successes: int, total: int, z: float = 1.96):
    """计算 Wilson score 95% 置信区间，返回 (lower, upper)。"""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    margin = z / denom * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---- 读取并聚合数据 -------------------------------------------------------
def load_one(delay: int, replan: int, rtc: bool):
    """读取一个 CSV 并返回跨任务聚合统计。

    Returns:
        dict with keys: delay, replan, rtc,
              mean_sr, sr_lo, sr_hi,        (成功率 & Wilson CI)
              mean_len, len_lo, len_hi       (平均长度 & 95% CI)
        如果文件不存在返回 None。
    """
    tag = "True" if rtc else "False"
    # fname = CSV_DIR / f"eval_results_delay{delay}_replan{replan}_rtc{tag}.csv"
    fname = CSV_DIR / f"{TASK_SUITE}_results_delay{delay}_replan{replan}_rtc{tag}.csv"
    if not fname.exists():
        print(f"[跳过] 文件不存在: {fname.name}")
        return None

    df = pd.read_csv(fname)

    # --- 成功率（Wilson score interval）---
    total_succ = int(df["successes"].sum())
    total_ep   = int(df["num_episodes"].sum())
    mean_sr = total_succ / total_ep if total_ep > 0 else 0.0
    sr_lo, sr_hi = wilson_ci(total_succ, total_ep)

    # --- 平均回合长度（均值 ± 1.96 * SEM）---
    lengths = df["avg_episode_length"].values
    mean_len = float(np.mean(lengths))
    sem_len  = float(np.std(lengths, ddof=1) / np.sqrt(len(lengths))) if len(lengths) > 1 else 0.0
    len_lo = mean_len - 1.96 * sem_len
    len_hi = mean_len + 1.96 * sem_len

    return dict(
        delay=delay, replan=replan, rtc=rtc,
        mean_sr=mean_sr, sr_lo=sr_lo, sr_hi=sr_hi,
        mean_len=mean_len, len_lo=len_lo, len_hi=len_hi,
    )


def main():
    assert len(DELAYS) == len(REPLANS), "DELAYS 和 REPLANS 数组长度必须一致"

    # 收集数据
    records_true, records_false = [], []
    for d, r in zip(DELAYS, REPLANS):
        rt = load_one(d, r, rtc=True)
        rf = load_one(d, r, rtc=False)
        if rt is not None:
            records_true.append(rt)
        if rf is not None:
            records_false.append(rf)

    if not records_true and not records_false:
        print("没有找到任何有效的 CSV 文件，退出。")
        return

    # X 轴标签: "d=1,r=7" 格式
    x_labels = [f"d={d},r={r}" for d, r in zip(DELAYS, REPLANS)]

    # 构建 numpy 数组，便于绘图
    def to_arrays(records):
        idx_map = {(rec["delay"], rec["replan"]): rec for rec in records}
        mean_sr, lo_sr, hi_sr = [], [], []
        mean_len, lo_len, hi_len = [], [], []
        valid_mask = []
        for d, r in zip(DELAYS, REPLANS):
            rec = idx_map.get((d, r))
            if rec is not None:
                mean_sr.append(rec["mean_sr"]); lo_sr.append(rec["sr_lo"]); hi_sr.append(rec["sr_hi"])
                mean_len.append(rec["mean_len"]); lo_len.append(rec["len_lo"]); hi_len.append(rec["len_hi"])
                valid_mask.append(True)
            else:
                mean_sr.append(np.nan); lo_sr.append(np.nan); hi_sr.append(np.nan)
                mean_len.append(np.nan); lo_len.append(np.nan); hi_len.append(np.nan)
                valid_mask.append(False)
        return (np.array(mean_sr), np.array(lo_sr), np.array(hi_sr),
                np.array(mean_len), np.array(lo_len), np.array(hi_len),
                np.array(valid_mask))

    sr_t, sr_lo_t, sr_hi_t, len_t, len_lo_t, len_hi_t, mask_t = to_arrays(records_true)
    sr_f, sr_lo_f, sr_hi_f, len_f, len_lo_f, len_hi_f, mask_f = to_arrays(records_false)

    x = np.arange(len(DELAYS))

    # ---- 绘图 ----
    fig, (ax_sr, ax_len) = plt.subplots(2, 1, figsize=(max(10, len(DELAYS) * 0.9), 8),
                                         sharex=True, gridspec_kw={"hspace": 0.12})

    color_true  = "#1f77b4"  # 蓝
    color_false = "#d62728"  # 红

    # -- 上子图：成功率 --
    if mask_t.any():
        ax_sr.plot(x[mask_t], sr_t[mask_t], "o-", color=color_true, label="RTC = True", linewidth=2, markersize=6)
        ax_sr.fill_between(x[mask_t], sr_lo_t[mask_t], sr_hi_t[mask_t], color=color_true, alpha=0.15)
    if mask_f.any():
        ax_sr.plot(x[mask_f], sr_f[mask_f], "s--", color=color_false, label="RTC = False", linewidth=2, markersize=6)
        ax_sr.fill_between(x[mask_f], sr_lo_f[mask_f], sr_hi_f[mask_f], color=color_false, alpha=0.15)

    ax_sr.set_ylabel("Success Rate", fontsize=13)
    # 自适应 Y 轴上下限：根据数据范围留 10% padding，让差异更明显
    all_sr_lo = np.concatenate([v for v in [sr_lo_t[mask_t], sr_lo_f[mask_f]] if len(v) > 0])
    all_sr_hi = np.concatenate([v for v in [sr_hi_t[mask_t], sr_hi_f[mask_f]] if len(v) > 0])
    sr_min, sr_max = float(np.nanmin(all_sr_lo)), float(np.nanmax(all_sr_hi))
    sr_pad = max((sr_max - sr_min) * 0.15, 0.01)  # 至少留 1% 余量
    ax_sr.set_ylim(max(0, sr_min - sr_pad), min(1.02, sr_max + sr_pad))
    ax_sr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_sr.legend(fontsize=11, loc="lower left")
    ax_sr.grid(axis="y", alpha=0.3)
    ax_sr.set_title("LIBERO-10: RTC Guidance Comparison", fontsize=14, fontweight="bold")

    # -- 下子图：平均回合长度 --
    if mask_t.any():
        ax_len.plot(x[mask_t], len_t[mask_t], "o-", color=color_true, label="RTC = True", linewidth=2, markersize=6)
        ax_len.fill_between(x[mask_t], len_lo_t[mask_t], len_hi_t[mask_t], color=color_true, alpha=0.15)
    if mask_f.any():
        ax_len.plot(x[mask_f], len_f[mask_f], "s--", color=color_false, label="RTC = False", linewidth=2, markersize=6)
        ax_len.fill_between(x[mask_f], len_lo_f[mask_f], len_hi_f[mask_f], color=color_false, alpha=0.15)

    ax_len.set_ylabel("Avg Episode Length", fontsize=13)
    ax_len.legend(fontsize=11, loc="upper left")
    ax_len.grid(axis="y", alpha=0.3)

    # X 轴标签
    ax_len.set_xticks(x)
    ax_len.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax_len.set_xlabel("(delay, replan) Setting", fontsize=13)

    fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    print(f"图表已保存至 {FIG_OUT}")
    plt.show()


if __name__ == "__main__":
    main()
