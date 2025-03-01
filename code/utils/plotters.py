import numpy as np
import pandas as pd
from config import PLOT_STYLE
from code.utils.helpers import add_footer
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use(PLOT_STYLE)

def plot_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ordered_labels: list
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        palette="coolwarm",
        orient="h",
        showfliers=False,
        order=ordered_labels,
        ax=ax
    )

    for i, label in enumerate(ordered_labels):
        filtered_data = data.loc[data[y] == label, x]
        if not filtered_data.empty:
            median_value = filtered_data.median()
            if pd.notnull(median_value) and np.isfinite(median_value):
                ax.text(
                    median_value, i, f"%{median_value:.0f}",
                    va="center", ha="right", fontsize=9, color="black", weight="bold"
                )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=35)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    add_footer(fig)
    st.pyplot(fig)

def plot_stacked_bar_chart(
    data: pd.DataFrame,
    stat_columns: list,
    total_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: dict,
    sort_by: str = None,
    ascending: bool = False
) -> None:
    data["Remaining"] = data[total_column] - data[stat_columns].sum(axis=1)

    if sort_by:
        data = data.sort_values(by=sort_by, ascending=ascending)

    fig, ax = plt.subplots(figsize=(10, 10))
    bottom_value = 0

    for col in stat_columns:
        ax.barh(data.index, data[col], label=col, color=colors.get(col, "#808080"), left=bottom_value)
        bottom_value += data[col]

    for i, row in data.iterrows():
        total = row[total_column]
        if total > 0:
            percentages = [(row[col] / total * 100) for col in stat_columns]
            start = 0
            for col, percent in zip(stat_columns, percentages):
                ax.text(start + row[col] / 2, i, f"%{percent:.1f}", ha="center", va="center", fontsize=9, color="black")
                start += row[col]

    ax.set_title(title, fontsize=12, fontweight="bold", pad=35)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax.set_ylabel(ylabel)
    ax.legend(
        title="",
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False
    )
    add_footer(fig)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

def plot_horizontal_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    percentages: list = None,
    calculate_percentages: bool = False,
    sort_by: str = None,
    ascending: bool = False,
    cmap_name: str = "coolwarm_r",
    is_int: bool = False
) -> None:
    if sort_by:
        data = data.sort_values(by=sort_by, ascending=ascending)

    if calculate_percentages and percentages is None:
        total = data[x].sum()
        percentages = (data[x] / total * 100).round(1)

    cmap = plt.get_cmap(cmap_name)
    normalize = plt.Normalize(vmin=data[x].min(), vmax=data[x].max())
    colors = cmap(normalize(data[x]))

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(data[y], data[x], color=colors)

    for bar, value in zip(bars, percentages if calculate_percentages else data[x]):
        if is_int:
            formatted_value = f"{int(value)}"
        else:
            formatted_value = f"{value:.2f}" if abs(value) < 1 else f"{value:.1f}"

        if value < 0:
            offset = -0.3 if abs(value) >= 1 else -0.05
            ha = "right"
        else:
            offset = 0.005 if value < 1 else 0.3
            ha = "left"
        if abs(bar.get_width()) > 0:
            ax.text(
                bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                formatted_value, va="center", ha=ha, fontsize=9, color="black"
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=35)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    add_footer(fig)
    st.pyplot(fig)

def plot_stacked_horizontal_bar(
    data: pd.DataFrame,
    stat_columns: list,
    total_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: dict,
    sort_by: str = None,
    ascending: bool = False
) -> None:
    if sort_by:
        data = data.sort_values(by=sort_by, ascending=ascending)

    fig, ax = plt.subplots(figsize=(10, 10))
    bottom_value = 0

    for col in stat_columns:
        ax.barh(data.index, data[col], label=col, color=colors.get(col, "#808080"), left=bottom_value)
        bottom_value += data[col]

    for i, row in data.iterrows():
        total = row[total_column]
        if total > 0:
            percentages = [(row[col] / total * 100) for col in stat_columns]
            start = 0
            for col, percent in zip(stat_columns, percentages):
                ax.text(start + row[col] / 2, i, f"%{percent:.1f}", ha="center", va="center", fontsize=9, color="black")
                start += row[col]

    ax.set_title(title, fontsize=12, fontweight="bold", pad=35)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax.set_ylabel(ylabel)
    ax.legend(
        title="",
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=False
    )
    add_footer(fig)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)