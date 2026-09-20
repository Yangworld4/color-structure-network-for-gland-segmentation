#!/usr/bin/env python3
"""Interactive comparison plotter for milestone CSV files.

Default input directory:
    /data/mission/models/Exp/DUAL/outputs

Example:
    python analysis/plot_csv_metrics_gui.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


DEFAULT_OUTPUT_DIR = Path("/data/mission/models/Exp/DUAL/outputs")


def is_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], []
        rows = list(reader)
        return rows, [name for name in reader.fieldnames if name]


def numeric_columns(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    result = []
    for column in columns:
        if column.lower() == "epoch":
            continue
        values = [row.get(column, "") for row in rows]
        if values and any(is_number(value) for value in values):
            result.append(column)
    return result


class CsvMetricPlotter:
    def __init__(self, root: tk.Tk, output_dir: Path):
        self.root = root
        self.output_dir = output_dir
        self.root.title("CSV 指标对比折线图")
        self.root.geometry("1320x820")
        self.root.minsize(980, 620)

        self.files: list[Path] = []
        self.file_rows: dict[Path, list[dict[str, str]]] = {}
        self.metric_vars: dict[str, tk.BooleanVar] = {}

        self._build_layout()
        self.refresh_files()

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        control = ttk.Frame(self.root, padding=8)
        control.grid(row=0, column=0, sticky="ns")
        plot_frame = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(1, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        ttk.Label(control, text="CSV 文件（可多选）", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        file_box_frame = ttk.Frame(control)
        file_box_frame.pack(fill="both", expand=False, pady=(4, 8))
        self.file_list = tk.Listbox(
            file_box_frame,
            selectmode=tk.EXTENDED,
            width=34,
            height=10,
            exportselection=False,
        )
        file_scroll = ttk.Scrollbar(file_box_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scroll.set)
        self.file_list.grid(row=0, column=0, sticky="nsew")
        file_scroll.grid(row=0, column=1, sticky="ns")
        file_box_frame.columnconfigure(0, weight=1)

        file_buttons = ttk.Frame(control)
        file_buttons.pack(fill="x", pady=(0, 12))
        ttk.Button(file_buttons, text="刷新 CSV", command=self.refresh_files).pack(side="left")
        ttk.Button(file_buttons, text="全选文件", command=self.select_all_files).pack(side="left", padx=4)
        ttk.Button(file_buttons, text="清空文件", command=self.clear_files).pack(side="left")

        ttk.Label(control, text="指标（可多选）", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        metric_frame = ttk.Frame(control)
        metric_frame.pack(fill="both", expand=True, pady=(4, 8))
        metric_frame.rowconfigure(0, weight=1)
        metric_frame.columnconfigure(0, weight=1)
        self.metric_canvas = tk.Canvas(metric_frame, width=270, highlightthickness=0)
        metric_scroll = ttk.Scrollbar(metric_frame, orient="vertical", command=self.metric_canvas.yview)
        self.metric_inner = ttk.Frame(self.metric_canvas)
        self.metric_inner.bind(
            "<Configure>",
            lambda _event: self.metric_canvas.configure(scrollregion=self.metric_canvas.bbox("all")),
        )
        self.metric_canvas.create_window((0, 0), window=self.metric_inner, anchor="nw")
        self.metric_canvas.configure(yscrollcommand=metric_scroll.set)
        self.metric_canvas.grid(row=0, column=0, sticky="nsew")
        metric_scroll.grid(row=0, column=1, sticky="ns")

        metric_buttons = ttk.Frame(control)
        metric_buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(metric_buttons, text="全选指标", command=self.select_all_metrics).pack(side="left")
        ttk.Button(metric_buttons, text="清空指标", command=self.clear_metrics).pack(side="left", padx=4)

        ttk.Button(control, text="绘制折线图", command=self.plot).pack(fill="x", pady=(4, 4))
        ttk.Button(control, text="保存当前图像", command=self.save_figure).pack(fill="x")
        self.status = ttk.Label(control, text="请选择文件和指标", foreground="#555555", wraplength=280)
        self.status.pack(anchor="w", pady=(12, 0))

        self.figure = Figure(figsize=(8.8, 6.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("CSV 指标对比")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="w")

    def refresh_files(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.files = sorted(self.output_dir.glob("*.csv"), key=lambda path: path.name.lower())
        self.file_rows = {}
        self.file_list.delete(0, tk.END)
        for path in self.files:
            try:
                rows, _columns = read_csv(path)
                self.file_rows[path] = rows
                self.file_list.insert(tk.END, path.name)
            except Exception as exc:  # keep one malformed CSV from breaking the GUI
                self.file_list.insert(tk.END, f"{path.name} [读取失败]")
                self.status.config(text=f"读取 {path.name} 失败：{exc}")

        self.rebuild_metrics()
        if self.files:
            self.file_list.selection_set(0, tk.END)
            self.status.config(text=f"已发现 {len(self.files)} 个 CSV。请选择指标后绘图。")
        else:
            self.status.config(text=f"目录中没有 CSV：{self.output_dir}")

    def rebuild_metrics(self) -> None:
        for child in self.metric_inner.winfo_children():
            child.destroy()
        self.metric_vars.clear()

        all_metrics: set[str] = set()
        for path in self.files:
            rows = self.file_rows.get(path, [])
            _rows, columns = read_csv(path) if path not in self.file_rows else (rows, list(rows[0]) if rows else [])
            all_metrics.update(numeric_columns(rows, columns))

        preferred = [
            "dice_mean",
            "agreement_rate",
            "WW_rate",
            "HC90WW_rate",
            "dice_student1",
            "dice_student2",
        ]
        ordered = [name for name in preferred if name in all_metrics]
        ordered += sorted(all_metrics.difference(ordered), key=str.lower)
        for metric in ordered:
            var = tk.BooleanVar(value=metric in {"dice_mean", "agreement_rate", "WW_rate", "HC90WW_rate"})
            self.metric_vars[metric] = var
            ttk.Checkbutton(self.metric_inner, text=metric, variable=var).pack(anchor="w")

    def select_all_files(self) -> None:
        if self.files:
            self.file_list.selection_set(0, tk.END)

    def clear_files(self) -> None:
        self.file_list.selection_clear(0, tk.END)

    def select_all_metrics(self) -> None:
        for var in self.metric_vars.values():
            var.set(True)

    def clear_metrics(self) -> None:
        for var in self.metric_vars.values():
            var.set(False)

    def selected_paths(self) -> list[Path]:
        return [self.files[index] for index in self.file_list.curselection() if index < len(self.files)]

    def selected_metrics(self) -> list[str]:
        return [metric for metric, var in self.metric_vars.items() if var.get()]

    @staticmethod
    def numeric_series(rows: list[dict[str, str]], column: str) -> tuple[list[float], list[float]]:
        x_values = []
        y_values = []
        for row in rows:
            x_raw = row.get("epoch", "")
            y_raw = row.get(column, "")
            if is_number(x_raw) and is_number(y_raw):
                x_values.append(float(x_raw))
                y_values.append(float(y_raw))
        order = sorted(range(len(x_values)), key=lambda index: x_values[index])
        return [x_values[index] for index in order], [y_values[index] for index in order]

    def plot(self) -> None:
        paths = self.selected_paths()
        metrics = self.selected_metrics()
        if not paths:
            messagebox.showwarning("缺少选择", "请至少选择一个 CSV 文件。")
            return
        if not metrics:
            messagebox.showwarning("缺少选择", "请至少选择一个指标。")
            return

        self.ax.clear()
        line_count = 0
        for path in paths:
            rows = self.file_rows.get(path, [])
            for metric in metrics:
                x_values, y_values = self.numeric_series(rows, metric)
                if not x_values:
                    continue
                self.ax.plot(x_values, y_values, marker="o", linewidth=1.7, markersize=3.5, label=f"{path.stem} | {metric}")
                line_count += 1

        self.ax.set_title("CSV 指标对比")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        if line_count:
            self.ax.legend(loc="best", fontsize=8)
            self.status.config(text=f"已绘制 {line_count} 条曲线。")
        else:
            self.status.config(text="选中的文件没有可绘制的 epoch/数值数据。")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_figure(self) -> None:
        target = filedialog.asksaveasfilename(
            title="保存折线图",
            initialdir=str(self.output_dir),
            initialfile="csv_metric_comparison.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not target:
            return
        self.figure.savefig(target, dpi=200, bbox_inches="tight")
        self.status.config(text=f"图像已保存：{target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式选择 outputs 下 CSV 和指标并绘制折线图")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV 所在目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    root = tk.Tk()
    CsvMetricPlotter(root, args.output_dir.resolve())
    root.mainloop()


if __name__ == "__main__":
    main()
