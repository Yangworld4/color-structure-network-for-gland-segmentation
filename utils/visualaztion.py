import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import matplotlib.pyplot as plt


class MetricsVisualizer:
    def __init__(self, root):
        self.root = root

        self.root.title("SSL Experiment Visualizer")
        self.root.geometry("1100x850")

        self.root_dir = None
        self.output_dir = None

        # ========================================================
        # data_files
        #
        # {
        #   "semi_xxx+metrics": {
        #       "path": Path,
        #       "type": "metrics",
        #       "x_col": "iteration",
        #       "df": DataFrame
        #   },
        #
        #   "semi_xxx+train": {
        #       "path": Path,
        #       "type": "train",
        #       "x_col": "epoch",
        #       "df": DataFrame
        #   }
        # }
        # ========================================================

        self.data_files = {}

        self.selected_files = []

        self.build_ui()

    # ============================================================
    # UI
    # ============================================================

    def build_ui(self):

        # ========================================================
        # Root directory
        # ========================================================

        frame_root = tk.Frame(self.root)

        frame_root.pack(
            fill="x",
            padx=10,
            pady=10
        )

        tk.Button(
            frame_root,
            text="选择根目录",
            command=self.select_root_dir,
            width=20
        ).pack(side="left")

        self.root_label = tk.Label(
            frame_root,
            text="尚未选择目录",
            anchor="w"
        )

        self.root_label.pack(
            side="left",
            padx=10,
            fill="x",
            expand=True
        )

        # ========================================================
        # Experiment / file selection
        # ========================================================

        frame_files = tk.LabelFrame(
            self.root,
            text="Step 1：选择需要比较的数据"
        )

        frame_files.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        files_container = tk.Frame(frame_files)

        files_container.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        scrollbar_files = tk.Scrollbar(
            files_container
        )

        scrollbar_files.pack(
            side="right",
            fill="y"
        )

        self.file_listbox = tk.Listbox(
            files_container,
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar_files.set,
            height=17
        )

        self.file_listbox.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar_files.config(
            command=self.file_listbox.yview
        )

        # ========================================================
        # File buttons
        # ========================================================

        file_button_frame = tk.Frame(frame_files)

        file_button_frame.pack(
            fill="x",
            pady=5
        )

        tk.Button(
            file_button_frame,
            text="全选",
            command=self.select_all_files,
            width=12
        ).pack(
            side="left",
            padx=5
        )

        tk.Button(
            file_button_frame,
            text="取消全选",
            command=self.clear_file_selection,
            width=12
        ).pack(
            side="left",
            padx=5
        )

        tk.Button(
            file_button_frame,
            text="读取所选数据",
            command=self.load_selected_files,
            width=20
        ).pack(
            side="left",
            padx=5
        )

        # ========================================================
        # Metric selection
        # ========================================================

        frame_metrics = tk.LabelFrame(
            self.root,
            text="Step 2：选择需要绘制的指标"
        )

        frame_metrics.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        metrics_container = tk.Frame(
            frame_metrics
        )

        metrics_container.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        scrollbar_metrics = tk.Scrollbar(
            metrics_container
        )

        scrollbar_metrics.pack(
            side="right",
            fill="y"
        )

        self.metric_listbox = tk.Listbox(
            metrics_container,
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar_metrics.set,
            height=15
        )

        self.metric_listbox.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar_metrics.config(
            command=self.metric_listbox.yview
        )

        # ========================================================
        # Metric buttons
        # ========================================================

        metric_button_frame = tk.Frame(
            frame_metrics
        )

        metric_button_frame.pack(
            fill="x",
            pady=5
        )

        tk.Button(
            metric_button_frame,
            text="全选指标",
            command=self.select_all_metrics,
            width=12
        ).pack(
            side="left",
            padx=5
        )

        tk.Button(
            metric_button_frame,
            text="取消全选",
            command=self.clear_metric_selection,
            width=12
        ).pack(
            side="left",
            padx=5
        )

        # ========================================================
        # Output
        # ========================================================

        frame_output = tk.Frame(
            self.root
        )

        frame_output.pack(
            fill="x",
            padx=10,
            pady=10
        )

        tk.Button(
            frame_output,
            text="选择保存目录",
            command=self.select_output_dir,
            width=20
        ).pack(
            side="left"
        )

        self.output_label = tk.Label(
            frame_output,
            text="默认保存到：根目录/plots",
            anchor="w"
        )

        self.output_label.pack(
            side="left",
            padx=10,
            fill="x",
            expand=True
        )

        # ========================================================
        # Plot
        # ========================================================

        tk.Button(
            self.root,
            text="绘图并保存",
            command=self.plot_metrics,
            font=("Arial", 12, "bold"),
            height=2
        ).pack(
            fill="x",
            padx=10,
            pady=10
        )

    # ============================================================
    # Select root
    # ============================================================

    def select_root_dir(self):

        folder = filedialog.askdirectory(
            title="选择 checkpoints 或其上级目录"
        )

        if not folder:
            return

        self.root_dir = Path(folder)

        self.root_label.config(
            text=str(self.root_dir)
        )

        self.output_dir = (
            self.root_dir / "plots"
        )

        self.output_label.config(
            text=str(self.output_dir)
        )

        self.scan_files()

    # ============================================================
    # Get experiment name
    # ============================================================

    @staticmethod
    def get_experiment_name(file_path, root_dir):
        """
        Example:

        /data/.../checkpoints/
        semi_10_seed14_rgbstain5/
        GLaS/
        CPS-xxx/
        metrics.csv

        -> semi_10_seed14_rgbstain5

        Same for train.log.
        """

        parts = file_path.parts

        # ========================================================
        # Prefer checkpoints/xxx
        # ========================================================

        if "checkpoints" in parts:

            idx = parts.index(
                "checkpoints"
            )

            if idx + 1 < len(parts):

                return parts[idx + 1]

        # ========================================================
        # Fallback
        # ========================================================

        try:

            relative = (
                file_path.relative_to(
                    root_dir
                )
            )

            if len(relative.parts) > 1:

                return relative.parts[0]

        except ValueError:
            pass

        return file_path.parent.name

    # ============================================================
    # Scan metrics.csv + train.log
    # ============================================================

    def scan_files(self):

        self.file_listbox.delete(
            0,
            tk.END
        )

        self.metric_listbox.delete(
            0,
            tk.END
        )

        self.data_files.clear()

        self.selected_files = []

        # ========================================================
        # Search
        # ========================================================

        metrics_paths = list(
            self.root_dir.rglob(
                "metrics.csv"
            )
        )

        train_paths = list(
            self.root_dir.rglob(
                "train.log"
            )
        )

        all_files = []

        for path in metrics_paths:

            all_files.append(
                (path, "metrics")
            )

        for path in train_paths:

            all_files.append(
                (path, "train")
            )

        if not all_files:

            messagebox.showwarning(
                "Warning",
                "没有找到 metrics.csv 或 train.log"
            )

            return

        # ========================================================
        # Handle duplicate names
        # ========================================================

        name_count = {}

        for file_path, file_type in sorted(
            all_files,
            key=lambda x: str(x[0])
        ):

            try:

                experiment_name = (
                    self.get_experiment_name(
                        file_path,
                        self.root_dir
                    )
                )

                # =================================================
                # Display name
                # =================================================

                if file_type == "metrics":

                    base_name = (
                        f"{experiment_name}+metrics"
                    )

                    x_col = "iteration"

                else:

                    base_name = (
                        f"{experiment_name}+train"
                    )

                    x_col = "epoch"

                # =================================================
                # Duplicate protection
                # =================================================

                if base_name in name_count:

                    name_count[base_name] += 1

                    display_name = (
                        f"{base_name}_"
                        f"{name_count[base_name]}"
                    )

                else:

                    name_count[base_name] = 1

                    display_name = base_name

                # =================================================
                # Store
                # =================================================

                self.data_files[
                    display_name
                ] = {

                    "path": file_path,

                    "type": file_type,

                    "x_col": x_col,

                    "df": None
                }

                self.file_listbox.insert(
                    tk.END,
                    display_name
                )

            except Exception as e:

                print(
                    f"[Warning] {file_path}"
                )

                print(
                    f"原因：{e}"
                )

        # ========================================================
        # Print
        # ========================================================

        print("\n" + "=" * 90)

        print(
            f"metrics.csv: {len(metrics_paths)}"
        )

        print(
            f"train.log:   {len(train_paths)}"
        )

        print(
            f"总数据源:     {len(self.data_files)}"
        )

        print("=" * 90)

        for name, info in (
            self.data_files.items()
        ):

            print()

            print(name)

            print(
                f"  type = {info['type']}"
            )

            print(
                f"  x    = {info['x_col']}"
            )

            print(
                f"  path = {info['path']}"
            )

    # ============================================================
    # Parse metrics.csv
    # ============================================================

    @staticmethod
    def parse_metrics_csv(path):

        df = pd.read_csv(
            path
        )

        # ========================================================
        # Require iteration
        # ========================================================

        candidates = [
            "iteration",
            "Iteration",
            "iter",
            "Iter",
            "step",
            "Step",
            "global_step"
        ]

        iteration_col = None

        for col in candidates:

            if col in df.columns:

                iteration_col = col

                break

        if iteration_col is None:

            raise ValueError(
                "metrics.csv 中找不到 iteration / iter / step"
            )

        # ========================================================
        # Uniform naming
        # ========================================================

        if iteration_col != "iteration":

            df = df.rename(
                columns={
                    iteration_col: "iteration"
                }
            )

        return df

    # ============================================================
    # Convert train.log value to number
    # ============================================================

    @staticmethod
    def parse_numeric_value(value):

        value = value.strip()

        # ========================================================
        # Remove common suffix
        # ========================================================

        if value.endswith("s"):

            value = value[:-1]

        if value.endswith("%"):

            value = value[:-1]

        try:

            return float(value)

        except ValueError:

            return None

    # ============================================================
    # Parse train.log
    # ============================================================

    @staticmethod
    def parse_train_log(path):
        """
        Parse lines like:

        epoch=21/200 |
        learning_rate=0.5 |
        average_train_loss=0.331827 |
        val_dice_student1=0.759496 |
        val_dice_student2=0.673185 |
        ...

        Only actual epoch summary lines are read.

        Milestone / Periodic best lines are ignored.
        """

        rows = []

        # ========================================================
        # Match epoch=21/200
        # ========================================================

        epoch_pattern = re.compile(
            r"\bepoch=(\d+)/(\d+)\b"
        )

        # ========================================================
        # General key=value
        # ========================================================

        kv_pattern = re.compile(
            r"([A-Za-z_][A-Za-z0-9_]*)=([^|\s]+)"
        )

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            for line in f:

                # =================================================
                # Only epoch summary line
                # =================================================

                epoch_match = (
                    epoch_pattern.search(
                        line
                    )
                )

                if epoch_match is None:

                    continue

                epoch = int(
                    epoch_match.group(1)
                )

                total_epoch = int(
                    epoch_match.group(2)
                )

                row = {

                    "epoch": epoch,

                    "total_epoch": total_epoch
                }

                # =================================================
                # Find all key=value
                # =================================================

                matches = (
                    kv_pattern.findall(
                        line
                    )
                )

                for key, value in matches:

                    # epoch handled separately
                    if key == "epoch":
                        continue

                    numeric_value = (
                        MetricsVisualizer
                        .parse_numeric_value(
                            value
                        )
                    )

                    if numeric_value is None:
                        continue

                    row[key] = numeric_value

                rows.append(
                    row
                )

        if len(rows) == 0:

            raise ValueError(
                "train.log 中没有找到 epoch=x/x 格式的训练记录"
            )

        df = pd.DataFrame(
            rows
        )

        # ========================================================
        # Avoid duplicate epochs
        # ========================================================

        df = (
            df
            .sort_values("epoch")
            .drop_duplicates(
                subset=["epoch"],
                keep="last"
            )
            .reset_index(drop=True)
        )

        return df

    # ============================================================
    # Select all
    # ============================================================

    def select_all_files(self):

        self.file_listbox.select_set(
            0,
            tk.END
        )

    def clear_file_selection(self):

        self.file_listbox.selection_clear(
            0,
            tk.END
        )

    def select_all_metrics(self):

        self.metric_listbox.select_set(
            0,
            tk.END
        )

    def clear_metric_selection(self):

        self.metric_listbox.selection_clear(
            0,
            tk.END
        )

    # ============================================================
    # Load selected
    # ============================================================

    def load_selected_files(self):

        selection = (
            self.file_listbox
            .curselection()
        )

        if not selection:

            messagebox.showwarning(
                "Warning",
                "请至少选择一个数据源"
            )

            return

        self.selected_files = [

            self.file_listbox.get(i)

            for i in selection
        ]

        all_metrics = set()

        failed = []

        print("\n" + "=" * 90)

        print("开始读取数据")

        print("=" * 90)

        for name in self.selected_files:

            info = self.data_files[
                name
            ]

            path = info["path"]

            file_type = info["type"]

            try:

                # =================================================
                # metrics.csv
                # =================================================

                if file_type == "metrics":

                    df = (
                        self.parse_metrics_csv(
                            path
                        )
                    )

                    x_col = "iteration"

                # =================================================
                # train.log
                # =================================================

                else:

                    df = (
                        self.parse_train_log(
                            path
                        )
                    )

                    x_col = "epoch"

                info["df"] = df

                info["x_col"] = x_col

                # =================================================
                # Collect numeric metrics
                # =================================================

                for col in df.columns:

                    if col in [
                        "iteration",
                        "epoch",
                        "total_epoch"
                    ]:
                        continue

                    if pd.api.types.is_numeric_dtype(
                        df[col]
                    ):

                        all_metrics.add(
                            col
                        )

                print()

                print(
                    f"数据源：{name}"
                )

                print(
                    f"Type：{file_type}"
                )

                print(
                    f"X：{x_col}"
                )

                print(
                    f"Shape：{df.shape}"
                )

                print(
                    f"Columns：{list(df.columns)}"
                )

            except Exception as e:

                failed.append(
                    f"{name}\n{e}"
                )

        # ========================================================
        # Update metric list
        # ========================================================

        self.metric_listbox.delete(
            0,
            tk.END
        )

        for metric in sorted(
            all_metrics
        ):

            self.metric_listbox.insert(
                tk.END,
                metric
            )

        if failed:

            messagebox.showwarning(
                "部分数据读取失败",
                "\n\n".join(failed)
            )

        if not all_metrics:

            messagebox.showwarning(
                "Warning",
                "没有找到数值指标"
            )

            return

        messagebox.showinfo(
            "读取完成",
            f"选择数据源：{len(self.selected_files)}\n"
            f"发现指标：{len(all_metrics)}"
        )

    # ============================================================
    # Output
    # ============================================================

    def select_output_dir(self):

        folder = filedialog.askdirectory(
            title="选择图片保存目录"
        )

        if not folder:
            return

        self.output_dir = Path(
            folder
        )

        self.output_label.config(
            text=str(self.output_dir)
        )

    # ============================================================
    # Safe filename
    # ============================================================

    @staticmethod
    def safe_filename(name):

        result = str(name)

        invalid_chars = [
            "/",
            "\\",
            ":",
            "*",
            "?",
            '"',
            "<",
            ">",
            "|",
            " "
        ]

        for char in invalid_chars:

            result = result.replace(
                char,
                "_"
            )

        return result

    # ============================================================
    # Plot one metric
    # ============================================================

    def plot_one_group(
        self,
        metric,
        x_col,
        sources
    ):

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        plotted = False

        for name in sources:

            info = self.data_files[
                name
            ]

            df = info["df"]

            if df is None:
                continue

            if metric not in df.columns:
                continue

            if x_col not in df.columns:
                continue

            # ====================================================
            # Data
            # ====================================================

            plot_df = df[
                [x_col, metric]
            ].copy()

            plot_df[x_col] = (
                pd.to_numeric(
                    plot_df[x_col],
                    errors="coerce"
                )
            )

            plot_df[metric] = (
                pd.to_numeric(
                    plot_df[metric],
                    errors="coerce"
                )
            )

            plot_df = (
                plot_df
                .dropna()
                .sort_values(x_col)
            )

            if len(plot_df) == 0:
                continue

            # ====================================================
            # Plot
            # ====================================================

            ax.plot(
                plot_df[x_col],
                plot_df[metric],
                label=name,
                linewidth=1.8
            )

            plotted = True

        if not plotted:

            plt.close(fig)

            return None

        # ========================================================
        # Label
        # ========================================================

        if x_col == "iteration":

            xlabel = "Iteration"

        else:

            xlabel = "Epoch"

        ax.set_xlabel(
            xlabel,
            fontsize=12
        )

        ax.set_ylabel(
            metric,
            fontsize=12
        )

        ax.set_title(
            f"{metric} vs {xlabel}",
            fontsize=14
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend(
            fontsize=9,
            loc="best"
        )

        fig.tight_layout()

        # ========================================================
        # Save
        # ========================================================

        safe_metric = (
            self.safe_filename(
                metric
            )
        )

        save_path = (
            self.output_dir
            /
            f"{safe_metric}_vs_{x_col}.png"
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )

        return save_path

    # ============================================================
    # Plot selected metrics
    # ============================================================

    def plot_metrics(self):

        if not self.selected_files:

            messagebox.showwarning(
                "Warning",
                "请先选择数据并点击“读取所选数据”"
            )

            return

        selection = (
            self.metric_listbox
            .curselection()
        )

        if not selection:

            messagebox.showwarning(
                "Warning",
                "请选择至少一个指标"
            )

            return

        selected_metrics = [

            self.metric_listbox.get(i)

            for i in selection
        ]

        # ========================================================
        # Output directory
        # ========================================================

        if self.output_dir is None:

            self.output_dir = (
                self.root_dir
                /
                "plots"
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ========================================================
        # Divide sources based on x-axis
        # ========================================================

        iteration_sources = []

        epoch_sources = []

        for name in self.selected_files:

            info = self.data_files[
                name
            ]

            if info["df"] is None:
                continue

            if info["x_col"] == "iteration":

                iteration_sources.append(
                    name
                )

            elif info["x_col"] == "epoch":

                epoch_sources.append(
                    name
                )

        saved_files = []

        print("\n" + "=" * 90)

        print("开始绘图")

        print("=" * 90)

        # ========================================================
        # Metrics
        # ========================================================

        for metric in selected_metrics:

            # ====================================================
            # metrics.csv -> iteration
            # ====================================================

            if iteration_sources:

                save_path = (
                    self.plot_one_group(
                        metric=metric,
                        x_col="iteration",
                        sources=iteration_sources
                    )
                )

                if save_path is not None:

                    saved_files.append(
                        save_path
                    )

                    print(
                        f"[Saved] {save_path}"
                    )

            # ====================================================
            # train.log -> epoch
            # ====================================================

            if epoch_sources:

                save_path = (
                    self.plot_one_group(
                        metric=metric,
                        x_col="epoch",
                        sources=epoch_sources
                    )
                )

                if save_path is not None:

                    saved_files.append(
                        save_path
                    )

                    print(
                        f"[Saved] {save_path}"
                    )

        # ========================================================
        # Finish
        # ========================================================

        if saved_files:

            messagebox.showinfo(
                "完成",
                f"共生成 {len(saved_files)} 张图\n\n"
                f"保存目录：\n"
                f"{self.output_dir}"
            )

        else:

            messagebox.showwarning(
                "Warning",
                "没有生成任何有效图片"
            )


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = MetricsVisualizer(
        root
    )

    root.mainloop()