# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 关键：保证 app/app_gui.py 能找到项目根目录下的 src
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.file_loader import SUPPORTED_SUFFIXES, list_supported_files, load_waveform
from src.infer_service import InferenceError, PWaveInferenceService


class PWavePickerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("P波到时自动识别")
        self.root.geometry("1380x860")
        self.root.minsize(1180, 760)

        self.default_model_path = PROJECT_ROOT / "models" / "best_model.pth"
        self.default_output_path = PROJECT_ROOT / "outputs" / "app_batch_results.csv"
        self.default_showcase_dir = PROJECT_ROOT / "showcase_sample"

        self.model_path_var = tk.StringVar(value=str(self.default_model_path))
        self.threshold_var = tk.StringVar(value="0.50")
        self.device_var = tk.StringVar(value="cpu")
        self.normalize_var = tk.StringVar(value="zscore")
        self.sampling_rate_var = tk.StringVar(value="1.0")
        self.current_file_var = tk.StringVar(value="未选择文件")
        self.result_text_var = tk.StringVar(value="结果：未识别")
        self.status_var = tk.StringVar(value="就绪")
        self.batch_folder_var = tk.StringVar(value="")

        self.service: Optional[PWaveInferenceService] = None
        self.batch_results: List[Dict] = []
        self.result_map: Dict[str, Dict] = {}

        self._build_ui()
        self._build_plot()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        row1 = ttk.Frame(top)
        row1.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(row1, text="模型").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.model_path_var, width=78).pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="选择PTH", command=self.choose_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="加载模型", command=self.reload_model).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(top)
        row2.pack(fill=tk.X)

        ttk.Label(row2, text="阈值").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.threshold_var, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="device").pack(side=tk.LEFT)
        ttk.Combobox(
            row2,
            textvariable=self.device_var,
            values=["cpu", "cuda"],
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="normalize").pack(side=tk.LEFT)
        ttk.Combobox(
            row2,
            textvariable=self.normalize_var,
            values=["zscore", "minmax", "none"],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="采样率").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.sampling_rate_var, width=10).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Button(row2, text="识别单文件", command=self.predict_single_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="批量识别文件夹", command=self.predict_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="导出批量结果", command=self.export_batch_results).pack(side=tk.LEFT, padx=4)

        info = ttk.LabelFrame(self.root, text="识别结果", padding=10)
        info.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(info, text="当前文件：").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.current_file_var).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(info, text="结果：").grid(row=1, column=0, sticky="w", pady=(6, 0))
        result_lbl = ttk.Label(info, textvariable=self.result_text_var, font=("Microsoft YaHei UI", 12, "bold"))
        result_lbl.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(body, text="批量结果", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right.config(width=520)

        plot_box = ttk.LabelFrame(left, text="波形图", padding=8)
        plot_box.pack(fill=tk.BOTH, expand=True)

        self.plot_container = plot_box

        cols = ("file_name", "status", "pick_index", "confidence", "length", "message")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=28)
        self.tree.heading("file_name", text="文件名")
        self.tree.heading("status", text="状态")
        self.tree.heading("pick_index", text="P波样点")
        self.tree.heading("confidence", text="置信度")
        self.tree.heading("length", text="长度")
        self.tree.heading("message", text="备注")

        self.tree.column("file_name", width=180, anchor="w")
        self.tree.column("status", width=70, anchor="center")
        self.tree.column("pick_index", width=80, anchor="center")
        self.tree.column("confidence", width=80, anchor="center")
        self.tree.column("length", width=70, anchor="center")
        self.tree.column("message", width=180, anchor="w")

        yscroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_select)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_plot(self) -> None:
        self.fig = plt.Figure(figsize=(8.8, 5.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("波形预览")
        self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

    def _get_service(self, force_reload: bool = False) -> PWaveInferenceService:
        config = {
            "default_model_path": self.model_path_var.get().strip(),
            "device": self.device_var.get().strip(),
            "threshold": float(self.threshold_var.get().strip()),
            "normalize": self.normalize_var.get().strip(),
            "sampling_rate": float(self.sampling_rate_var.get().strip()),
        }

        if self.service is None or force_reload:
            self.service = PWaveInferenceService(config)
            self.service.load_model()

        return self.service

    def reload_model(self) -> None:
        try:
            self._get_service(force_reload=True)
            self.status_var.set(f"模型已加载：{self.model_path_var.get()}")
            messagebox.showinfo("成功", "模型加载成功。")
        except Exception as exc:
            self.status_var.set(f"模型加载失败：{exc}")
            messagebox.showerror("错误", f"模型加载失败：\n{exc}")

    def _draw_waveform(
        self,
        waveform: np.ndarray,
        pick_index: Optional[int] = None,
        title: str = "波形预览",
    ) -> None:
        self.ax.clear()
        self.ax.plot(np.arange(len(waveform)), waveform, linewidth=1.0)
        if pick_index is not None and 0 <= pick_index < len(waveform):
            self.ax.axvline(pick_index, linestyle="--", linewidth=1.5)
            self.ax.text(
                pick_index,
                float(np.max(waveform)) if len(waveform) else 0.0,
                f"  P={pick_index}",
                fontsize=10,
                verticalalignment="top",
            )
        self.ax.set_title(title)
        self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def predict_single_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择波形文件",
            filetypes=[
                ("Supported", "*.npy *.txt *.csv"),
                ("NumPy", "*.npy"),
                ("Text", "*.txt"),
                ("CSV", "*.csv"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.status_var.set("正在识别单文件...")
            self.root.update_idletasks()

            service = self._get_service(force_reload=False)
            result = service.predict_file(file_path)

            self.current_file_var.set(file_path)
            if result["detected"]:
                txt = (
                    f"识别成功｜P波样点={result['pick_index']}｜"
                    f"置信度={result['confidence']:.4f}"
                )
            else:
                txt = f"未过阈值｜峰值置信度={result['confidence']:.4f}"
            self.result_text_var.set(txt)

            self._draw_waveform(
                result["waveform"],
                pick_index=result["pick_index"],
                title=result["file_name"],
            )
            self.status_var.set("单文件识别完成。")

        except Exception as exc:
            self.current_file_var.set(file_path)
            self.result_text_var.set("识别失败")
            self.status_var.set(f"单文件识别失败：{exc}")
            messagebox.showerror("错误", f"识别失败：\n{exc}")

    def _clear_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    def predict_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择要批量识别的文件夹")
        if not folder:
            return

        try:
            files = list_supported_files(folder)
            if not files:
                messagebox.showwarning(
                    "提示",
                    f"该文件夹下没有可识别文件。\n当前支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}",
                )
                return

            self.status_var.set(f"正在批量识别，共 {len(files)} 个文件...")
            self.root.update_idletasks()

            service = self._get_service(force_reload=False)
            self.batch_results = service.predict_folder(folder)
            self.result_map = {str(r["file_path"]): r for r in self.batch_results}

            self._clear_tree()
            for r in self.batch_results:
                conf = "" if r.get("confidence") is None else f"{r['confidence']:.4f}"
                msg = r.get("message", "")
                if len(msg) > 80:
                    msg = msg[:80] + "..."
                self.tree.insert(
                    "",
                    tk.END,
                    iid=str(r["file_path"]),
                    values=(
                        r.get("file_name", ""),
                        r.get("status", ""),
                        "" if r.get("pick_index") is None else r.get("pick_index"),
                        conf,
                        "" if r.get("length") is None else r.get("length"),
                        msg,
                    ),
                )

            ok_num = sum(1 for r in self.batch_results if r.get("ok", False))
            detect_num = sum(1 for r in self.batch_results if r.get("detected", False))
            fail_num = sum(1 for r in self.batch_results if r.get("status") == "失败")

            self.batch_folder_var.set(folder)
            self.status_var.set(
                f"批量识别完成：总数={len(self.batch_results)}，成功推理={ok_num}，识别到P波={detect_num}，失败={fail_num}"
            )
            messagebox.showinfo(
                "完成",
                f"批量识别完成。\n总数：{len(self.batch_results)}\n成功推理：{ok_num}\n识别到P波：{detect_num}\n失败：{fail_num}"
            )

        except Exception as exc:
            self.status_var.set(f"批量识别失败：{exc}")
            messagebox.showerror("错误", f"批量识别失败：\n{exc}")

    def on_tree_select(self, event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return

        file_path = sel[0]
        r = self.result_map.get(file_path)
        if not r:
            return

        self.current_file_var.set(file_path)

        if r.get("status") == "失败":
            self.result_text_var.set(f"识别失败｜{r.get('message', '')}")
            return

        try:
            if "waveform" in r and r["waveform"] is not None:
                waveform = r["waveform"]
            else:
                item = load_waveform(file_path)
                waveform = item["waveform"]

            if r.get("detected"):
                self.result_text_var.set(
                    f"识别成功｜P波样点={r.get('pick_index')}｜置信度={r.get('confidence', 0):.4f}"
                )
            else:
                self.result_text_var.set(
                    f"未过阈值｜峰值置信度={r.get('confidence', 0):.4f}"
                )

            self._draw_waveform(
                waveform,
                pick_index=r.get("pick_index"),
                title=r.get("file_name", Path(file_path).name),
            )
        except Exception as exc:
            self.result_text_var.set(f"绘图失败｜{exc}")

    def export_batch_results(self) -> None:
        if not self.batch_results:
            messagebox.showwarning("提示", "当前没有批量结果可导出。")
            return

        out_path = filedialog.asksaveasfilename(
            title="保存批量结果",
            defaultextension=".csv",
            initialfile="app_batch_results.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not out_path:
            return

        rows = []
        for r in self.batch_results:
            rows.append(
                {
                    "file_path": r.get("file_path"),
                    "file_name": r.get("file_name"),
                    "suffix": r.get("suffix"),
                    "status": r.get("status"),
                    "detected": r.get("detected"),
                    "pick_index": r.get("pick_index"),
                    "pick_time": r.get("pick_time"),
                    "confidence": r.get("confidence"),
                    "threshold": r.get("threshold"),
                    "length": r.get("length"),
                    "message": r.get("message"),
                }
            )

        try:
            df = pd.DataFrame(rows)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            self.status_var.set(f"批量结果已导出：{out_path}")
            messagebox.showinfo("成功", f"批量结果已导出：\n{out_path}")
        except Exception as exc:
            messagebox.showerror("错误", f"导出失败：\n{exc}")


def main() -> None:
    root = tk.Tk()
    app = PWavePickerApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise