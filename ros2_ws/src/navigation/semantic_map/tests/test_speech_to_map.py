"""
Interactive test: speech mock → semantic map → live 3D visualisation.

Loads (or creates) a persistent map at tests/test_semantic_map.pkl.
Each detection from MockSpeech is added to the map and the 3D view updates
in real time.

Run from the TRAVIS root:
    python3 tests/test_speech_to_map.py
"""

import logging
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "travis_brain"))
sys.path.insert(0, str(_ROOT / "speech"))

from semantic_map.semantic_map import SemanticMap, SemanticNode
from speech_mock.mock_speech import MockSpeech, SpeechDetection
from visualisation.view_semantic_map import visualise_semantic_map

# ------------------------------------------------------------------
# Logging — DEBUG so all semantic map events are visible
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s — %(message)s",
)

MAP_PATH = Path(__file__).parent / "test_semantic_map.pkl"


# ------------------------------------------------------------------
# Application
# ------------------------------------------------------------------

class SpeechToMapApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Speech → Semantic Map")

        self.speech = MockSpeech()
        self.map = self._load_or_create_map()

        self._build_ui()
        self._refresh_plot()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Map persistence
    # ------------------------------------------------------------------

    def _load_or_create_map(self) -> SemanticMap:
        if MAP_PATH.exists():
            m = SemanticMap.load_semantic_map(MAP_PATH)
            logging.getLogger(__name__).info(
                f"Loaded existing map from {MAP_PATH}  ({len(m)} nodes)"
            )
            return m
        logging.getLogger(__name__).info("No existing map found — starting fresh.")
        return SemanticMap()

    def _save_map(self) -> None:
        self.map.save_semantic_map(MAP_PATH)
        self._log(f"Map saved → {MAP_PATH}  ({len(self.map)} nodes)")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Left control panel ──────────────────────────────────────
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.grid(row=0, column=0, sticky="nsew")

        ttk.Label(ctrl, text="Mode:").grid(row=0, column=0, sticky="w")
        self._mode_var = tk.StringVar(value=self.speech.mode)
        mode_menu = ttk.OptionMenu(
            ctrl, self._mode_var,
            self.speech.mode, "hardcoded", "random", "user_input",
            command=self._on_mode_change,
        )
        mode_menu.grid(row=0, column=1, sticky="ew", pady=(0, 10))

        # User-input fields (shown only in user_input mode)
        self._user_frame = ttk.LabelFrame(ctrl, text="Detection input", padding=6)
        self._user_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        fields = [("Class", "mug"), ("Confidence", "95.0"),
                  ("X (m)", "0.0"), ("Y (m)", "0.0"), ("Z (m)", "0.0")]
        self._entries: dict[str, tk.StringVar] = {}
        for i, (label, default) in enumerate(fields):
            ttk.Label(self._user_frame, text=label).grid(row=i, column=0, sticky="w")
            var = tk.StringVar(value=default)
            ttk.Entry(self._user_frame, textvariable=var, width=12).grid(
                row=i, column=1, sticky="ew", padx=(4, 0)
            )
            self._entries[label] = var

        # Buttons
        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Button(btn_frame, text="Add Detection",
                   command=self._add_detection).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Save Map",
                   command=self._save_map).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Reset Map",
                   command=self._reset_map).pack(fill="x", pady=2)

        # Log
        ttk.Label(ctrl, text="Log:").grid(row=3, column=0, sticky="w")
        self._log_text = tk.Text(ctrl, width=32, height=18, state="disabled",
                                 font=("Courier", 9))
        self._log_text.grid(row=4, column=0, columnspan=2, sticky="nsew")
        scroll = ttk.Scrollbar(ctrl, command=self._log_text.yview)
        scroll.grid(row=4, column=2, sticky="ns")
        self._log_text.configure(yscrollcommand=scroll.set)

        # ── Right matplotlib canvas ──────────────────────────────────
        self._fig = plt.figure(figsize=(8, 6))
        self._ax = self._fig.add_subplot(111, projection="3d")

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._update_field_visibility()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_mode_change(self, value: str) -> None:
        self.speech.mode = value
        self._update_field_visibility()

    def _update_field_visibility(self) -> None:
        state = "normal" if self.speech.mode == "user_input" else "disabled"
        for child in self._user_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _add_detection(self) -> None:
        detection = self._get_detection()
        if detection is None:
            return

        self._log(
            f"► {detection.object_class}  conf={detection.confidence:.1f}  "
            f"({detection.x:.2f}, {detection.y:.2f}, {detection.z:.2f})"
        )

        node = SemanticNode(
            label=detection.object_class,
            confidence=detection.confidence,
            x=detection.x,
            y=detection.y,
            z=detection.z,
            timestamp=time.time(),
        )
        result = self.map.add_node(node)

        if result is None:
            self._log(f"  ✗ rejected (confidence below threshold)")
        elif node.id != result:
            self._log(f"  ↻ merged into existing id={result}")
        else:
            self._log(f"  ✓ added as id={result}  (map: {len(self.map)} nodes)")

        self._refresh_plot()

    def _get_detection(self) -> SpeechDetection | None:
        if self.speech.mode == "user_input":
            return self._detection_from_fields()
        return self.speech.get_detection()

    def _detection_from_fields(self) -> SpeechDetection | None:
        try:
            return SpeechDetection(
                object_class=self._entries["Class"].get().strip(),
                confidence=float(self._entries["Confidence"].get()),
                x=float(self._entries["X (m)"].get()),
                y=float(self._entries["Y (m)"].get()),
                z=float(self._entries["Z (m)"].get()),
            )
        except ValueError as e:
            self._log(f"  ✗ invalid input: {e}")
            return None

    def _reset_map(self) -> None:
        self.map = SemanticMap()
        if MAP_PATH.exists():
            MAP_PATH.unlink()
        self._log("Map reset.")
        self._refresh_plot()

    def _on_close(self) -> None:
        self._save_map()
        plt.close("all")
        self.root.destroy()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        self._ax.cla()
        visualise_semantic_map(
            self.map,
            fig=self._fig,
            ax=self._ax,
            title="Semantic Map (live)",
        )
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Log widget
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", message + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechToMapApp(root)
    root.mainloop()
