#!/usr/bin/env python3
"""
GPR Live B-Scan  —  NanoVNA V2 (S-A-A-2)
==========================================
Verbindet sich mit dem NanoVNA V2, liest S21-Spuren in Echtzeit,
berechnet TDR via IFFT und zeigt den B-Scan live an.

Voraussetzungen:
    pip install pyserial numpy scipy matplotlib

Verwendung:
    python gpr_live.py
    python gpr_live.py --port /dev/ttyACM0 --er 9 --spacing 0.05
    python gpr_live.py --start 100e6 --stop 3e9 --points 201 --scans 20

Argumente (alle optional):
    --port      Serieller Port (Standard: automatische Erkennung)
    --start     Startfrequenz Hz (Standard: 100 MHz)
    --stop      Stoppfrequenz Hz (Standard: 3 GHz)
    --points    Frequenzpunkte 1–1024 (Standard: 201)
    --avg       Mittelungen pro Punkt (Standard: 1)
    --scans     Anzahl Spuren im B-Scan Fenster (Standard: 20)
    --er        Relative Permittivität des Bodens (Standard: 9)
    --spacing   Spurabstand in Metern (Standard: 0.05)
    --tmax      Maximale Tiefe in Metern (Standard: auto)
    --outdir    Ordner für .s2p Dateien (Standard: ./captures)
    --prefix    Dateiname-Präfix (Standard: trace)
    --window    Fensterfunktion: hann|hamming|blackman|none (Standard: hann)
    --gain      TGC in dB/ns (Standard: 2.0)
    --cmap      Colormap (Standard: bwr)
    --no-dewow  Dewow-Filter deaktivieren
    --bgrem     Hintergrundentfernung (Mittelwert abziehen)

Steuerung:
    LEERTASTE / ENTER  →  Neue Messserie starten (Reset)
    s                  →  Aktuelle Serie als .s2p Dateien speichern
    q                  →  Beenden
"""

import argparse
import os
import sys
import time
import datetime
import threading
import struct
from struct import pack, unpack_from
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # interaktives Backend; Fallback weiter unten
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import windows as sig_windows
import serial
import serial.tools.list_ports

# ═══════════════════════════════════════════════════════════════════════════════
# NanoVNA V2 Binärprotokoll
# ═══════════════════════════════════════════════════════════════════════════════

CMD_NOP       = 0x00
CMD_READ      = 0x10
CMD_READ2     = 0x11
CMD_READ4     = 0x12
CMD_READFIFO  = 0x18
CMD_WRITE     = 0x20
CMD_WRITE2    = 0x21
CMD_WRITE4    = 0x22
CMD_WRITE8    = 0x23
CMD_WRITEFIFO = 0x28

ADDR_SWEEP_START      = 0x00
ADDR_SWEEP_STEP       = 0x10
ADDR_SWEEP_POINTS     = 0x20
ADDR_SWEEP_VALS_PER_F = 0x22
ADDR_VALUES_FIFO      = 0x30
ADDR_DEVICE_VARIANT   = 0xF0
ADDR_PROTOCOL_VERSION = 0xF1
ADDR_HW_REVISION      = 0xF2
ADDR_FW_MAJOR         = 0xF3
ADDR_FW_MINOR         = 0xF4

WRITE_SLEEP = 0.05

NANOVNA2_USB_IDS = [
    (0x04b4, 0x0008),
    (0x16c0, 0x0483),
    (0x0483, 0x5740),
]


class NanoVNA2:
    def __init__(self, port, baud=115200, timeout=10):
        self.port          = port
        self.baud          = baud
        self.timeout       = timeout
        self.ser           = None
        self.sweep_start   = 100_000_000
        self.sweep_step    = 1_000_000
        self.sweep_points  = 201
        self.sweep_avg     = 1

    def connect(self):
        self.ser = serial.Serial(
            self.port, baudrate=self.baud, timeout=self.timeout,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        time.sleep(0.3)
        self.ser.write(pack("<Q", 0))   # Protokoll-Reset
        time.sleep(WRITE_SLEEP)
        return self._read_version()

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _w1(self, addr, v):
        self.ser.write(pack("<BBB", CMD_WRITE, addr, v & 0xFF))
        time.sleep(WRITE_SLEEP)

    def _w2(self, addr, v):
        self.ser.write(pack("<BBH", CMD_WRITE2, addr, v & 0xFFFF))
        time.sleep(WRITE_SLEEP)

    def _w8(self, addr, v):
        self.ser.write(pack("<BBQ", CMD_WRITE8, addr, int(v)))
        time.sleep(WRITE_SLEEP)

    def _r1(self, addr):
        self.ser.write(pack("<BB", CMD_READ, addr))
        time.sleep(WRITE_SLEEP)
        d = self.ser.read(1)
        return d[0] if d else 0

    def _read_version(self):
        variant  = self._r1(ADDR_DEVICE_VARIANT)
        proto    = self._r1(ADDR_PROTOCOL_VERSION)
        hw_rev   = self._r1(ADDR_HW_REVISION)
        fw_major = self._r1(ADDR_FW_MAJOR)
        fw_minor = self._r1(ADDR_FW_MINOR)
        if fw_major == 0xFF:
            raise IOError("Gerät im DFU-Modus!")
        return f"Variante {variant:#04x}  FW {fw_major}.{fw_minor}  HW-Rev {hw_rev}  Protokoll v{proto}"

    def set_sweep(self, start_hz, stop_hz, points, avg=1):
        self.sweep_start  = int(start_hz)
        self.sweep_points = int(points)
        self.sweep_avg    = int(avg)
        self.sweep_step   = int((stop_hz - start_hz) / max(1, points - 1))
        self._w8(ADDR_SWEEP_START,      self.sweep_start)
        self._w8(ADDR_SWEEP_STEP,       self.sweep_step)
        self._w2(ADDR_SWEEP_POINTS,     self.sweep_points)
        self._w1(ADDR_SWEEP_VALS_PER_F, self.sweep_avg)

    def get_frequencies(self):
        return np.array([self.sweep_start + i * self.sweep_step
                         for i in range(self.sweep_points)], dtype=float)

    def read_scan(self):
        """Liest einen vollständigen Sweep. Gibt (s11, s21) zurück."""
        n   = self.sweep_points
        avg = self.sweep_avg

        acc   = np.zeros((n, 4), dtype=complex)
        count = np.zeros(n, dtype=int)

        # FIFO leeren
        self.ser.write(pack("<BBB", CMD_WRITE, ADDR_VALUES_FIFO, 0))
        time.sleep(WRITE_SLEEP)
        old_to = self.ser.timeout
        self.ser.timeout = 0.05
        self.ser.read(4096)
        self.ser.timeout = max(4.0, avg * 2.0)

        left = n * avg
        while left > 0:
            nr = min(255, left)
            self.ser.write(pack("<BBB", CMD_READFIFO, ADDR_VALUES_FIFO, nr))
            time.sleep(WRITE_SLEEP)
            raw = self.ser.read(nr * 32)
            if len(raw) < nr * 32:
                raise RuntimeError(f"Nur {len(raw)}/{nr*32} Bytes empfangen")
            for i in range(nr):
                fwr, fwi, r0r, r0i, r1r, r1i, fi = unpack_from("<iiiiiih6x", raw, i * 32)
                if 0 <= fi < n:
                    fwd  = complex(fwr, fwi)
                    rev0 = complex(r0r, r0i)
                    rev1 = complex(r1r, r1i)
                    acc[fi] += np.array([fwd, rev0, 0.0, rev1])
                    count[fi] += 1
            left -= nr

        self.ser.timeout = old_to

        s11 = np.zeros(n, dtype=complex)
        s21 = np.zeros(n, dtype=complex)
        for i in range(n):
            if count[i] > 0:
                fwd = acc[i, 0] / count[i]
                if abs(fwd) > 0:
                    s11[i] = (acc[i, 1] / count[i]) / fwd
                    s21[i] = (acc[i, 3] / count[i]) / fwd
        return s11, s21


# ═══════════════════════════════════════════════════════════════════════════════
# Signalverarbeitung
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tdr(freqs, s21, window_type="hann"):
    N  = len(freqs)
    df = freqs[1] - freqs[0]
    if window_type == "hann":
        win = sig_windows.hann(N)
    elif window_type == "hamming":
        win = sig_windows.hamming(N)
    elif window_type == "blackman":
        win = sig_windows.blackman(N)
    else:
        win = np.ones(N)
    N_fft = max(4 * N, 2 ** int(np.ceil(np.log2(4 * N))))
    tdr   = np.real(np.fft.ifft(s21 * win, n=N_fft))
    dt    = 1.0 / (N_fft * df)
    t_ns  = np.arange(N_fft) * dt * 1e9
    return t_ns, tdr

def dewow(trace, window=None):
    if window is None:
        window = max(3, len(trace) // 20)
    trend = np.convolve(trace, np.ones(window) / window, mode="same")
    return trace - trend

def apply_tgc(trace, t_ns, gain_db_per_ns):
    return trace * 10 ** (gain_db_per_ns * t_ns / 20.0)

def time_to_depth(t_ns, er):
    c = 0.299792458  # m/ns
    return c / np.sqrt(er) * t_ns / 2.0

def process_trace(freqs, s21, window_type, do_dewow, gain_db_per_ns):
    t_ns, tdr = compute_tdr(freqs, s21, window_type)
    if do_dewow:
        tdr = dewow(tdr)
    tdr = apply_tgc(tdr, t_ns, gain_db_per_ns)
    return t_ns, tdr


# ═══════════════════════════════════════════════════════════════════════════════
# Touchstone Export
# ═══════════════════════════════════════════════════════════════════════════════

def save_s2p(filepath, freqs, s11, s21):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"! GPR Live Capture  —  {now}",
        "# Hz S RI R 50",
    ]
    for i, f in enumerate(freqs):
        s11r, s11i = np.real(s11[i]), np.imag(s11[i])
        s21r, s21i = np.real(s21[i]), np.imag(s21[i])
        lines.append(
            f"{f:.6e}  {s11r:.10f} {s11i:.10f}  "
            f"{s21r:.10f} {s21i:.10f}  "
            f"{s21r:.10f} {s21i:.10f}  "
            f"{s11r:.10f} {s11i:.10f}"
        )
    with open(filepath, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Live-Anzeige
# ═══════════════════════════════════════════════════════════════════════════════

def find_nanovna2_port():
    for p in serial.tools.list_ports.comports():
        if p.vid and p.pid:
            for vid, pid in NANOVNA2_USB_IDS:
                if p.vid == vid and p.pid == pid:
                    return p.device
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower()
               for k in ["cypress", "usb serial", "acm", "usbmodem"]):
            return p.device
    return None


def run(args):
    # ── Port ──────────────────────────────────────────────────────────────────
    port = args.port or find_nanovna2_port()
    if port is None:
        print("[Fehler] Kein NanoVNA V2 gefunden.")
        print("Verfügbare Ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  [{getattr(p,'vid',None):#06x}:{getattr(p,'pid',None):#06x}]  {p.description}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # ── Gerät verbinden ───────────────────────────────────────────────────────
    print(f"Verbinde: {port} …")
    vna = NanoVNA2(port)
    try:
        info = vna.connect()
        print(f"Verbunden: {info}")
    except Exception as e:
        print(f"[Fehler] {e}")
        sys.exit(1)

    vna.set_sweep(args.start, args.stop, args.points, args.avg)
    freqs = vna.get_frequencies()

    # TDR-Länge vorab bestimmen
    N_fft = max(4 * args.points, 2 ** int(np.ceil(np.log2(4 * args.points))))
    df    = (args.stop - args.start) / max(1, args.points - 1)
    dt_ns = 1e9 / (N_fft * df)
    t_ns_full = np.arange(N_fft) * dt_ns
    depth_full = time_to_depth(t_ns_full, args.er)

    if args.tmax is not None:
        idx_max = np.searchsorted(depth_full, args.tmax)
    else:
        idx_max = N_fft // 2

    t_ns  = t_ns_full[:idx_max]
    depth = depth_full[:idx_max]

    c   = 0.299792458
    v   = c / np.sqrt(args.er)
    dx  = args.spacing
    N   = args.scans

    # ── Shared State ──────────────────────────────────────────────────────────
    bscan      = np.zeros((N, idx_max))   # Ringpuffer
    bscan_s11  = [None] * N              # für Export
    bscan_s21  = [None] * N
    scan_count = [0]                     # laufende Gesamtanzahl
    slot       = [0]                     # aktueller Ringpuffer-Slot
    reset_flag = [False]
    save_flag  = [False]
    quit_flag  = [False]
    lock       = threading.Lock()
    status_msg = ["Warte auf erste Messung …"]

    # ── Matplotlib-Fenster ────────────────────────────────────────────────────
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

    plt.rcParams.update({
        "figure.facecolor":  "#0a0e14",
        "axes.facecolor":    "#0a0e14",
        "text.color":        "#c8d0dc",
        "axes.labelcolor":   "#c8d0dc",
        "axes.edgecolor":    "#2a3040",
        "xtick.color":       "#7a8899",
        "ytick.color":       "#7a8899",
        "grid.color":        "#1a2030",
        "grid.linewidth":    0.6,
    })

    fig = plt.figure(figsize=(13, 7), facecolor="#0a0e14")
    fig.canvas.manager.set_window_title("GPR Live B-Scan  —  NanoVNA V2")

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.07, right=0.97,
        top=0.91, bottom=0.10,
        hspace=0.45, wspace=0.35,
        height_ratios=[3, 1],
    )

    ax_bscan = fig.add_subplot(gs[0, :])   # B-Scan oben, volle Breite
    ax_s21   = fig.add_subplot(gs[1, 0])   # S21 in dB
    ax_tdr   = fig.add_subplot(gs[1, 1])   # aktuelle TDR-Spur

    # B-Scan Bild (Platzhalter)
    bscan_img = ax_bscan.imshow(
        bscan.T,
        aspect="auto",
        extent=[0, N * dx, depth[-1], depth[0]],
        cmap=args.cmap,
        vmin=-1, vmax=1,
        interpolation="bilinear",
        origin="upper",
    )
    cbar = plt.colorbar(bscan_img, ax=ax_bscan, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors="#7a8899")
    cbar.set_label("Amplitude", fontsize=8, color="#7a8899")

    ax_bscan.set_xlabel("Position [m]", fontsize=9)
    ax_bscan.set_ylabel("Tiefe [m]",    fontsize=9)
    ax_bscan.xaxis.set_major_locator(ticker.MultipleLocator(dx * 2))
    ax_bscan.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_bscan.grid(True, which="major")

    # S21-Plot
    s21_line, = ax_s21.plot(freqs / 1e6, np.zeros(len(freqs)), color="#00c8ff", lw=1)
    ax_s21.set_xlabel("Frequenz [MHz]", fontsize=8)
    ax_s21.set_ylabel("S21 [dB]",       fontsize=8)
    ax_s21.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
    ax_s21.set_ylim(-80, 5)
    ax_s21.grid(True)

    # TDR-Plot
    tdr_line, = ax_tdr.plot(depth, np.zeros(idx_max), color="#ff6b35", lw=1)
    ax_tdr.set_xlabel("Tiefe [m]",      fontsize=8)
    ax_tdr.set_ylabel("TDR-Amplitude",  fontsize=8)
    ax_tdr.set_xlim(depth[0], depth[-1])
    ax_tdr.grid(True)

    title = fig.suptitle(
        f"GPR Live B-Scan  |  εr={args.er}  v={v:.3f} m/ns  "
        f"Δx={args.spacing*100:.0f} cm  |  {freqs[0]/1e6:.0f}–{freqs[-1]/1e6:.0f} MHz",
        fontsize=11, color="#e0e8f0", y=0.97,
    )

    status_text = fig.text(
        0.5, 0.005,
        "LEERTASTE/ENTER = Reset  |  S = Speichern  |  Q = Beenden",
        ha="center", fontsize=8, color="#556070",
        transform=fig.transFigure,
    )
    scan_text = ax_bscan.set_title("Scan 0 / " + str(N), fontsize=9, color="#7a8899")

    # vertikale Linie: aktuelle Position
    cur_line = ax_bscan.axvline(x=0, color="#ffcc00", lw=1.2, alpha=0.7)

    # ── Messthread ────────────────────────────────────────────────────────────
    def measure_loop():
        global_idx = 0
        while not quit_flag[0]:
            # Reset-Anforderung
            with lock:
                if reset_flag[0]:
                    bscan[:] = 0
                    bscan_s11[:] = [None] * N
                    bscan_s21[:] = [None] * N
                    scan_count[0] = 0
                    slot[0] = 0
                    global_idx = 0
                    reset_flag[0] = False
                    status_msg[0] = "Reset — warte auf erste Messung …"

            try:
                s11, s21 = vna.read_scan()
                t_ns_cur, tdr_cur = process_trace(
                    freqs, s21, args.window, not args.no_dewow, args.gain
                )
                tdr_crop = tdr_cur[:idx_max]

                # Hintergrundentfernung
                with lock:
                    if args.bgrem and scan_count[0] > 0:
                        mean_col = np.mean(bscan, axis=0)
                        tdr_crop = tdr_crop - mean_col

                    sl = slot[0]
                    bscan[sl, :]   = tdr_crop
                    bscan_s11[sl]  = s11.copy()
                    bscan_s21[sl]  = s21.copy()
                    slot[0]        = (sl + 1) % N
                    scan_count[0] += 1
                    global_idx    += 1
                    status_msg[0]  = f"Scan {scan_count[0]}"

            except Exception as e:
                status_msg[0] = f"Fehler: {e}"
                time.sleep(0.3)

    mthread = threading.Thread(target=measure_loop, daemon=True)
    mthread.start()

    # ── Tastatureingabe im Hauptthread (matplotlib key_press_event) ───────────
    def on_key(event):
        k = (event.key or "").lower()
        with lock:
            if k in ("enter", " "):
                reset_flag[0] = True
            elif k == "s":
                save_flag[0] = True
            elif k in ("q", "escape"):
                quit_flag[0] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ── Animations-Update ─────────────────────────────────────────────────────
    def update(_frame):
        with lock:
            bs      = bscan.copy()
            sc      = scan_count[0]
            sl      = slot[0]
            msg     = status_msg[0]
            do_save = save_flag[0]
            if do_save:
                save_flag[0] = False
            s21_cur = bscan_s21[(sl - 1) % N]
            s11_cur = bscan_s11[(sl - 1) % N]

        # B-Scan: Ringpuffer so rollen, dass älteste Spur links
        # Slot zeigt auf nächsten freien Platz → sl ist älteste Spur
        rolled = np.roll(bs, -sl, axis=0)

        # Normierung
        vmax = np.percentile(np.abs(rolled), 98) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        bscan_img.set_data(rolled.T)
        bscan_img.set_norm(norm)
        bscan_img.set_extent([0, N * dx, depth[-1], depth[0]])

        # aktuelle Position
        cur_x = min(sc, N) * dx
        cur_line.set_xdata([cur_x, cur_x])

        scan_text.set_text(f"Scan {sc}  (Fenster: {min(sc, N)}/{N})")

        # S21-Kurve
        if s21_cur is not None:
            amp = np.abs(s21_cur)
            amp = np.where(amp < 1e-12, 1e-12, amp)
            db  = 20 * np.log10(amp)
            s21_line.set_ydata(db)
            ax_s21.relim()
            ax_s21.autoscale_view(scalex=False)

        # TDR-Kurve
        if s21_cur is not None:
            _, tdr_cur = process_trace(
                freqs, s21_cur, args.window, not args.no_dewow, args.gain
            )
            tdr_line.set_ydata(tdr_cur[:idx_max])
            ax_tdr.relim()
            ax_tdr.autoscale_view(scalex=False)

        status_text.set_text(
            f"{msg}   |   LEERTASTE/ENTER=Reset  S=Speichern  Q=Beenden"
        )

        # Speichern
        if do_save:
            _save_all_scans(args, freqs, bscan_s11, bscan_s21, sc)

        if quit_flag[0]:
            plt.close("all")

        fig.canvas.draw_idle()
        return bscan_img, s21_line, tdr_line, cur_line, status_text

    ani = matplotlib.animation.FuncAnimation(
        fig, update, interval=200, blit=False, cache_frame_data=False
    )

    try:
        plt.show()
    finally:
        quit_flag[0] = True
        vna.disconnect()
        print("\nVerbindung getrennt.")


def _save_all_scans(args, freqs, bscan_s11, bscan_s21, total):
    """Speichert alle vorhandenen Spuren als .s2p Dateien."""
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = 0
    for i, (s11, s21) in enumerate(zip(bscan_s11, bscan_s21)):
        if s11 is None or s21 is None:
            continue
        fname = os.path.join(args.outdir, f"{args.prefix}_{i+1:04d}_{ts}.s2p")
        save_s2p(fname, freqs, s11, s21)
        saved += 1
    print(f"\n  ✓ {saved} Spur(en) gespeichert in '{args.outdir}'")


# ═══════════════════════════════════════════════════════════════════════════════
# Argparse & Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import matplotlib.animation  # noqa: F401 — sicherstellen dass importiert

    p = argparse.ArgumentParser(
        description="GPR Live B-Scan — NanoVNA V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--port",    default=None)
    p.add_argument("--start",   type=float, default=100e6)
    p.add_argument("--stop",    type=float, default=3000e6)
    p.add_argument("--points",  type=int,   default=201)
    p.add_argument("--avg",     type=int,   default=1)
    p.add_argument("--scans",   type=int,   default=20)
    p.add_argument("--er",      type=float, default=9.0)
    p.add_argument("--spacing", type=float, default=0.05)
    p.add_argument("--tmax",    type=float, default=None)
    p.add_argument("--outdir",  default="./captures")
    p.add_argument("--prefix",  default="trace")
    p.add_argument("--window",  default="hann",
                   choices=["hann", "hamming", "blackman", "none"])
    p.add_argument("--gain",    type=float, default=2.0)
    p.add_argument("--cmap",    default="bwr")
    p.add_argument("--no-dewow", action="store_true")
    p.add_argument("--bgrem",   action="store_true")
    args = p.parse_args()

    print("═" * 60)
    print("  GPR Live B-Scan  —  NanoVNA V2 (S-A-A-2)")
    print("═" * 60)
    print(f"  Frequenz : {args.start/1e6:.1f} – {args.stop/1e6:.1f} MHz")
    print(f"  Punkte   : {args.points}")
    print(f"  Scans    : {args.scans}")
    print(f"  εr       : {args.er}  →  v = {0.299792458/np.sqrt(args.er):.4f} m/ns")
    print(f"  Δx       : {args.spacing*100:.0f} cm")
    print(f"  Ausgabe  : {args.outdir}")
    print()

    run(args)


if __name__ == "__main__":
    import matplotlib.animation
    main()
