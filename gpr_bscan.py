#!/usr/bin/env python3
"""
GPR B-Scan Processor
====================
Liest S21-Touchstone (.s2p) Dateien eines Bodenradars ein,
berechnet per IFFT die TDR-Antwort und stellt den B-Scan
(Reflexionstiefe über Messposition) als Bild dar.

Verwendung:
    python gpr_bscan.py --folder ./messungen --er 9 --spacing 0.05

Argumente:
    --folder    Ordner mit den .s2p Dateien (Standard: aktuelles Verzeichnis)
    --er        Relative Permittivität des Bodens (Standard: 9)
    --spacing   Abstand zwischen Messpunkten in Metern (Standard: 0.05 m)
    --output    Ausgabedatei (Standard: bscan.png)
    --tmax      Maximale Tiefe in Metern (Standard: automatisch)
    --window    Fensterfunktion: hann, hamming, blackman, none (Standard: hann)
    --gain      Zeitabhängige Verstärkung (TGC) in dB/ns (Standard: 2.0)
    --cmap      Colormap: bwr, seismic, gray, viridis (Standard: bwr)
    --dpi       Auflösung des Ausgabebildes (Standard: 200)
    --no-dewow  Dewow-Filter deaktivieren
    --bgrem     Hintergrundentfernung (mittlere Spur abziehen)
"""

import argparse
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import windows as sig_windows

# ── Touchstone Parser ──────────────────────────────────────────────────────────

def parse_s2p(filepath):
    """
    Liest eine .s2p Datei und gibt (frequencies_Hz, S21_komplex) zurück.
    Unterstützt MA (Magnitude/Angle), DB (dB/Angle) und RI (Real/Imag) Format.
    """
    frequencies = []
    s11_raw = []
    s21_raw = []
    s12_raw = []
    s22_raw = []
    format_type = "MA"
    freq_unit = 1.0  # Hz

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parts = line.upper().split()
                # Frequenzeinheit
                for unit, factor in [("GHZ", 1e9), ("MHZ", 1e6), ("KHZ", 1e3), ("HZ", 1.0)]:
                    if unit in parts:
                        freq_unit = factor
                        break
                # Format
                for fmt in ["MA", "DB", "RI"]:
                    if fmt in parts:
                        format_type = fmt
                        break
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                freq = float(parts[0]) * freq_unit
                frequencies.append(freq)
                s11_raw.append((float(parts[1]), float(parts[2])))
                s21_raw.append((float(parts[3]), float(parts[4])))
                s12_raw.append((float(parts[5]), float(parts[6])))
                s22_raw.append((float(parts[7]), float(parts[8])))
            except ValueError:
                continue

    freqs = np.array(frequencies)

    def to_complex(raw, fmt):
        a = np.array([r[0] for r in raw])
        b = np.array([r[1] for r in raw])
        if fmt == "RI":
            return a + 1j * b
        elif fmt == "DB":
            mag = 10 ** (a / 20.0)
            return mag * np.exp(1j * np.deg2rad(b))
        else:  # MA
            return a * np.exp(1j * np.deg2rad(b))

    s21 = to_complex(s21_raw, format_type)
    return freqs, s21


# ── TDR Berechnung ─────────────────────────────────────────────────────────────

def compute_tdr(freqs, s21, window_type="hann"):
    """
    Berechnet die TDR-Antwort via IFFT mit optionaler Fensterfunktion.
    Gibt (time_ns, tdr_real) zurück.
    """
    N = len(freqs)
    df = freqs[1] - freqs[0]  # Frequenzauflösung in Hz

    # Fensterfunktion
    if window_type == "hann":
        win = sig_windows.hann(N)
    elif window_type == "hamming":
        win = sig_windows.hamming(N)
    elif window_type == "blackman":
        win = sig_windows.blackman(N)
    else:
        win = np.ones(N)

    s21_win = s21 * win

    # Zero-padding auf nächste Potenz von 2 (Faktor 4) für bessere Zeitauflösung
    N_fft = max(4 * N, 2 ** int(np.ceil(np.log2(4 * N))))

    # IFFT — Spektrum auf positive Frequenzen
    tdr_complex = np.fft.ifft(s21_win, n=N_fft)
    tdr_real = np.real(tdr_complex)

    # Zeitachse
    dt = 1.0 / (N_fft * df)  # Zeitschritt in Sekunden
    time_s = np.arange(N_fft) * dt
    time_ns = time_s * 1e9

    return time_ns, tdr_real


# ── Signalverarbeitung ─────────────────────────────────────────────────────────

def dewow(trace, window_samples=None):
    """Entfernt den Gleichanteil (wow) durch gleitenden Mittelwert."""
    if window_samples is None:
        window_samples = max(3, len(trace) // 20)
    kernel = np.ones(window_samples) / window_samples
    trend = np.convolve(trace, kernel, mode="same")
    return trace - trend


def apply_tgc(traces, time_ns, gain_db_per_ns):
    """Zeitabhängige Verstärkung (Time Gain Compensation)."""
    gain_linear = 10 ** (gain_db_per_ns * time_ns / 20.0)
    return traces * gain_linear[np.newaxis, :]


# ── Tiefenachse ────────────────────────────────────────────────────────────────

def time_to_depth(time_ns, er):
    """Konvertiert Laufzeit (ns) in Tiefe (m). Faktor 2 für Hin- und Rückweg."""
    c = 0.299792458  # m/ns
    v = c / np.sqrt(er)
    depth_m = v * time_ns / 2.0
    return depth_m


# ── Hauptprogramm ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPR B-Scan aus S21 Touchstone-Dateien",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--folder",   default=".",       help="Ordner mit .s2p Dateien")
    parser.add_argument("--er",       type=float, default=9.0, help="Relative Permittivität (Standard: 9)")
    parser.add_argument("--spacing",  type=float, default=0.05, help="Spurabstand in Metern (Standard: 0.05)")
    parser.add_argument("--output",   default="bscan.png", help="Ausgabedatei")
    parser.add_argument("--tmax",     type=float, default=None, help="Max. Tiefe in Metern")
    parser.add_argument("--window",   default="hann", choices=["hann","hamming","blackman","none"])
    parser.add_argument("--gain",     type=float, default=2.0, help="TGC in dB/ns")
    parser.add_argument("--cmap",     default="bwr", help="Colormap")
    parser.add_argument("--dpi",      type=int, default=200)
    parser.add_argument("--no-dewow", action="store_true", help="Dewow deaktivieren")
    parser.add_argument("--bgrem",    action="store_true", help="Hintergrundentfernung")
    args = parser.parse_args()

    # ── Dateien einlesen ──────────────────────────────────────────────────────
    pattern = os.path.join(args.folder, "*.s2p")
    files = sorted(glob.glob(pattern))
    if not files:
        # auch .S2P (Großbuchstaben) versuchen
        files = sorted(glob.glob(os.path.join(args.folder, "*.S2P")))
    if not files:
        print(f"[Fehler] Keine .s2p Dateien in '{args.folder}' gefunden.")
        sys.exit(1)

    print(f"Gefunden: {len(files)} Dateien")

    all_traces = []
    time_ns = None

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            freqs, s21 = parse_s2p(fpath)
            t_ns, tdr = compute_tdr(freqs, s21, window_type=args.window)

            if time_ns is None:
                time_ns = t_ns
                N_time = len(t_ns)

            # Auf gemeinsame Länge kürzen
            tdr = tdr[:N_time]

            # Dewow
            if not args.no_dewow:
                tdr = dewow(tdr)

            all_traces.append(tdr)

            if (i + 1) % 10 == 0 or i == len(files) - 1:
                print(f"  Verarbeitet: {i+1}/{len(files)}")

        except Exception as e:
            print(f"  [Warnung] Datei übersprungen: {fname} — {e}")

    if not all_traces:
        print("[Fehler] Keine gültigen Spuren geladen.")
        sys.exit(1)

    # ── B-Scan Matrix ─────────────────────────────────────────────────────────
    bscan = np.array(all_traces)  # shape: (n_traces, n_time)

    # Hintergrundentfernung (mittlere Spur abziehen)
    if args.bgrem:
        mean_trace = np.mean(bscan, axis=0)
        bscan -= mean_trace[np.newaxis, :]

    # TGC anwenden
    bscan = apply_tgc(bscan, time_ns, args.gain)

    # Tiefenachse
    depth_m = time_to_depth(time_ns, args.er)

    # Zeitfenster beschränken
    if args.tmax is not None:
        idx_max = np.searchsorted(depth_m, args.tmax)
    else:
        # Automatisch: sinnvolle Zeitgrenzen (bis 80% der IFFT-Länge)
        idx_max = int(0.5 * len(time_ns))

    bscan_crop = bscan[:, :idx_max]
    depth_crop = depth_m[:idx_max]

    # Positionsachse
    n_traces = bscan_crop.shape[0]
    positions = np.arange(n_traces) * args.spacing

    # ── Plot ──────────────────────────────────────────────────────────────────
    c = 0.299792458
    v = c / np.sqrt(args.er)
    print(f"\nParameter:")
    print(f"  εr = {args.er}  →  v = {v:.4f} m/ns")
    print(f"  Spurabstand: {args.spacing} m")
    print(f"  Profillänge: {positions[-1]:.2f} m")
    print(f"  Tiefenbereich: 0 – {depth_crop[-1]:.2f} m")
    print(f"  Fensterfunktion: {args.window}")
    print(f"  TGC: {args.gain} dB/ns")

    fig, ax = plt.subplots(figsize=(max(10, n_traces * 0.08), 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Normierung symmetrisch um 0
    vmax = np.percentile(np.abs(bscan_crop), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        bscan_crop.T,
        aspect="auto",
        extent=[positions[0], positions[-1], depth_crop[-1], depth_crop[0]],
        cmap=args.cmap,
        norm=norm,
        interpolation="bilinear",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Reflexionsamplitude (normiert)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Achsenbeschriftung
    ax.set_xlabel("Position [m]", color="white", fontsize=11)
    ax.set_ylabel("Tiefe [m]", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Titel
    title = (
        f"GPR B-Scan  |  {len(files)} Spuren  |  "
        f"εr = {args.er}  |  v = {v:.3f} m/ns  |  "
        f"Δx = {args.spacing*100:.0f} cm"
    )
    ax.set_title(title, color="white", fontsize=11, pad=10)

    # Grid
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(which="major", color="#333", linewidth=0.5, linestyle="--")
    ax.grid(which="minor", color="#222", linewidth=0.3)

    plt.tight_layout()
    outpath = args.output
    plt.savefig(outpath, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nBild gespeichert: {outpath}")

    # ── Info ──────────────────────────────────────────────────────────────────
    print("\nHinweis: Rohre erscheinen als charakteristische Hyperbeln.")
    print("  Scheitel der Hyperbel = Rohrtiefe")
    print("  Öffnungswinkel der Hyperbel ∝ 1/v → εr verfeinern für exakte Tiefe")


if __name__ == "__main__":
    main()
