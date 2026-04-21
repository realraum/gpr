#!/usr/bin/env python3
"""
NanoVNA S21 Capture
===================
Verbindet sich mit einem NanoVNA über USB/Serial, zeigt Live-Daten
und speichert bei Tastendruck S21-Messungen als Touchstone (.s2p) Dateien
mit chronologisch aufsteigendem Dateinamen.

Voraussetzungen:
    pip install pyserial numpy

Verwendung:
    python nanovna_capture.py
    python nanovna_capture.py --port COM3 --points 101 --outdir ./messungen
    python nanovna_capture.py --port /dev/ttyACM0 --prefix scan --start 100e6 --stop 3e9

Argumente:
    --port      Serieller Port (Standard: automatische Erkennung)
    --baud      Baudrate (Standard: 115200)
    --points    Anzahl Messpunkte (Standard: 101)
    --start     Startfrequenz in Hz (Standard: 100 MHz)
    --stop      Stoppfrequenz in Hz (Standard: 3 GHz)
    --outdir    Ausgabeordner (Standard: ./captures)
    --prefix    Dateiname-Präfix (Standard: trace)
    --timeout   Serial Timeout in Sekunden (Standard: 5)

Steuerung während der Aufnahme:
    LEERTASTE oder ENTER  →  Messung speichern
    q                     →  Beenden
"""

import argparse
import os
import sys
import time
import datetime
import struct
import threading

import numpy as np
import serial
import serial.tools.list_ports

# ── NanoVNA Protokoll ──────────────────────────────────────────────────────────

class NanoVNA:
    """Kommunikation mit NanoVNA über USB-Serial (Text-Protokoll)."""

    PROMPT = b"ch> "

    def __init__(self, port, baud=115200, timeout=5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    def connect(self):
        self.ser = serial.Serial(
            self.port,
            baudrate=self.baud,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        time.sleep(0.5)
        self._flush()
        # Verbindung testen
        ver = self.send_command("version")
        return ver.strip()

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _flush(self):
        self.ser.reset_input_buffer()
        self.ser.write(b"\r\n")
        time.sleep(0.1)
        self.ser.reset_input_buffer()

    def send_command(self, cmd, wait_prompt=True):
        self.ser.write((cmd + "\r\n").encode())
        if not wait_prompt:
            return ""
        buf = b""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            chunk = self.ser.read(256)
            if chunk:
                buf += chunk
                if self.PROMPT in buf:
                    break
        # Prompt und Echo entfernen
        response = buf.replace(self.PROMPT, b"").decode(errors="replace")
        # Echo (erste Zeile = Befehl) entfernen
        lines = response.splitlines()
        lines = [l for l in lines if l.strip() and l.strip() != cmd.strip()]
        return "\n".join(lines)

    def set_sweep(self, start_hz, stop_hz, points):
        """Konfiguriert Sweep-Bereich und Punktanzahl."""
        self.send_command(f"sweep {int(start_hz)} {int(stop_hz)} {int(points)}")
        time.sleep(0.2)

    def fetch_frequencies(self):
        """Liest die aktuellen Frequenzpunkte aus."""
        raw = self.send_command("frequencies")
        freqs = []
        for line in raw.splitlines():
            line = line.strip()
            try:
                freqs.append(float(line))
            except ValueError:
                continue
        return np.array(freqs)

    def fetch_s21(self):
        """
        Liest S21 als komplexe Werte (Real/Imag).
        NanoVNA gibt 'data 1' für S21 aus.
        """
        raw = self.send_command("data 1")
        real_parts = []
        imag_parts = []
        for line in raw.splitlines():
            line = line.strip()
            parts = line.split()
            if len(parts) == 2:
                try:
                    real_parts.append(float(parts[0]))
                    imag_parts.append(float(parts[1]))
                except ValueError:
                    continue
        if not real_parts:
            return None
        return np.array(real_parts) + 1j * np.array(imag_parts)

    def fetch_s11(self):
        """Liest S11 (data 0)."""
        raw = self.send_command("data 0")
        real_parts = []
        imag_parts = []
        for line in raw.splitlines():
            line = line.strip()
            parts = line.split()
            if len(parts) == 2:
                try:
                    real_parts.append(float(parts[0]))
                    imag_parts.append(float(parts[1]))
                except ValueError:
                    continue
        if not real_parts:
            return None
        return np.array(real_parts) + 1j * np.array(imag_parts)

    def resume_sweep(self):
        """Setzt den Sweep fort (Einzelmessung auslösen)."""
        self.send_command("resume", wait_prompt=True)
        time.sleep(0.05)

    def pause_sweep(self):
        """Pausiert den Sweep."""
        self.send_command("pause", wait_prompt=True)


# ── Port-Erkennung ─────────────────────────────────────────────────────────────

NANOVNA_USB_IDS = [
    (0x0483, 0x5740),  # STM32 Virtual COM Port (NanoVNA v1)
    (0x04b4, 0x0008),  # NanoVNA-H
    (0x16c0, 0x0483),  # NanoVNA-SAA2
]

def find_nanovna_port():
    """Sucht automatisch nach einem angeschlossenen NanoVNA."""
    ports = list(serial.tools.list_ports.comports())
    # Zuerst nach bekannten VID/PID suchen
    for port in ports:
        if port.vid is not None and port.pid is not None:
            for vid, pid in NANOVNA_USB_IDS:
                if port.vid == vid and port.pid == pid:
                    return port.device
    # Fallback: nach Namen suchen
    for port in ports:
        desc = (port.description or "").lower()
        name = (port.device or "").lower()
        if any(kw in desc or kw in name for kw in ["nanovna", "stm32", "acm", "usbmodem"]):
            return port.device
    return None


# ── Touchstone Export ──────────────────────────────────────────────────────────

def save_s2p(filepath, freqs, s11, s21, z0=50.0):
    """
    Speichert S-Parameter als Touchstone 2-Port .s2p Datei (RI Format).
    s12 und s22 werden als Nullen gesetzt (nur S11 und S21 gemessen).
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"! NanoVNA S21 Capture",
        f"! Datum: {now}",
        f"! Frequenzpunkte: {len(freqs)}",
        f"! Format: RI (Real/Imag)",
        f"# Hz S RI R {z0:.0f}",
    ]
    for i, f in enumerate(freqs):
        s11_r = np.real(s11[i]) if s11 is not None else 0.0
        s11_i = np.imag(s11[i]) if s11 is not None else 0.0
        s21_r = np.real(s21[i])
        s21_i = np.imag(s21[i])
        # S2P: S11 S21 S12 S22 (S12=S21, S22=S11 für passive Messung)
        line = (
            f"{f:.6e}  "
            f"{s11_r:.10f} {s11_i:.10f}  "
            f"{s21_r:.10f} {s21_i:.10f}  "
            f"{s21_r:.10f} {s21_i:.10f}  "   # S12 ≈ S21
            f"{s11_r:.10f} {s11_i:.10f}"      # S22 ≈ S11
        )
        lines.append(line)
    with open(filepath, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def generate_filename(outdir, prefix, index):
    """Erzeugt chronologisch aufsteigenden Dateinamen."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{index:04d}_{timestamp}.s2p"
    return os.path.join(outdir, fname)


# ── Konsolenausgabe ────────────────────────────────────────────────────────────

def db(s):
    """Umrechnung in dB."""
    amp = np.abs(s)
    amp = np.where(amp < 1e-12, 1e-12, amp)
    return 20 * np.log10(amp)

def print_status(freqs, s21, index, last_file):
    """Zeigt Live-Status in der Konsole."""
    if s21 is None or len(s21) == 0:
        return
    s21_db = db(s21)
    f_mhz = freqs / 1e6
    print("\033[2J\033[H", end="")  # Terminal löschen
    print("━" * 60)
    print("  NanoVNA S21 Capture")
    print("━" * 60)
    print(f"  Frequenz:   {f_mhz[0]:.1f} – {f_mhz[-1]:.1f} MHz  ({len(freqs)} Punkte)")
    print(f"  S21 min:    {s21_db.min():.2f} dB  @ {f_mhz[np.argmin(s21_db)]:.1f} MHz")
    print(f"  S21 max:    {s21_db.max():.2f} dB  @ {f_mhz[np.argmax(s21_db)]:.1f} MHz")
    print(f"  S21 Mitte:  {s21_db[len(s21_db)//2]:.2f} dB  @ {f_mhz[len(f_mhz)//2]:.1f} MHz")
    print(f"  Gespeichert: {index} Messung(en)")
    if last_file:
        print(f"  Letzte Datei: {os.path.basename(last_file)}")
    print("━" * 60)
    # Mini-Balkendiagramm
    n_bars = 40
    step = max(1, len(s21_db) // n_bars)
    vals = s21_db[::step][:n_bars]
    v_min, v_max = vals.min(), vals.max()
    v_range = v_max - v_min if v_max != v_min else 1.0
    bar_h = 8
    print()
    for row in range(bar_h, 0, -1):
        threshold = v_min + (row / bar_h) * v_range
        bar = ""
        for v in vals:
            bar += "█" if v >= threshold else " "
        if row == bar_h:
            label = f"{v_max:+6.1f} dB"
        elif row == 1:
            label = f"{v_min:+6.1f} dB"
        else:
            label = "        "
        print(f"  {label} │{bar}│")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  LEERTASTE / ENTER  →  Messung speichern        │")
    print("  │  q + ENTER          →  Beenden                  │")
    print("  └─────────────────────────────────────────────────┘")
    print()


# ── Keyboard-Eingabe (plattformübergreifend) ───────────────────────────────────

def get_key():
    """Liest eine Taste (blockierend). Gibt None zurück bei Fehler."""
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
    except ImportError:
        # Windows
        import msvcrt
        ch = msvcrt.getwch()
        return ch


# ── Hauptprogramm ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NanoVNA S21 Capture → Touchstone .s2p",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--port",    default=None,      help="Serieller Port (Standard: auto)")
    parser.add_argument("--baud",    type=int, default=115200)
    parser.add_argument("--points",  type=int, default=201, help="Messpunkte (Standard: 101)")
    parser.add_argument("--start",   type=float, default=1e6, help="Startfrequenz Hz")
    parser.add_argument("--stop",    type=float, default=1300e6, help="Stoppfrequenz Hz")
    parser.add_argument("--outdir",  default="./captures", help="Ausgabeordner")
    parser.add_argument("--prefix",  default="trace", help="Dateiname-Präfix")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    # Port ermitteln
    port = args.port
    if port is None:
        port = find_nanovna_port()
        if port is None:
            print("[Fehler] Kein NanoVNA gefunden. Verfügbare Ports:")
            for p in serial.tools.list_ports.comports():
                print(f"  {p.device}  —  {p.description}")
            print("\nBitte mit --port <PORT> angeben.")
            sys.exit(1)
        print(f"NanoVNA gefunden: {port}")

    # Ausgabeordner
    os.makedirs(args.outdir, exist_ok=True)

    # Verbinden
    vna = NanoVNA(port, baud=args.baud, timeout=args.timeout)
    try:
        version = vna.connect()
        print(f"Verbunden. Firmware: {version or '(unbekannt)'}")
    except Exception as e:
        print(f"[Fehler] Verbindung fehlgeschlagen: {e}")
        sys.exit(1)

    # Sweep konfigurieren
    print(f"Konfiguriere: {args.start/1e6:.1f} – {args.stop/1e6:.1f} MHz, {args.points} Punkte ...")
    vna.set_sweep(args.start, args.stop, args.points)
    time.sleep(0.5)

    freqs = vna.fetch_frequencies()
    if len(freqs) == 0:
        # Frequenzen manuell generieren wenn Gerät keine liefert
        freqs = np.linspace(args.start, args.stop, args.points)
        print("  (Frequenzachse manuell generiert)")
    else:
        print(f"  Frequenzachse: {freqs[0]/1e6:.2f} – {freqs[-1]/1e6:.2f} MHz")

    capture_index = 0
    last_file = None
    s21 = None
    s11 = None

    print("\nBereit. Starte Live-Anzeige ...\n")
    time.sleep(0.5)

    # Live-Loop
    running = True
    input_lock = threading.Lock()

    def input_loop():
        nonlocal running, capture_index, last_file
        while running:
            try:
                key = get_key()
            except Exception:
                # Fallback für Umgebungen ohne raw terminal
                try:
                    key = input()
                except EOFError:
                    running = False
                    break

            with input_lock:
                if key in (" ", "\r", "\n"):
                    # Messung speichern
                    if s21 is not None and len(s21) > 0:
                        capture_index += 1
                        fpath = generate_filename(args.outdir, args.prefix, capture_index)
                        save_s2p(fpath, freqs, s11, s21)
                        last_file = fpath
                        print(f"\n  ✓ Gespeichert: {os.path.basename(fpath)}\n")
                    else:
                        print("\n  [!] Noch keine Messdaten verfügbar.\n")
                elif key in ("q", "Q"):
                    running = False
                    break

    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()

    try:
        while running:
            try:
                vna.resume_sweep()
                # Kurz warten damit Sweep abgeschlossen
                time.sleep(0.3)
                new_s21 = vna.fetch_s21()
                new_s11 = vna.fetch_s11()

                with input_lock:
                    if new_s21 is not None and len(new_s21) > 0:
                        s21 = new_s21
                        s11 = new_s11 if new_s11 is not None else np.zeros_like(s21)

                        # Sicherstellen dass Längen übereinstimmen
                        if len(freqs) != len(s21):
                            freqs = np.linspace(args.start, args.stop, len(s21))

                        print_status(freqs, s21, capture_index, last_file)
                    else:
                        print("  [Warte auf Daten ...]")

            except serial.SerialException as e:
                print(f"\n[Fehler] Serielle Verbindung: {e}")
                running = False
                break
            except Exception as e:
                print(f"\n[Warnung] {e}")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nAbgebrochen.")
    finally:
        running = False
        vna.disconnect()
        print(f"\nBeendet. {capture_index} Messung(en) in '{args.outdir}' gespeichert.")


if __name__ == "__main__":
    main()
 