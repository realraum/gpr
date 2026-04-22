#!/usr/bin/env python3
"""
NanoVNA V2 (S-A-A-2) S21 Capture
==================================
Verbindet sich mit einem NanoVNA V2 über USB und speichert bei Tastendruck
S21-Messungen als Touchstone (.s2p) Dateien mit chronologisch aufsteigendem
Dateinamen.

Das NanoVNA V2 verwendet ein BINÄRES Register-Protokoll (kein Textprotokoll
wie der NanoVNA V1). Das Gerät wechselt beim ersten Schreibzugriff automatisch
in den "USB MODE" (Display zeigt "USB MODE").

Voraussetzungen:
    pip install pyserial numpy

Windows: Cypress USB-CDC Treiber erforderlich (von nanorfe.com/nanovna-v2-software.html)
Linux/Mac: Kein Treiber nötig (USB-CDC Klasse)

Verwendung:
    python nanovna2_capture.py
    python nanovna2_capture.py --port COM3 --points 201 --outdir ./messungen
    python nanovna2_capture.py --port /dev/ttyACM0 --start 100e6 --stop 3e9 --points 101

Argumente:
    --port      Serieller Port (Standard: automatische Erkennung)
    --baud      Baudrate (Standard: 115200)
    --points    Anzahl Messpunkte 1–1024 (Standard: 101)
    --start     Startfrequenz in Hz (Standard: 100 MHz)
    --stop      Stoppfrequenz in Hz (Standard: 3 GHz)
    --avg       Mittelungen pro Messpunkt (Standard: 1)
    --outdir    Ausgabeordner (Standard: ./captures)
    --prefix    Dateiname-Präfix (Standard: trace)

Steuerung:
    LEERTASTE oder ENTER  →  Messung speichern
    q                     →  Beenden
"""

import argparse
import os
import sys
import time
import datetime
import threading

import numpy as np
import serial
import serial.tools.list_ports
from struct import pack, unpack_from

# ── NanoVNA V2 Binärprotokoll Konstanten ──────────────────────────────────────
# Quelle: UG1101 Appendix II + offizieller NanoRFE Beispielcode
# https://gist.github.com/nanovna/af6d7c93221a5673451e6f6e64f210e7

CMD_NOP       = 0x00
CMD_INDICATE  = 0x0D
CMD_READ      = 0x10  # Read 1 byte  from register AA
CMD_READ2     = 0x11  # Read 2 bytes from register AA
CMD_READ4     = 0x12  # Read 4 bytes from register AA
CMD_READFIFO  = 0x18  # Read NN values from FIFO at AA
CMD_WRITE     = 0x20  # Write 1 byte  to register AA
CMD_WRITE2    = 0x21  # Write 2 bytes to register AA (little-endian)
CMD_WRITE4    = 0x22  # Write 4 bytes to register AA
CMD_WRITE8    = 0x23  # Write 8 bytes to register AA
CMD_WRITEFIFO = 0x28  # Write NN bytes to FIFO at AA

# Register-Adressen
ADDR_SWEEP_START       = 0x00  # uint64, Startfrequenz in Hz
ADDR_SWEEP_STEP        = 0x10  # uint64, Schrittweite in Hz
ADDR_SWEEP_POINTS      = 0x20  # uint16, Anzahl Frequenzpunkte
ADDR_SWEEP_VALS_PER_F  = 0x22  # uint8,  Mittelungen pro Punkt
ADDR_VALUES_FIFO       = 0x30  # FIFO: je 32 Bytes pro Messpunkt
ADDR_DEVICE_VARIANT    = 0xF0  # uint8, 0x02 = NanoVNA V2
ADDR_PROTOCOL_VERSION  = 0xF1  # uint8, 0x01
ADDR_HW_REVISION       = 0xF2
ADDR_FW_MAJOR          = 0xF3
ADDR_FW_MINOR          = 0xF4

WRITE_SLEEP = 0.05  # Pause nach jedem Schreibbefehl (s)

# ── NanoVNA V2 Geräteklasse ───────────────────────────────────────────────────

class NanoVNA2:
    """
    Kommunikation mit NanoVNA V2 / S-A-A-2 über binäres USB-Protokoll.

    FIFO-Datenformat (32 Bytes pro Punkt, little-endian int32):
      Offset  0: fwd_real   (Referenzkanal Real)
      Offset  4: fwd_imag   (Referenzkanal Imag)
      Offset  8: rev0_real  (Port1/S11 Real)
      Offset 12: rev0_imag  (Port1/S11 Imag)
      Offset 16: rev1_real  (Port2/S21 Real)
      Offset 20: rev1_imag  (Port2/S21 Imag)
      Offset 24: freq_index (int16, Frequenzindex 0..sweepPoints-1)
      Offset 26: (6 Bytes reserviert)
    """

    def __init__(self, port, baud=115200, timeout=10):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.sweep_start_hz = 100e6
        self.sweep_step_hz  = 1e6
        self.sweep_points   = 101
        self.sweep_avg      = 1

    def connect(self):
        self.ser = serial.Serial(
            self.port,
            baudrate=self.baud,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        time.sleep(0.3)
        # Protokoll-Reset: 8 Null-Bytes senden
        self.ser.write(pack("<Q", 0))
        time.sleep(WRITE_SLEEP)
        return self.read_version()

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    # ── Low-Level Register I/O ────────────────────────────────────────────────

    def _write1(self, addr, value):
        self.ser.write(pack("<BBB", CMD_WRITE, addr, value & 0xFF))
        time.sleep(WRITE_SLEEP)

    def _write2(self, addr, value):
        self.ser.write(pack("<BBH", CMD_WRITE2, addr, value & 0xFFFF))
        time.sleep(WRITE_SLEEP)

    def _write8(self, addr, value):
        self.ser.write(pack("<BBQ", CMD_WRITE8, addr, int(value)))
        time.sleep(WRITE_SLEEP)

    def _read1(self, addr):
        self.ser.write(pack("<BB", CMD_READ, addr))
        time.sleep(WRITE_SLEEP)
        data = self.ser.read(1)
        return data[0] if data else 0

    # ── Firmware-Version ──────────────────────────────────────────────────────

    def read_version(self):
        variant  = self._read1(ADDR_DEVICE_VARIANT)
        proto    = self._read1(ADDR_PROTOCOL_VERSION)
        hw_rev   = self._read1(ADDR_HW_REVISION)
        fw_major = self._read1(ADDR_FW_MAJOR)
        fw_minor = self._read1(ADDR_FW_MINOR)
        if fw_major == 0xFF:
            raise IOError("Gerät ist im DFU-Modus!")
        return {
            "variant": variant,
            "protocol": proto,
            "hw_rev": hw_rev,
            "fw": f"{fw_major}.{fw_minor}",
        }

    # ── Sweep-Konfiguration ───────────────────────────────────────────────────

    def set_sweep(self, start_hz, stop_hz, points, avg=1):
        self.sweep_start_hz = int(start_hz)
        self.sweep_points   = int(points)
        self.sweep_avg      = int(avg)
        step_hz = int((stop_hz - start_hz) / max(1, points - 1))
        self.sweep_step_hz  = step_hz

        # Reihenfolge wichtig: start → step → points → avg
        # Schreiben eines beliebigen Sweep-Registers aktiviert USB-Modus
        self._write8(ADDR_SWEEP_START, self.sweep_start_hz)
        self._write8(ADDR_SWEEP_STEP,  self.sweep_step_hz)
        self._write2(ADDR_SWEEP_POINTS, self.sweep_points)
        self._write1(ADDR_SWEEP_VALS_PER_F, self.sweep_avg)

    def get_frequencies(self):
        """Berechnet Frequenzachse aus konfigurierten Sweep-Parametern."""
        return np.array([
            self.sweep_start_hz + i * self.sweep_step_hz
            for i in range(self.sweep_points)
        ])

    # ── Messung lesen ─────────────────────────────────────────────────────────

    def read_data(self):
        """
        Liest einen vollständigen Sweep aus dem FIFO.
        Gibt (s11_array, s21_array) als komplexe numpy-Arrays zurück.
        """
        n   = self.sweep_points
        avg = self.sweep_avg

        # Akkumulator: [fwd, rev0(S11), _, rev1(S21)] pro Frequenzpunkt
        acc      = np.zeros((n, 4), dtype=complex)
        count    = np.zeros(n, dtype=int)

        # FIFO leeren: Schreibe beliebigen Wert an 0x30
        self.ser.write(pack("<BBB", CMD_WRITE, ADDR_VALUES_FIFO, 0))
        time.sleep(WRITE_SLEEP)
        # Gepufferte Daten verwerfen
        old_timeout = self.ser.timeout
        self.ser.timeout = 0.05
        self.ser.read(4096)
        self.ser.timeout = max(3.0, avg * 2.0)

        points_left = n * avg
        while points_left > 0:
            to_read = min(255, points_left)
            # FIFO-Leseanfrage
            self.ser.write(pack("<BBB", CMD_READFIFO, ADDR_VALUES_FIFO, to_read))
            time.sleep(WRITE_SLEEP)

            n_bytes = to_read * 32
            raw = self.ser.read(n_bytes)

            if len(raw) < n_bytes:
                raise RuntimeError(
                    f"Zu wenig Bytes empfangen: {len(raw)}/{n_bytes}. "
                    "Timeout oder Verbindungsproblem."
                )

            # 32-Byte Pakete parsen
            for i in range(to_read):
                (fwd_r, fwd_i,
                 rev0_r, rev0_i,
                 rev1_r, rev1_i,
                 fi) = unpack_from("<iiiiiih6x", raw, i * 32)

                if fi < 0 or fi >= n:
                    continue  # Ungültiger Index überspringen

                fwd  = complex(fwd_r,  fwd_i)
                rev0 = complex(rev0_r, rev0_i)
                rev1 = complex(rev1_r, rev1_i)
                acc[fi] += np.array([fwd, rev0, 0.0, rev1])
                count[fi] += 1

            points_left -= to_read

        self.ser.timeout = old_timeout

        # Mittelwert bilden und S-Parameter berechnen
        # S11 = rev0 / fwd,  S21 = rev1 / fwd
        s11 = np.zeros(n, dtype=complex)
        s21 = np.zeros(n, dtype=complex)
        for i in range(n):
            c = count[i]
            if c == 0:
                continue
            fwd  = acc[i, 0] / c
            rev0 = acc[i, 1] / c
            rev1 = acc[i, 3] / c
            if abs(fwd) > 0:
                s11[i] = rev0 / fwd
                s21[i] = rev1 / fwd

        return s11, s21


# ── Port-Erkennung ────────────────────────────────────────────────────────────

NANOVNA2_USB_IDS = [
    (0x04b4, 0x0008),  # Cypress USB-Serial (NanoVNA V2)
    (0x16c0, 0x0483),  # NanoVNA-SAA2 Variante
    (0x0483, 0x5740),  # STM32 (manche Klone)
]

def find_nanovna2_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if port.vid and port.pid:
            for vid, pid in NANOVNA2_USB_IDS:
                if port.vid == vid and port.pid == pid:
                    return port.device
    # Namensbasierter Fallback
    for port in ports:
        desc = (port.description or "").lower()
        if any(kw in desc for kw in ["cypress", "usb serial", "acm", "usbmodem"]):
            return port.device
    return None


# ── Touchstone Export ─────────────────────────────────────────────────────────

def save_s2p(filepath, freqs, s11, s21, z0=50.0):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"! NanoVNA V2 (S-A-A-2) Capture",
        f"! Datum:    {now}",
        f"! Punkte:   {len(freqs)}",
        f"! Format:   RI (Real/Imag)",
        f"# Hz S RI R {z0:.0f}",
    ]
    for i, f in enumerate(freqs):
        s11_r = np.real(s11[i])
        s11_i = np.imag(s11[i])
        s21_r = np.real(s21[i])
        s21_i = np.imag(s21[i])
        lines.append(
            f"{f:.6e}  "
            f"{s11_r:.10f} {s11_i:.10f}  "
            f"{s21_r:.10f} {s21_i:.10f}  "
            f"{s21_r:.10f} {s21_i:.10f}  "
            f"{s11_r:.10f} {s11_i:.10f}"
        )
    with open(filepath, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def generate_filename(outdir, prefix, index):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(outdir, f"{prefix}_{index:04d}_{ts}.s2p")


# ── Konsolenausgabe ───────────────────────────────────────────────────────────

def db20(s):
    a = np.abs(s)
    a = np.where(a < 1e-12, 1e-12, a)
    return 20 * np.log10(a)

def print_status(freqs, s11, s21, index, last_file):
    s21_db = db20(s21)
    s11_db = db20(s11)
    f_mhz  = freqs / 1e6

    print("\033[2J\033[H", end="")
    print("━" * 62)
    print("  NanoVNA V2 (S-A-A-2) Capture")
    print("━" * 62)
    print(f"  Frequenz : {f_mhz[0]:.1f} – {f_mhz[-1]:.1f} MHz  ({len(freqs)} Punkte)")
    print(f"  S21 min  : {s21_db.min():.2f} dB  @ {f_mhz[np.argmin(s21_db)]:.1f} MHz")
    print(f"  S21 max  : {s21_db.max():.2f} dB  @ {f_mhz[np.argmax(s21_db)]:.1f} MHz")
    print(f"  S11 min  : {s11_db.min():.2f} dB  @ {f_mhz[np.argmin(s11_db)]:.1f} MHz")
    print(f"  Gespeichert: {index} Messung(en)")
    if last_file:
        print(f"  Letzte Datei: {os.path.basename(last_file)}")
    print("━" * 62)

    # Mini-Balkendiagramm S21
    n_bars = 42
    step = max(1, len(s21_db) // n_bars)
    vals = s21_db[::step][:n_bars]
    v_min, v_max = vals.min(), vals.max()
    v_rng = v_max - v_min if v_max != v_min else 1.0
    bar_h = 7
    print()
    for row in range(bar_h, 0, -1):
        thr = v_min + (row / bar_h) * v_rng
        bar = "".join("█" if v >= thr else " " for v in vals)
        if row == bar_h:   lbl = f"{v_max:+6.1f} dB"
        elif row == 1:     lbl = f"{v_min:+6.1f} dB"
        else:              lbl = "         "
        print(f"  {lbl} │{bar}│")
    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  LEERTASTE / ENTER  →  Messung speichern             │")
    print("  │  q + ENTER          →  Beenden                       │")
    print("  └──────────────────────────────────────────────────────┘\n")


# ── Keyboard-Eingabe ──────────────────────────────────────────────────────────

def get_key():
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
        import msvcrt
        return msvcrt.getwch()


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NanoVNA V2 (S-A-A-2) S21 Capture → Touchstone .s2p",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--port",    default=None)
    parser.add_argument("--baud",    type=int,   default=115200)
    parser.add_argument("--points",  type=int,   default=101,   help="Messpunkte 1–1024")
    parser.add_argument("--start",   type=float, default=200e6, help="Startfrequenz Hz")
    parser.add_argument("--stop",    type=float, default=1000e6,help="Stoppfrequenz Hz")
    parser.add_argument("--avg",     type=int,   default=1,     help="Mittelungen (Standard: 1)")
    parser.add_argument("--outdir",  default="./captures")
    parser.add_argument("--prefix",  default="trace")
    args = parser.parse_args()

    # Port ermitteln
    port = args.port or find_nanovna2_port()
    if port is None:
        print("[Fehler] Kein NanoVNA V2 gefunden. Verfügbare Ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  [{p.vid:#06x}:{p.pid:#06x}]  {p.description}")
        print("\nBitte mit --port <PORT> angeben.")
        sys.exit(1)
    print(f"Verbinde mit: {port}")

    os.makedirs(args.outdir, exist_ok=True)

    vna = NanoVNA2(port, baud=args.baud)
    try:
        info = vna.connect()
        print(f"Verbunden. Gerät: Variante {info['variant']:#04x}  "
              f"FW {info['fw']}  HW-Rev {info['hw_rev']}  "
              f"Protokoll v{info['protocol']}")
    except Exception as e:
        print(f"[Fehler] Verbindung fehlgeschlagen: {e}")
        sys.exit(1)

    step = int((args.stop - args.start) / max(1, args.points - 1))
    print(f"Konfiguriere: {args.start/1e6:.2f} – {args.stop/1e6:.2f} MHz, "
          f"{args.points} Punkte, Schrittweite {step/1e3:.1f} kHz ...")
    print("  → Gerät wechselt jetzt in USB-Modus (Display zeigt 'USB MODE')")

    vna.set_sweep(args.start, args.stop, args.points, args.avg)
    freqs = vna.get_frequencies()
    print(f"  OK. Erste Messung läuft ...")

    capture_index = 0
    last_file     = None
    s11_data      = None
    s21_data      = None
    lock          = threading.Lock()
    running       = True

    def input_thread():
        nonlocal running, capture_index, last_file
        while running:
            try:
                key = get_key()
            except Exception:
                try:
                    key = input()
                except EOFError:
                    running = False
                    break

            with lock:
                if key in (" ", "\r", "\n"):
                    if s21_data is not None:
                        capture_index += 1
                        fpath = generate_filename(args.outdir, args.prefix, capture_index)
                        save_s2p(fpath, freqs, s11_data, s21_data)
                        last_file = fpath
                        print(f"\n  ✓ Gespeichert ({capture_index}): {os.path.basename(fpath)}\n")
                    else:
                        print("\n  [!] Noch keine Daten verfügbar.\n")
                elif key in ("q", "Q"):
                    running = False

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    try:
        while running:
            try:
                new_s11, new_s21 = vna.read_data()
                with lock:
                    s11_data = new_s11
                    s21_data = new_s21
                print_status(freqs, s11_data, s21_data, capture_index, last_file)
            except RuntimeError as e:
                print(f"\n  [Warnung] Messung unvollständig: {e}")
                time.sleep(0.5)
            except serial.SerialException as e:
                print(f"\n[Fehler] Serielle Verbindung unterbrochen: {e}")
                running = False
    except KeyboardInterrupt:
        print("\n\nAbgebrochen.")
    finally:
        running = False
        vna.disconnect()
        print(f"\nBeendet. {capture_index} Messung(en) in '{args.outdir}' gespeichert.")


if __name__ == "__main__":
    main()
