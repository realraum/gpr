Ich habe GPR-Daten (Ground Penetrating Radar) als S21-Touchstone-Dateien und möchte:

IFFT/TDR anwenden um Zeitbereich-Reflexionen zu berechnen,
B-Scan (2D Radarbild) über alle Messpositionen erstellen
Hyperbeln von vergrabenen Objekten sichtbar machen


# Minimal (εr = 9, Spurabstand = 5 cm)
python gpr_bscan.py --folder ./messungen --er 9 --spacing 0.05

# Vollständig
python gpr_bscan.py \
  --folder ./messungen \
  --er 6.5 \
  --spacing 0.03 \
  --output bscan.png \
  --tmax 2.0 \
  --window hann \
  --gain 2.0 \
  --cmap bwr \
  --bgrem

Verarbeitungspipeline
SchrittMethodeEinlesenS2P-Parser (MA / DB / RI Format, alle Frequenzeinheiten)TDRIFFT mit Zero-Padding (4×N) → ZeitbereichssignalFensterungHann / Hamming / Blackman (reduziert Seitenkeulen)DewowGleitender Mittelwert-Abzug (DC-Anteil)TGCZeitabhängige Verstärkung (kompensiert Dämpfung im Boden)Hintergrund

--bgrem: Mittelspur abziehen (entfernt Direktwelle/Ringing)Tiefenachsedepth = v · t / 2, wobei v = c / √εr

Tipps für gute Ergebnisse

--bgrem aktivieren wenn die Direktwelle (oberste Zeile) die Tiefenreflexionen überlagert
εr bestimmen über Hyperbel-Fitting: flachere Hyperbeln → höheres εr, steilere → niedrigeres εr

--tmax setzen um den relevanten Tiefenbereich zu beschränken (sonst erscheinen IFFT-Artefakte)

--gain erhöhen (z.B. 3–5 dB/ns) wenn tiefe Reflexionen zu schwach erscheinen
Rohre erscheinen als Hyperbeln — Scheitel = Rohrtiefe, Öffnung hängt von v ab
