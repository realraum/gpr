# GPR (Ground Penetrating Radar)

Messprinzip
  
               ┌──────────┐    ┌─────────────────────┐  
    ┌──────────┤   VNA    │.s2p│ nanovna_capture.py  │   
    ▼      ┌──►│(NanoVNA) |───►│    gpr_bscan.py     │ ───► BSCAN.PNG
───   ────     │          │    │                     │




  Ant1  Ant2   └──────────┘    └─────────────────────┘ 




 
 

Ich habe GPR-Daten (Ground Penetrating Radar) als S21-Touchstone-Dateien gewonnen mit einem Vectornetworkanalyzer und will:

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

# Verarbeitungspipeline
SchrittMethodeEinlesenS2P-Parser (MA / DB / RI Format, alle Frequenzeinheiten)TDRIFFT mit Zero-Padding (4×N) → ZeitbereichssignalFensterungHann / Hamming / Blackman (reduziert Seitenkeulen)DewowGleitender Mittelwert-Abzug (DC-Anteil)TGCZeitabhängige Verstärkung (kompensiert Dämpfung im Boden)Hintergrund

--bgrem: Mittelspur abziehen (entfernt Direktwelle/Ringing)Tiefenachsedepth = v · t / 2, wobei v = c / √εr

# Tipps für gute Ergebnisse

--bgrem aktivieren wenn die Direktwelle (oberste Zeile) die Tiefenreflexionen überlagert
εr bestimmen über Hyperbel-Fitting: flachere Hyperbeln → höheres εr, steilere → niedrigeres εr

--tmax setzen um den relevanten Tiefenbereich zu beschränken (sonst erscheinen IFFT-Artefakte)

--gain erhöhen (z.B. 3–5 dB/ns) wenn tiefe Reflexionen zu schwach erscheinen

# Beispiel für Objekte in Luft
--folder "." --bgrem --er 1 --gain 0.1

Rohre erscheinen als Hyperbeln — Scheitel = Rohrtiefe, Öffnung hängt von v ab

# Literatur

https://github.com/spel-oe/gpr

https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00021644/ilm1-2010000403.pdf

https://www.microwavejournal.com/articles/1866-characterization-of-ultra-wideband-bow-tie-antennas-for-ground-penetrating-radar-systems

https://www.researchgate.net/publication/370756883_Bow-Tie_Antenna_with_Improved_Performance_for_Advanced_GPR_Applications

https://www.sciencedirect.com/science/article/abs/pii/S0926985114003139


