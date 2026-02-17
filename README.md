# Dicejack – Q-Learning Reinforcement Learning Projekt

Dieses Projekt implementiert einen Reinforcement-Learning-Agenten (Q-Learning) für ein vereinfachtes, würfelbasiertes Blackjack-Spiel namens **Dicejack**.  
Der Agent lernt durch Interaktion mit der Umgebung, wann es sinnvoll ist zu *HIT* (weiter würfeln) oder *STAND* (stehen bleiben).

Das Projekt dient als kompakte Demonstration zentraler RL-Konzepte wie:
- Q-Learning
- Temporal-Difference Learning
- Exploration vs. Exploitation
- Policy-Visualisierung

---

## Spielidee

Dicejack ist eine vereinfachte Variante von Blackjack mit sechsseitigen Würfeln.

- Spieler und Dealer starten jeweils mit zwei Würfen.
- Der Spieler sieht seine eigene Summe und den ersten Würfel des Dealers.
- Aktionen:
  - **HIT** → weiterer Würfel
  - **STAND** → Runde beenden
- Dealer würfelt automatisch bis mindestens 17.
- Über 21 → sofortiger Bust.

Belohnungen:
- Gewinn: +1  
- Verlust: −1  
- Unentschieden: 0  

---

## Projektstruktur

## dicejack_env.py
Implementiert die vollständige Spielmechanik:
- Zustandsdefinition `(player_sum, dealer_up)`
- Aktionslogik
- Dealer-Policy
- Terminalbedingungen
- Reward-Berechnung

## qlearning.py
Enthält:
- Q-Tabelle
- Q-Learning-Update-Regel
- ε-greedy-Strategie
- Trainings- und Evaluationslogik

## config.py
Zentrale Hyperparameter:
- Lernrate (α)
- Diskontfaktor (γ)
- Anzahl Episoden
- Epsilon-Decay

## app.py
Interaktive Streamlit-Oberfläche mit:
- Trainings-Tab
- Evaluations-Tab
- Policy-Heatmap
- Live-Spielmodus

---

###  Installation

## Repository klonen
```bash
git clone https://github.com/mdotloading/Reinforcement-Learning
cd Reinforcement-Learning
```

```bash
python -m venv venv
.\venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Anwendung starten

```bash
streamlit run app.py
```

