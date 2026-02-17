# Dicejack – Q-Learning Reinforcement Learning Project

This project implements Q-Learning for a Blackjack-Variant called **Dicejack**.  
Agent learns when to *HIT* or *STAND* by repeatedly interacting with the environment without any prior knowledge thereof.

Project demonstrates basic Reinforcement Learning concepts like:
- Q-Learning
- Temporal-Difference Learning
- Exploration vs. Exploitation
- Policy-Visualisation

---

## Idea



- Agent and dealer play with 2 dice each
- First visible state is own sum of 2 dice and dealers sum
- Actions:
  - **HIT** → roll again
  - **STAND** → let the dealer play
- dealer has to hit until 17
- going over 21 → bust

Rewards
- Win: +1  
- Lose: −1  
- Draw: 0  

---

## Structure

## dicejack_env.py
Implements the entire environment
- State `(player_sum, dealer_up)`
- Action Logic
- Dealer-Policy
- Reward-Handling

## qlearning.py
Contains:
- Q-Table
- Q-Learning-Updates
- ε-greedy-Strategy 


## config.py
Hyperparameters:
- Learning Rate (α)
- Discount Factor (γ)
- Number of Episodes
- Epsilon-Decay

## app.py
Interactive Streamlit UI:
- Training-Tab
- Evaluation-Tab
- Policy-Heatmap
- Live-Gameplay after Training

---

###  Installation

## Clone repository
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

## Start Application

```bash
streamlit run app.py
```

