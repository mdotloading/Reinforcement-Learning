from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import numpy as np
from dicejack_env import DicejackEnv, State, STAND, HIT

Action = int


@dataclass
class TrainConfig:
    episodes: int = 50_000
    alpha: float = 0.10
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    seed: Optional[int] = 42


def _init_q() -> Dict[State, np.ndarray]:
    # player_sum 2..21, dealer_start_sum 2..12
    q: Dict[State, np.ndarray] = {}
    for ps in range(2, 22):
        for dss in range(2, 13):
            q[(ps, dss)] = np.zeros(2, dtype=np.float64)  # [stand, hit]
    return q


def choose_action(q: Dict[State, np.ndarray], s: State, epsilon: float, rng: np.random.Generator) -> Action:
    if rng.random() < epsilon:
        return int(rng.integers(0, 2))
    return int(np.argmax(q[s]))


def train_q_learning(cfg: TrainConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    env = DicejackEnv(seed=cfg.seed)
    q = _init_q()

    eps = float(cfg.epsilon_start)
    outcomes: List[float] = []
    win_rate_ma: List[float] = []

    def update_ma():
        window = 2000
        chunk = outcomes[-window:] if len(outcomes) >= window else outcomes
        if not chunk:
            win_rate_ma.append(0.0)
        else:
            wins = sum(1 for r in chunk if r > 0)
            win_rate_ma.append(wins / len(chunk))

    for _ in range(cfg.episodes):
        s = env.reset()
        terminal = env.check_terminal_after_reset()
        if terminal is not None:
            outcomes.append(terminal.reward)
            update_ma()
            eps = max(cfg.epsilon_end, eps * cfg.epsilon_decay)
            continue

        done = False
        while not done:
            a = choose_action(q, s, eps, rng)
            res = env.step(a)
            r = res.reward
            done = res.done

            old = q[s][a]
            if done:
                target = r
            else:
                s2 = res.state
                assert s2 is not None
                target = r + cfg.gamma * float(np.max(q[s2]))
            q[s][a] = old + cfg.alpha * (target - old)

            if done:
                outcomes.append(r)
            else:
                s = res.state  # type: ignore

        update_ma()
        eps = max(cfg.epsilon_end, eps * cfg.epsilon_decay)

    return {"q": q, "outcomes": outcomes, "win_rate_ma": win_rate_ma, "final_epsilon": eps, "cfg": cfg}


def evaluate_policy(q: Dict[State, np.ndarray], n_games: int = 20_000, seed: Optional[int] = 123) -> Dict[str, float]:
    env = DicejackEnv(seed=seed)
    wins = losses = draws = naturals = 0

    for _ in range(n_games):
        s = env.reset()
        terminal = env.check_terminal_after_reset()
        if terminal is not None:
            if terminal.reward == 1.5:
                naturals += 1
                wins += 1
            elif terminal.reward > 0:
                wins += 1
            elif terminal.reward < 0:
                losses += 1
            else:
                draws += 1
            continue

        done = False
        while not done:
            a = int(np.argmax(q[s]))  # greedy
            res = env.step(a)
            done = res.done
            if done:
                r = res.reward
                if r > 0:
                    wins += 1
                elif r < 0:
                    losses += 1
                else:
                    draws += 1
            else:
                s = res.state  # type: ignore

    total = wins + losses + draws
    return {
        "win_rate": wins / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
        "draw_rate": draws / total if total else 0.0,
        "natural_rate": naturals / total if total else 0.0,
    }


def policy_table(q: Dict[State, np.ndarray]) -> np.ndarray:
    # rows: player_sum 2..21 (20), cols: dealer_start_sum 2..12 (11)
    arr = np.zeros((20, 11), dtype=int)
    for ps in range(2, 22):
        for dss in range(2, 13):
            arr[ps - 2, dss - 2] = int(np.argmax(q[(ps, dss)]))
    return arr





def evaluate_simple_strategy(n_games: int = 20_000, seed: int = 123, threshold: int = 17) -> Dict[str, float]:
    """Hit until player_sum >= threshold, then STAND"""
    env = DicejackEnv(seed=seed)
    wins = losses = draws = 0

    for _ in range(n_games):
        s = env.reset()
        terminal = env.check_terminal_after_reset()
        if terminal is not None:
            if terminal.reward > 0: wins += 1
            elif terminal.reward < 0: losses += 1
            else: draws += 1
            continue

        done = False
        while not done:
            player_sum, dealer_start = s
            a = HIT if player_sum < threshold else STAND
            res = env.step(a)
            done = res.done
            if done:
                if res.reward > 0: wins += 1
                elif res.reward < 0: losses += 1
                else: draws += 1
            else:
                s = res.state

    total = wins + losses + draws
    return {"win_rate": wins / total, "loss_rate": losses / total, "draw_rate": draws / total}