from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

Action = int  # 0=STAND, 1=HIT
State = Tuple[int, int]  # (player_sum, dealer_start_sum)

STAND = 0
HIT = 1


@dataclass
class StepResult:
    state: Optional[State]
    reward: float
    done: bool
    info: Dict[str, Any]


class DicejackEnv:
    """
    Dicejack (W6) — Variante: HIT = 2 Würfel (2d6)

    - Spieler & Dealer starten mit je 2 Würfen (2d6)
    - Beobachtung: (player_sum, dealer_start_sum) wobei dealer_start_sum = Summe der ersten 2 Dealerwürfel
    - Spieler: HIT (+2 Würfel) oder STAND
    - Dealer: würfelt weiter (+2 Würfel) bis Summe >= 17
    - Reward: win +1, loss -1, draw 0 (Natural nach Deal bleibt im Code, tritt aber nicht auf)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.player_sum: int = 0
        self.dealer_sum: int = 0
        self.dealer_start_sum: int = 0
        self._started: bool = False

        self._player_first_two: Tuple[int, int] = (0, 0)
        self._dealer_first_two: Tuple[int, int] = (0, 0)

    def _roll(self) -> int:
        return int(self.rng.integers(1, 7))  # 1..6

    def _roll2(self) -> int:
        # 2 dice per hit
        return self._roll() + self._roll()

    def reset(self) -> State:
        p1, p2 = self._roll(), self._roll()
        d1, d2 = self._roll(), self._roll()

        self._player_first_two = (p1, p2)
        self._dealer_first_two = (d1, d2)

        self.player_sum = p1 + p2
        self.dealer_sum = d1 + d2
        self.dealer_start_sum = self.dealer_sum
        self._started = True

        return (self.player_sum, self.dealer_start_sum)

    def check_terminal_after_reset(self) -> Optional[StepResult]:
        if not self._started:
            raise RuntimeError("Call reset() first.")

        # Bei 2d6 kann 21 nicht direkt nach Deal passieren (max 12),
        # aber wir lassen die Logik drin, falls du später Regeln änderst.
        if self.player_sum == 21 and self.dealer_sum == 21:
            return StepResult(
                state=None,
                reward=0.0,
                done=True,
                info={"reason": "both_natural", "player_sum": 21, "dealer_sum": 21},
            )
        if self.player_sum == 21:
            return StepResult(
                state=None,
                reward=1.5,
                done=True,
                info={"reason": "player_natural", "player_sum": 21, "dealer_sum": self.dealer_sum},
            )
        if self.dealer_sum == 21:
            return StepResult(
                state=None,
                reward=-1.0,
                done=True,
                info={"reason": "dealer_natural", "player_sum": self.player_sum, "dealer_sum": 21},
            )
        return None

    def step(self, action: Action) -> StepResult:
        if not self._started:
            raise RuntimeError("Call reset() first.")
        if action not in (STAND, HIT):
            raise ValueError("Action must be 0 (STAND) or 1 (HIT).")

        if action == HIT:
            self.player_sum += self._roll2()  # NEW: +2 dice

            if self.player_sum > 21:
                return StepResult(
                    state=None,
                    reward=-1.0,
                    done=True,
                    info={"reason": "player_bust", "player_sum": self.player_sum, "dealer_sum": self.dealer_sum},
                )

            return StepResult(
                state=(self.player_sum, self.dealer_start_sum),
                reward=0.0,
                done=False,
                info={"reason": "continue", "player_sum": self.player_sum},
            )

        # STAND: Dealer spielt zu Ende (2 dice per dealer hit)
        while self.dealer_sum < 17:
            self.dealer_sum += self._roll2()  # NEW: +2 dice

        if self.dealer_sum > 21:
            return StepResult(
                state=None,
                reward=1.0,
                done=True,
                info={"reason": "dealer_bust", "player_sum": self.player_sum, "dealer_sum": self.dealer_sum},
            )

        if self.player_sum > self.dealer_sum:
            r, reason = 1.0, "win"
        elif self.player_sum < self.dealer_sum:
            r, reason = -1.0, "loss"
        else:
            r, reason = 0.0, "draw"

        return StepResult(
            state=None,
            reward=r,
            done=True,
            info={"reason": reason, "player_sum": self.player_sum, "dealer_sum": self.dealer_sum},
        )
