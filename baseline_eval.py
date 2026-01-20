import numpy as np
from dicejack_env import DicejackEnv, HIT, STAND

def eval_fixed(policy_fn, n=20000, seed=0):
    env = DicejackEnv(seed=seed)
    wins=losses=draws=0
    for _ in range(n):
        s = env.reset()
        term = env.check_terminal_after_reset()
        if term is not None:
            r = term.reward
        else:
            done = False
            while not done:
                a = policy_fn(s)
                res = env.step(a)
                done = res.done
                if not done:
                    s = res.state
            r = res.reward
        if r > 0: wins += 1
        elif r < 0: losses += 1
        else: draws += 1
    return wins/n, losses/n, draws/n

def always_stand(_s): return STAND
def always_hit(_s): return HIT
def hit_to_17(s): 
    ps, _ds = s
    return HIT if ps < 17 else STAND

if __name__ == "__main__":
    for name, pi in [("stand",always_stand),("hit",always_hit),("hit_to_17",hit_to_17)]:
        w,l,d = eval_fixed(pi, n=20000, seed=1)
        print(name, w,l,d)
