import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from qlearning import TrainConfig, train_q_learning, evaluate_policy, policy_table, evaluate_simple_strategy
from dicejack_env import HIT, STAND

st.set_page_config(page_title="Dicejack RL (Q-Learning)", layout="wide")

st.title("Dicejack — Q-Learning in Streamlit")
st.caption("Variante: HIT = 2 Würfel (2d6) für Spieler + Dealer. State: (player_sum, dealer_start_sum).")


if "q" not in st.session_state:
    st.session_state.q = None
if "train_log" not in st.session_state:
    st.session_state.train_log = None




def _roll(rng: np.random.Generator) -> int:
    return int(rng.integers(1, 7))

def _roll2(rng: np.random.Generator) -> tuple[int, int]:
    return _roll(rng), _roll(rng)

def new_visual_game(seed: int, player_label: str = "you"):
    rng = np.random.default_rng(int(seed))
    p1, p2 = _roll2(rng)
    d1, d2 = _roll2(rng)

    g = {
        "rng": rng,
        "phase": "player",
        "player_rolls": [p1, p2],      # list of ints (from initial 2 dice)
        "dealer_rolls": [d1, d2],      # list of ints (from initial 2 dice)
        "player_sum": p1 + p2,
        "dealer_sum": d1 + d2,
        "dealer_start_sum": d1 + d2,
        "done": False,
        "reward": 0.0,
        "reason": None,
        "events": [],
    }

    g["events"].append({"who": "init", "action": "DEAL", "rolls": None, "player_sum": g["player_sum"], "dealer_sum": g["dealer_start_sum"]})

    # Naturals are basically impossible with 2d6 start, but keep it robust
    if g["player_sum"] == 21 and g["dealer_start_sum"] == 21:
        g["done"] = True; g["phase"] = "done"; g["reward"] = 0.0; g["reason"] = "both_natural"
    elif g["player_sum"] == 21:
        g["done"] = True; g["phase"] = "done"; g["reward"] = 1.5; g["reason"] = "player_natural"
    elif g["dealer_start_sum"] == 21:
        g["done"] = True; g["phase"] = "done"; g["reward"] = -1.0; g["reason"] = "dealer_natural"

    return g

def player_hit(g, who="you"):
    if g["done"] or g["phase"] != "player":
        return
    r1, r2 = _roll2(g["rng"])
    g["player_rolls"].extend([r1, r2])
    g["player_sum"] += (r1 + r2)
    g["events"].append({"who": who, "action": "HIT", "rolls": (r1, r2), "player_sum": g["player_sum"], "dealer_sum": g["dealer_start_sum"]})
    if g["player_sum"] > 21:
        g["done"] = True; g["phase"] = "done"; g["reward"] = -1.0; g["reason"] = "player_bust"

def player_stand(g, who="you"):
    if g["done"] or g["phase"] != "player":
        return
    g["events"].append({"who": who, "action": "STAND", "rolls": None, "player_sum": g["player_sum"], "dealer_sum": g["dealer_start_sum"]})
    g["phase"] = "dealer"

def resolve_after_dealer(g):
    if g["done"]:
        return
    ps, ds = g["player_sum"], g["dealer_sum"]
    if ds > 21:
        g["reward"], g["reason"] = 1.0, "dealer_bust"
    else:
        if ps > ds:
            g["reward"], g["reason"] = 1.0, "win"
        elif ps < ds:
            g["reward"], g["reason"] = -1.0, "loss"
        else:
            g["reward"], g["reason"] = 0.0, "draw"
    g["done"] = True
    g["phase"] = "done"
    g["events"].append({"who": "result", "action": g["reason"], "rolls": None, "player_sum": ps, "dealer_sum": ds})

def dealer_roll_once(g):
    if g["done"] or g["phase"] != "dealer":
        return

    if g["dealer_sum"] >= 17:
        resolve_after_dealer(g)
        return

    r1, r2 = _roll2(g["rng"])
    g["dealer_rolls"].extend([r1, r2])
    g["dealer_sum"] += (r1 + r2)
    g["events"].append({"who": "dealer", "action": "ROLL", "rolls": (r1, r2), "player_sum": g["player_sum"], "dealer_sum": g["dealer_sum"]})

    if g["dealer_sum"] > 21:
        g["done"] = True; g["phase"] = "done"; g["reward"] = 1.0; g["reason"] = "dealer_bust"
        g["events"].append({"who": "result", "action": "dealer_bust", "rolls": None, "player_sum": g["player_sum"], "dealer_sum": g["dealer_sum"]})
        return

    if g["dealer_sum"] >= 17:
        resolve_after_dealer(g)

def agent_choose_action(q_table, player_sum: int, dealer_start_sum: int) -> int:
    return int(np.argmax(q_table[(player_sum, dealer_start_sum)]))

def render_events(events, max_rows=60):
    tail = events[-max_rows:]
    lines = []
    for e in tail:
        who = e["who"]; action = e["action"]; rolls = e["rolls"]
        ps = e["player_sum"]; ds = e["dealer_sum"]
        if action == "DEAL":
            lines.append(f"- DEAL → player_sum={ps}, dealer_start_sum={ds}")
        elif action == "HIT":
            lines.append(f"- **{who}** HIT rolls={rolls} → player_sum={ps}")
        elif action == "ROLL":
            lines.append(f"- **dealer** ROLL rolls={rolls} → dealer_sum={ds}")
        elif action == "STAND":
            lines.append(f"- **{who}** STAND → player_sum={ps}")
        else:
            lines.append(f"- RESULT `{action}` → player_sum={ps}, dealer_sum={ds}")
    st.markdown("\n".join(lines))


# tab_train, tab_eval, tab_policy, tab_play = st.tabs(
#     ["Train", "Evaluate", "Policy Heatmap", "Play (step-by-step)"]
# )


tab_train, tab_eval, tab_baseline, tab_policy, tab_play = st.tabs(
    ["Train", "Evaluate", "Baseline Test", "Policy Heatmap", "Play (step-by-step)"]
)

# ----------------------------
# Train
# ----------------------------
with tab_train:
    st.subheader("Training")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        episodes = st.number_input("Episodes", 1000, 500_000, 50_000, step=1000)
    with c2:
        alpha = st.slider("α (learning rate)", 0.01, 0.50, 0.10, step=0.01)
    with c3:
        gamma = st.slider("γ (discount)", 0.0, 0.99, 0.95, step=0.01)
    with c4:
        eps_start = st.slider("ε start", 0.0, 1.0, 1.0, step=0.05)
    with c5:
        eps_end = st.slider("ε end", 0.0, 1.0, 0.01, step=0.01)

    c6, c7 = st.columns(2)
    with c6:
        eps_decay = st.number_input(
            "ε decay (multiplicative)",
            min_value=0.90,
            max_value=0.999999,
            value=0.9995,
            step=0.0001,
            format="%.6f",
        )
    with c7:
        seed = st.number_input("Seed", 0, 10_000, 42, step=1)

    if st.button("Train Q-Learning", type="primary"):
        cfg = TrainConfig(
            episodes=int(episodes),
            alpha=float(alpha),
            gamma=float(gamma),
            epsilon_start=float(eps_start),
            epsilon_end=float(eps_end),
            epsilon_decay=float(eps_decay),
            seed=int(seed),
        )
        out = train_q_learning(cfg)
        st.session_state.q = out["q"]
        st.session_state.train_log = out
        st.success(f"Done. final ε = {out['final_epsilon']:.4f}")

    if st.session_state.train_log is not None:
        log = st.session_state.train_log
        st.write("**Training curve (Win-Rate Moving Average)**")
        fig = plt.figure()
        plt.plot(log["win_rate_ma"])
        plt.xlabel("Episode")
        plt.ylabel("Win-Rate (MA)")
        st.pyplot(fig)

        q = st.session_state.q
        payload = {f"{k[0]}_{k[1]}": v.tolist() for k, v in q.items()}
        st.download_button(
            "Download Q-table (json)",
            data=json.dumps(payload, indent=2),
            file_name="q_table.json",
            mime="application/json",
        )

# ----------------------------
# Evaluate
# ----------------------------
with tab_eval:
    st.subheader("Evaluation (greedy policy, ε=0)")
    if st.session_state.q is None:
        st.info("Train zuerst im Tab 'Train'.")
    else:
        n_games = st.number_input("Games", 1000, 200_000, 20_000, step=1000)
        seed_eval = st.number_input("Eval Seed", 0, 10_000, 123, step=1)
        if st.button("📏 Evaluate"):
            metrics = evaluate_policy(st.session_state.q, n_games=int(n_games), seed=int(seed_eval))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win rate", f"{metrics['win_rate']*100:.2f}%")
            c2.metric("Loss rate", f"{metrics['loss_rate']*100:.2f}%")
            c3.metric("Draw rate", f"{metrics['draw_rate']*100:.2f}%")
            c4.metric("Natural rate", f"{metrics['natural_rate']*100:.2f}%")



# ----------------------------
# Baseline Test
# ----------------------------
with tab_baseline:
    st.subheader("Baseline Strategy Test")
    st.caption("Simple strategies: Hit until threshold, then STAND")
    
    n_games_baseline = st.number_input("Games", 1000, 100_000, 20_000, step=1000, key="baseline_games")
    seed_baseline = st.number_input("Seed", 0, 10_000, 456, step=1, key="baseline_seed")
    
    if st.button("Test All Baselines"):
        results = []
        for threshold in [14, 15, 16, 17, 18]:
            metrics = evaluate_simple_strategy(
                n_games=int(n_games_baseline), 
                seed=int(seed_baseline), 
                threshold=threshold
            )
            results.append({
                "Strategy": f"Hit until {threshold-1}",
                "Threshold": threshold,
                "Win Rate": f"{metrics['win_rate']*100:.2f}%",
                "Loss Rate": f"{metrics['loss_rate']*100:.2f}%",
                "Draw Rate": f"{metrics['draw_rate']*100:.2f}%"
            })
        
        st.table(results)
        
        st.divider()
        st.write("**Interpretation:**")
        st.write("- Wenn ALLE unter 45% sind → Spiel ist Dealer-favorisiert")
        st.write("- Die beste Baseline zeigt das theoretische Maximum")
        st.write("- Dein Agent sollte mindestens so gut sein wie die beste Baseline")

# ----------------------------
# Policy
# ----------------------------
with tab_policy:
    st.subheader("Policy / Confidence Heatmap (dealer_start_sum 2..12)")
    if st.session_state.q is None:
        st.info("Train zuerst im Tab 'Train'.")
    else:
        view = st.radio(
            "Ansicht",
            ["Policy (0=Stand, 1=Hit)", "Confidence (Q_hit - Q_stand)"],
            horizontal=True,
        )
        q = st.session_state.q

        if view.startswith("Policy"):
            arr = policy_table(q)  # (20,11)
            fig = plt.figure()
            plt.imshow(arr, aspect="auto")
            plt.yticks(range(0, 20), [str(ps) for ps in range(2, 22)])
            plt.xticks(range(0, 11), [str(ds) for ds in range(2, 13)])
            plt.xlabel("Dealer start sum (2..12)")
            plt.ylabel("Player sum (2..21)")
            plt.colorbar()
            st.pyplot(fig)
        else:
            arr = np.zeros((20, 11), dtype=float)
            for ps in range(2, 22):
                for ds in range(2, 13):
                    arr[ps - 2, ds - 2] = float(q[(ps, ds)][HIT] - q[(ps, ds)][STAND])

            fig = plt.figure()
            plt.imshow(arr, aspect="auto")
            plt.yticks(range(0, 20), [str(ps) for ps in range(2, 22)])
            plt.xticks(range(0, 11), [str(ds) for ds in range(2, 13)])
            plt.xlabel("Dealer start sum (2..12)")
            plt.ylabel("Player sum (2..21)")
            plt.colorbar()
            st.pyplot(fig)
            st.caption("Positive → HIT bevorzugt. Negative → STAND bevorzugt.")

# ----------------------------
# Play
# ----------------------------
with tab_play:
    st.subheader("Play (Step-by-step)")

    mode = st.radio(
        "Mode",
        [
            "Du vs Dealer (Dealer Schritt für Schritt)",
            "Agent Autoplay (alles Schritt für Schritt)",
            "Coach Mode (du spielst + Agent erklärt)",
        ],
        horizontal=True,
    )
    seed_play = st.number_input("Seed", 0, 10_000, 7, step=1, key="seed_play")

    if mode == "Du vs Dealer (Dealer Schritt für Schritt)":
        if "g_you" not in st.session_state or st.button("🎲 New game", key="new_you"):
            st.session_state.g_you = new_visual_game(seed=int(seed_play), player_label="you")
            st.rerun()

        g = st.session_state.g_you

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Du – Summe", g["player_sum"])
        c2.metric("Dealer – Startsumme", g["dealer_start_sum"])
        c3.metric("Phase", g["phase"])
        c4.metric("Dealer – aktuelle Summe", g["dealer_sum"] if g["phase"] in ("dealer", "done") else "—")
        c5.metric("Done", "✅" if g["done"] else "…")

        if not g["done"]:
            if g["phase"] == "player":
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("HIT (2 Würfel)", key="you_hit"):
                        player_hit(g, who="you")
                        st.rerun()
                with b2:
                    if st.button("STAND", key="you_stand"):
                        player_stand(g, who="you")
                        st.rerun()
            elif g["phase"] == "dealer":
                if st.button("Dealer würfelt (2 Würfel)", key="dealer_step_you"):
                    dealer_roll_once(g)
                    st.rerun()
        else:
            st.subheader("Ergebnis")
            st.write(f"Reason: `{g['reason']}` | Reward: {g['reward']} | Dealer sum: {g['dealer_sum']}")

        st.divider()
        st.write("**Würfel-Historie**")
        st.write(f"Du: {g['player_rolls']}  (sum={g['player_sum']})")
        st.write(f"Dealer: {g['dealer_rolls']}  (sum={g['dealer_sum']})")
        st.write("**Event Log**")
        render_events(g["events"])

    elif mode == "Agent Autoplay (alles Schritt für Schritt)":
        if st.session_state.q is None:
            st.info("Train zuerst im Tab 'Train'.")
        else:
            if "g_agent" not in st.session_state or st.button("New autoplay", key="new_agent"):
                st.session_state.g_agent = new_visual_game(seed=int(seed_play), player_label="agent")
                st.rerun()

            g = st.session_state.g_agent
            q = st.session_state.q

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Agent – Summe", g["player_sum"])
            c2.metric("Dealer – Startsumme", g["dealer_start_sum"])
            c3.metric("Phase", g["phase"])
            c4.metric("Dealer – aktuelle Summe", g["dealer_sum"] if g["phase"] in ("dealer", "done") else "—")
            c5.metric("Done", "done" if g["done"] else "…")

            if not g["done"]:
                if st.button("▶ Next Step", key="agent_next"):
                    if g["phase"] == "player":
                        a = agent_choose_action(q, g["player_sum"], g["dealer_start_sum"])
                        if a == HIT:
                            player_hit(g, who="agent")
                        else:
                            player_stand(g, who="agent")
                    else:
                        dealer_roll_once(g)
                    st.rerun()
            else:
                st.subheader("Ergebnis")
                st.write(f"Reason: `{g['reason']}` | Reward: {g['reward']} | Dealer sum: {g['dealer_sum']}")

            st.divider()
            st.write("**Würfel-Historie**")
            st.write(f"Agent: {g['player_rolls']}  (sum={g['player_sum']})")
            st.write(f"Dealer: {g['dealer_rolls']}  (sum={g['dealer_sum']})")
            st.write("**Event Log**")
            render_events(g["events"])

    else:  # Coach Mode
        if st.session_state.q is None:
            st.info("Train zuerst im Tab 'Train'.")
        else:
            if "g_coach" not in st.session_state or st.button("New game (coach)", key="new_coach"):
                st.session_state.g_coach = new_visual_game(seed=int(seed_play), player_label="you")
                st.rerun()

            g = st.session_state.g_coach
            q = st.session_state.q

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Du – Summe", g["player_sum"])
            c2.metric("Dealer – Startsumme", g["dealer_start_sum"])
            c3.metric("Phase", g["phase"])
            c4.metric("Dealer – aktuelle Summe", g["dealer_sum"] if g["phase"] in ("dealer", "done") else "—")
            c5.metric("Done", "done" if g["done"] else "…")

            if not g["done"] and g["phase"] == "player":
                a = agent_choose_action(q, g["player_sum"], g["dealer_start_sum"])
                q_hit = float(q[(g["player_sum"], g["dealer_start_sum"])][HIT])
                q_stand = float(q[(g["player_sum"], g["dealer_start_sum"])][STAND])
                st.info(f"Agent: **{'HIT' if a==HIT else 'STAND'}** | Q(hit)={q_hit:.3f} Q(stand)={q_stand:.3f} Δ={q_hit-q_stand:.3f}")

            if not g["done"]:
                if g["phase"] == "player":
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("HIT (2 Würfel)", key="coach_hit"):
                            player_hit(g, who="you")
                            st.rerun()
                    with b2:
                        if st.button("STAND", key="coach_stand"):
                            player_stand(g, who="you")
                            st.rerun()
                else:
                    if st.button("Dealer würfelt (2 Würfel)", key="dealer_step_coach"):
                        dealer_roll_once(g)
                        st.rerun()
            else:
                st.subheader("Ergebnis")
                st.write(f"Reason: `{g['reason']}` | Reward: {g['reward']} | Dealer sum: {g['dealer_sum']}")

            st.divider()
            st.write("**Würfel-Historie**")
            st.write(f"Du: {g['player_rolls']}  (sum={g['player_sum']})")
            st.write(f"Dealer: {g['dealer_rolls']}  (sum={g['dealer_sum']})")
            st.write("**Event Log**")
            render_events(g["events"])
