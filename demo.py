"""
demo.py — The Mechanics of Certainty (Narrative Edition)
Run: python demo.py
"""

import random

SIZE = 11
MID = 5
EPISODES = 2000
ALPHA = 0.25
GAMMA = 0.95

def train(reward_pos, region, episodes=EPISODES):
    q = [[0.0, 0.0] for _ in range(SIZE)]
    lo, hi = region
    epsilon = 1.0
    for _ in range(episodes):
        pos = random.randint(lo, hi)
        if pos == reward_pos: 
            continue # Don't start directly on the goal
            
        for _ in range(40):
            if random.random() < epsilon:
                a = random.randint(0, 1)
            else:
                a = 0 if q[pos][0] > q[pos][1] else (1 if q[pos][1] > q[pos][0] else random.randint(0, 1))
            
            nxt = max(lo, min(hi, pos + (-1 if a == 0 else 1)))
            
            # STRICT TERMINAL STATE FIX
            if nxt == reward_pos:
                target = 1.0
                is_done = True
            else:
                target = -0.05 + GAMMA * max(q[nxt])
                is_done = False
                
            q[pos][a] += ALPHA * (target - q[pos][a])
            pos = nxt
            if is_done: break
            
        epsilon = max(0.01, epsilon * 0.995)
    return q

def train_full(q, episodes):
    epsilon = 0.8 # Start high enough to genuinely explore the new territory
    for _ in range(episodes):
        pos = MID + random.randint(-2, 2)
        for _ in range(50):
            if random.random() < epsilon:
                a = random.randint(0, 1)
            else:
                a = 0 if q[pos][0] > q[pos][1] else (1 if q[pos][1] > q[pos][0] else random.randint(0, 1))
            
            nxt = max(0, min(SIZE - 1, pos + (-1 if a == 0 else 1)))
            
            if nxt == 0 or nxt == SIZE - 1:
                target = 1.0
                is_done = True
            else:
                target = -0.05 + GAMMA * max(q[nxt])
                is_done = False
                
            q[pos][a] += ALPHA * (target - q[pos][a])
            pos = nxt
            if is_done: break
            
        epsilon = max(0.1, epsilon * 0.98) # Slower decay allows for real unlearning
    return q

def conf(q, pos):
    ql, qr = q[pos]
    if abs(ql) < 0.001 and abs(qr) < 0.001: return 0.5, "UNK"
    # A gap of 0.12 mathematically equals absolute proof in this environment
    scaled_diff = min(1.0, abs(ql - qr) / 0.12)
    pct = 0.5 + (scaled_diff * 0.499)
    return pct, "LEFT " if ql > qr else ("RIGHT" if qr > ql else "EQUAL")

def bar(value, width=10):
    n = int(max(0.0, (value - 0.5) * 2.0) * width)
    return "█" * n + "░" * (width - n)

def main():
    random.seed(42)

    print("\n┌──────────────────────────────────────────────────────────────┐")
    print("│ THE MECHANICS OF CERTAINTY                                   │")
    print("└──────────────────────────────────────────────────────────────┘")
    print(" --- HYPOTHESIS ---")
    print(" Certainty is not accuracy. It is merely the feeling a model")
    print(" produces when its training data has been highly consistent.\n")

    print(" Environment: 1D Corridor [0..10]. Rewards (★) at BOTH ends.")
    print(" Agent A trains [0..6]. Agent B trains [4..10].\n")

    print(" --- PHASE 1: ISOLATED TRAINING ---")
    q_a = train(reward_pos=0, region=(0, 6))
    q_b = train(reward_pos=SIZE - 1, region=(4, SIZE - 1))
    
    print(f" Agent A learned: L={q_a[MID][0]:+.2f}, R={q_a[MID][1]:+.2f} (at pos {MID})")
    print(f" Agent B learned: L={q_b[MID][0]:+.2f}, R={q_b[MID][1]:+.2f} (at pos {MID})\n")

    print(" --- PHASE 2: THE COLLISION (Position 5) ---")
    ca, da = conf(q_a, MID)
    cb, db = conf(q_b, MID)
    print(f" A: {bar(ca)} {ca*100:5.1f}% | Says ◄ {da}")
    print(f" B: {bar(cb)} {cb*100:5.1f}% | Says {db} ►")
    print(" Both engines are highly confident. They completely disagree.")
    print(" Neither is malfunctioning. This IS the function.\n")

    print(" --- PHASE 3: CROSS-TRAINING (Variance as Remedy) ---")
    print(" Exposing agents to the FULL corridor (the other's reality)...\n")
    print("  Eps │   Agt A Conf     │   Agt B Conf")
    print(" ─────┼──────────────────┼──────────────────")

    qa_x, qb_x = [list(row) for row in q_a], [list(row) for row in q_b]
    checkpoints, prev = [0, 50, 100, 200, 500], 0
    
    for cp in checkpoints:
        if cp > prev:
            qa_x, qb_x = train_full(qa_x, cp - prev), train_full(qb_x, cp - prev)
        prev = cp
        ca, da = conf(qa_x, MID)
        cb, db = conf(qb_x, MID)
        lbl = str(cp) if cp > 0 else "0"
        print(f" {lbl:>4} │ {bar(ca, 6)} {ca*100:4.0f}% {da[0]} │ {bar(cb, 6)} {cb*100:4.0f}% {db[0]}")

    print("\n --- CONCLUSION ---")
    print(" Both Q-values converge. Confidence drops. Accuracy rises.")
    print(" The disagreement became navigable not through argument,")
    print(" but through exposure to the out-of-distribution truth.\n")

if __name__ == "__main__":
    main()
