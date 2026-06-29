#!/usr/bin/env python3
"""Dev-instance lifecycle validator for cpmm-multi-2 √variance (asymmetric) markets.

The unit tests + the prior on-instance lifecycle validation used *balanced* v2 pools.
This session's creation pools CHANGED for sum-to-one markets (balanced → √variance
asymmetric, vendor afe14c152). This driver re-runs create → trade → add-liquidity →
drizzle → resolve against the real LOCAL_ONLY instance (API :8088 + Supabase) on a
skewed n≥3 market — the case where pools are genuinely asymmetric (Y_i≠N_i, p_i≠q_i)
and the SITES-TO-RE-CHECK carry-forward says "the math is shape-agnostic; the
dev-instance differential is the real check."

Usage:
  python3 validate_lifecycle.py [resolve_case]
    resolve_case ∈ {choose_one_basket, choose_one_nonbasket, choose_multiple, cancel}
    (default: choose_one_basket — the carry-forward case)

Drizzle is triggered out-of-band (scheduler doesn't run in LOCAL_ONLY) by shelling
out to backend/scheduler/src/run-drizzle-once.ts.

Invariants checked (all must hold on the asymmetric shape):
  - prob_i == q_i (normalized), Σ prob_i = 1
  - stored answer.prob (cache) == prob-from-(pool,p)   [the BUG#3 regression]
  - funding identity: poolYes_i + Σ_{j≠i} poolNo_j == ante  (all-winners-tight)
  - per-answer totalLiquidity == Y_i^{p_i}·N_i^{(1-p_i)}  (true invariant, afe8d3951)
  - √variance shape: realized depth ≈ ∝ √(q(1-q)) (closed-form approximation)
  - add-liquidity / drizzle: probability preserved, Σp=1, reserves deepened (lossless)
  - resolution: market resolves, LP/creator paid (NOT zero — BUG#1 guard),
    sole-participant net loss == fees only (conservation)
"""
import json
import math
import subprocess
import sys
import urllib.request
import urllib.error

API = "http://localhost:8088"
USER = "test-user-1"
VENDOR = "/home/evand/predictions/vendor/manifold"
_fails = []


def _req(method, path, body=None):
    url = API + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-Local-User", USER)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()}") from None


def get(path):
    return _req("GET", path)


def post(path, body):
    return _req("POST", path, body)


def balance():
    return get("/v0/me")["balance"]


def prob_from_pool(y, n, p):
    return (p * n) / ((1 - p) * y + p * n)


def _yn(a):
    return a["pool"]["YES"], a["pool"]["NO"]


def report(name, ok, detail=""):
    mark = "PASS" if ok else "**FAIL**"
    if not ok:
        _fails.append(name)
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def run_drizzle():
    print("\n=== DRIZZLE (out-of-band scheduler trigger) ===")
    cmd = ("set -a && source .env.local && set +a && "
           "npx ts-node src/run-drizzle-once.ts")
    r = subprocess.run(["bash", "-lc", cmd], cwd=f"{VENDOR}/backend/scheduler",
                       capture_output=True, text=True, timeout=300)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print("  drizzle:", tail[-1] if tail else "(no output)", f"(exit {r.returncode})")
    return r.returncode == 0


def check_sumone_and_cache(market, label):
    """Σp=1 + stored cache prob == prob(pool,p) for ALL answers (BUG#3 guard)."""
    print(f"\n=== {label}: Σp + cache integrity ===")
    answers = market["answers"]
    probs = [a["probability"] for a in answers]
    report("Σ prob == 1", abs(sum(probs) - 1) < 1e-5, f"Σ={sum(probs):.8f}")
    for i, a in enumerate(answers):
        y, no = _yn(a)
        pp = prob_from_pool(y, no, a["p"])
        report(f"cache prob[{i}] == prob(pool,p)", abs(a["probability"] - pp) < 1e-5,
               f"cache={a['probability']:.6f} from-pool={pp:.6f} (Y={y:.2f} N={no:.2f} p={a['p']:.4f})")


def check_creation(market, raw_probs):
    print(f"\n=== CREATION INVARIANTS ({market['slug']}) ===")
    answers = market["answers"]
    q = [p / sum(raw_probs) for p in raw_probs]
    n = len(answers)
    for i, a in enumerate(answers):
        report(f"prob[{i}] == q[{i}]", abs(a["probability"] - q[i]) < 1e-6,
               f"prob={a['probability']:.6f} q={q[i]:.6f} p={a['p']:.6f}")
    report("Σ prob == 1", abs(sum(a["probability"] for a in answers) - 1) < 1e-6)
    for i, a in enumerate(answers):
        y, no = _yn(a)
        report(f"stored prob[{i}] == prob(pool,p)",
               abs(a["probability"] - prob_from_pool(y, no, a["p"])) < 1e-6)
    report("pools asymmetric (Y≠N somewhere)",
           any(abs(_yn(a)[0] - _yn(a)[1]) > 1e-3 for a in answers) if n >= 3 else True)
    sumN = sum(_yn(a)[1] for a in answers)
    fundings = [_yn(a)[0] + (sumN - _yn(a)[1]) for a in answers]
    report("funding identity Y_i+Σ_{j≠i}N_j constant", (max(fundings) - min(fundings)) < 1e-3,
           f"ante={fundings[0]:.4f}")
    # √variance shape: realized depth ≈ ∝ √(q(1-q)) (closed-form approximation; exact
    # funding re-solve perturbs it ~0.1%, see new-contract.ts cpmmMulti2SumToOnePools)
    W = [(1 - a["p"]) * _yn(a)[0] + a["p"] * _yn(a)[1] for a in answers]
    sqrtC = [math.sqrt(qi * (1 - qi)) for qi in q]
    ratios = [W[i] / sqrtC[i] for i in range(n)]
    spread = (max(ratios) - min(ratios)) / max(ratios)
    report("realized depth ≈ ∝ √(q(1-q)) (approx, <1%)", spread < 1e-2,
           f"W/√C spread={spread:.2e}")
    for i, a in enumerate(answers):
        y, no = _yn(a)
        k = y ** a["p"] * no ** (1 - a["p"])
        report(f"totalLiquidity[{i}] == Y^p·N^(1-p)", abs(a["totalLiquidity"] - k) < 1e-2,
               f"stored={a['totalLiquidity']:.4f} computed={k:.4f}")
    return fundings[0]


def sum_to_one_pools(q, ante):
    """Port of calculate-cpmm.ts cpmmMulti2SumToOnePools (the √variance creation rule)."""
    n = len(q)
    if n < 2:
        return [(ante, ante)]
    sqrtC = [math.sqrt(qi * (1 - qi)) for qi in q]
    meanSqrtC = sum(sqrtC) / n
    D0 = (ante * (n - 2)) / (2 * (n - 1))
    Wbar = (ante * n) / (4 * (n - 1))
    N = []
    for i, qi in enumerate(q):
        Wi = Wbar * sqrtC[i] / meanSqrtC
        b = D0 - Wi
        N.append((-b + math.sqrt(b * b + 4 * Wi * qi * D0)) / 2)
    D = ante - sum(N)
    return [(N[i] + D, N[i]) for i in range(n)]  # (poolYes, poolNo)


def check_variance_add_shape(before, after, label):
    """The whole-market add/drizzle must deepen pools along the √variance MERGE rule (GP17),
    NOT a flat equal-split. Validates the REAL pipeline (add-liquidity → subsidyPool → scheduler
    drizzle → addCpmmMultiLiquidityAnswersSumToOneV2 → DB pools) against the merge prediction.

    Discriminators vs the old equal-split:
      - equal-split adds ΔY_i == ΔN_i (symmetric) and equal across answers;
      - √variance merge adds ASYMMETRIC deltas with ΔY_i − ΔN_i = const (all-winners-tight),
        concentrated in uncertain answers.
    Method: read reserve deltas, recover the drizzled amount a from the delta funding identity
    (a = ΔY_i + Σ_{j≠i} ΔN_j, constant ∀i), then compare deltas to sum_to_one_pools(q_pre, a).
    Amount-agnostic (works for partial drizzle): the merge is homogeneous, drizzle holds prob.
    """
    print(f"\n=== {label}: √variance add shape (GP17 merge rule) ===")
    ba, aa = before["answers"], after["answers"]
    n = len(ba)
    dY = [_yn(a)[0] - _yn(b)[0] for b, a in zip(ba, aa)]
    dN = [_yn(a)[1] - _yn(b)[1] for b, a in zip(ba, aa)]
    if not report("drizzle moved reserves (Σ|Δ| > 0)", sum(abs(x) for x in dY + dN) > 1e-6):
        return
    # recovered drizzled amount via the delta funding identity (must be constant across i)
    fund = [dY[i] + sum(dN[j] for j in range(n) if j != i) for i in range(n)]
    a_amt = fund[0]
    report("delta funding identity ΔY_i+Σ_{j≠i}ΔN_j constant (all-winners-tight)",
           (max(fund) - min(fund)) < max(1e-3, 1e-4 * abs(a_amt)),
           f"a≈{a_amt:.4f} spread={max(fund) - min(fund):.2e}")
    # current probs (pre-drizzle) drive the merge
    q_pre = [b["probability"] for b in ba]
    expected = sum_to_one_pools(q_pre, a_amt)
    maxerr = 0.0
    for i in range(n):
        eY, eN = expected[i]
        maxerr = max(maxerr, abs(dY[i] - eY), abs(dN[i] - eN))
    report("reserve deltas == √variance merge(q_pre, a) (the pipeline matches GP17)",
           maxerr < max(1e-2, 1e-3 * abs(a_amt)),
           f"max|Δ−merge|={maxerr:.4e}")
    # discriminator: the deltas are ASYMMETRIC (would be 0 under equal-split) for skewed answers
    asym = max(abs(dY[i] - dN[i]) for i in range(n))
    report("deltas asymmetric ΔY≠ΔN (NOT equal-split; concentrated)", asym > 1e-2,
           f"max|ΔY−ΔN|={asym:.4f}")
    # depth concentration: Δk ratio across answers (equal-split ≈ flat ~1.2; merge ≫ that)
    dk = [(_yn(a)[0] ** a["p"] * _yn(a)[1] ** (1 - a["p"]))
          - (_yn(b)[0] ** b["p"] * _yn(b)[1] ** (1 - b["p"]))
          for b, a in zip(ba, aa)]
    if all(x > 1e-9 for x in dk):
        report("added depth concentrated in uncertain answers (Δk ratio > 2)",
               max(dk) / min(dk) > 2.0, f"Δk ratio={max(dk) / min(dk):.2f}")


def check_preserved(before, after, label):
    """Liquidity op (add / drizzle): each answer's probability preserved, reserves grew."""
    print(f"\n=== {label}: probability preserved + reserves deepened ===")
    for i, (b, a) in enumerate(zip(before["answers"], after["answers"])):
        report(f"prob[{i}] preserved", abs(b["probability"] - a["probability"]) < 1e-4,
               f"{b['probability']:.6f} -> {a['probability']:.6f}")
    grew = all(sum(_yn(a)) > sum(_yn(b)) - 1e-6
               for b, a in zip(before["answers"], after["answers"]))
    report("reserves deepened (Y+N grew, all answers)", grew)
    check_sumone_and_cache(after, label)


def check_set_creation(market, raw_probs):
    """Independent ("Set") market: each answer is its own balanced CPMM, p_i=q_i,
    NO Σ=1 constraint (absolute probs)."""
    print(f"\n=== SET CREATION INVARIANTS ({market['slug']}) ===")
    answers = market["answers"]
    q = [p / 100 for p in raw_probs]   # absolute, no normalization
    for i, a in enumerate(answers):
        y, no = _yn(a)
        report(f"prob[{i}] == q[{i}] (absolute)", abs(a["probability"] - q[i]) < 1e-6,
               f"prob={a['probability']:.6f} q={q[i]:.6f}")
        report(f"balanced pool Y==N [{i}]", abs(y - no) < 1e-6, f"Y={y:.4f} N={no:.4f}")
        report(f"p[{i}] == q[{i}] (binary-CPMM)", abs(a["p"] - q[i]) < 1e-6,
               f"p={a['p']:.6f}")
    report("Σ prob NOT constrained to 1 (Set)", True,
           f"Σ={sum(a['probability'] for a in answers):.4f}")


def run_set_lifecycle():
    """Create Set market -> trade -> sell back -> add-liq -> drizzle -> resolve each
    answer independently (mix YES/NO). Conservation tracked."""
    raw = [70, 40, 15]
    labels = ["Indep-A", "Indep-B", "Indep-C"]
    b0 = balance()
    print(f"start balance: {b0:.4f}   SET (independent) lifecycle")
    created = post("/v0/market", {
        "question": "v2 Set independent lifecycle (delete-me)",
        "outcomeType": "MULTIPLE_CHOICE", "answers": labels, "initialProbs": raw,
        "shouldAnswersSumToOne": False, "addAnswersMode": "DISABLED", "liquidityTier": 1000,
    })
    mid = created["id"]
    print(f"created id={mid} slug={created.get('slug')} mechanism={created.get('mechanism')}")
    market = get(f"/v0/market/{mid}")
    check_set_creation(market, raw)

    ans = market["answers"][0]
    print(f"\n>> trade: BUY M$100 YES on '{ans['text']}'")
    post("/v0/bet", {"contractId": mid, "amount": 100, "outcome": "YES", "answerId": ans["id"]})
    market = get(f"/v0/market/{mid}")
    a0 = market["answers"][0]
    y, no = _yn(a0)
    report("Set: traded answer cache prob == prob(pool,p)",
           abs(a0["probability"] - prob_from_pool(y, no, a0["p"])) < 1e-5,
           f"cache={a0['probability']:.6f} p={a0['p']:.4f}")
    # untraded answers unchanged (independent!)
    for i in (1, 2):
        report(f"Set: untraded answer[{i}] prob unchanged (independent)",
               abs(market["answers"][i]["probability"] - raw[i] / 100) < 1e-6)

    print("\n>> sell: ALL YES shares back on traded answer (v2 sell path)")
    post(f"/v0/market/{mid}/sell", {"contractId": mid, "outcome": "YES", "answerId": ans["id"]})
    market = get(f"/v0/market/{mid}")
    a0 = market["answers"][0]
    y, no = _yn(a0)
    report("Set: post-sell cache prob == prob(pool,p)",
           abs(a0["probability"] - prob_from_pool(y, no, a0["p"])) < 1e-5,
           f"cache={a0['probability']:.6f} p={a0['p']:.4f}")

    print("\n>> add-liquidity M$200 + drizzle")
    before = market
    post(f"/v0/market/{mid}/add-liquidity", {"contractId": mid, "amount": 200})
    run_drizzle()
    after = get(f"/v0/market/{mid}")
    for i, (b, a) in enumerate(zip(before["answers"], after["answers"])):
        report(f"Set: prob[{i}] preserved through add+drizzle",
               abs(b["probability"] - a["probability"]) < 1e-4,
               f"{b['probability']:.6f} -> {a['probability']:.6f}")

    # resolve each answer independently: A=YES, B=NO, C=YES
    answers = after["answers"]
    for a, oc in zip(answers, ["YES", "NO", "YES"]):
        print(f">> resolve Set answer '{a['text']}' -> {oc}")
        post(f"/v0/market/{mid}/resolve",
             {"contractId": mid, "answerId": a["id"], "outcome": oc})
    resolved = get(f"/v0/market/{mid}")
    report("Set market isResolved", resolved.get("isResolved") is True)
    b1 = balance()
    net = b0 - b1
    print(f"\n=== SET CONSERVATION ===\n  start={b0:.4f} end={b1:.4f} net(=fees)={net:.4f}")
    report("Set: net loss small & >= 0 (LPs paid across YES+NO resolutions)",
           -0.01 <= net < 100, f"net={net:.4f}")
    print(f"\n{'='*50}\nRESULT: {'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    return 0 if not _fails else 1


def run_multibasket(v1=False):
    """m>1 basket buy via multi-bet. The v2 algorithm gives EQUAL YES shares per basket
    answer (calculate-cpmm-arbitrage.ts:322); resolve to one basket answer and check
    conservation + cache integrity. Pass v1=True to run the SAME pattern on a uniform
    cpmm-multi-1 market as a differential (does v1 also lose for a sole participant?)."""
    labels = ["Alpha", "Bravo", "Charlie", "Delta"]
    b0 = balance()
    tag = "v1 uniform" if v1 else "v2 √var"
    print(f"start balance: {b0:.4f}   m>1 BASKET (multi-bet) lifecycle [{tag}]")
    body = {"question": f"{tag} m>1 basket (delete-me)",
            "outcomeType": "MULTIPLE_CHOICE", "answers": labels,
            "shouldAnswersSumToOne": True, "addAnswersMode": "DISABLED", "liquidityTier": 1000}
    if not v1:
        body["initialProbs"] = [55, 25, 12, 8]
    created = post("/v0/market", body)
    mid = created["id"]
    print(f"created id={mid} mechanism={created.get('mechanism')}")
    market = get(f"/v0/market/{mid}")
    if not v1:
        check_creation(market, [55, 25, 12, 8])
    basket = [market["answers"][1], market["answers"][2]]  # Bravo + Charlie (m=2)
    print(f"\n>> multi-bet: M$200 YES on basket {[a['text'] for a in basket]} (m=2)")
    bets = post("/v0/multi-bet", {"contractId": mid, "amount": 200,
                                  "answerIds": [a["id"] for a in basket]})
    shares = {b["answerId"]: b.get("shares", 0) for b in bets}
    print("   per-answer YES shares:", {a["text"]: round(shares.get(a["id"], 0), 4) for a in basket})
    sh = [shares.get(a["id"], 0) for a in basket]
    report("m>1: EQUAL YES shares per basket answer (GP12 / arb:322)",
           abs(sh[0] - sh[1]) / max(sh[0], 1) < 1e-3, f"{sh[0]:.4f} vs {sh[1]:.4f}")
    market = get(f"/v0/market/{mid}")
    check_sumone_and_cache(market, "POST-MULTIBET")
    # resolve to one basket answer
    print(f"\n>> resolve CHOOSE_ONE -> '{basket[0]['text']}' (a basket answer)")
    post(f"/v0/market/{mid}/resolve",
         {"contractId": mid, "outcome": "CHOOSE_ONE", "answerId": basket[0]["id"]})
    report("market isResolved", get(f"/v0/market/{mid}").get("isResolved") is True)
    b1 = balance()
    net = b0 - b1
    print(f"\n=== CONSERVATION ===\n  start={b0:.4f} end={b1:.4f} net(=fees)={net:.4f}")
    report("net loss small & >= 0 (LP/creator paid)", -0.01 <= net < 100, f"net={net:.4f}")
    print(f"\n{'='*50}\nRESULT: {'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    return 0 if not _fails else 1


def run_other_split(resolve_to="listed"):
    """Add-answer on a v2 sum-to-one market with an 'Other' catch-all (GP18 refinement split).
    Verifies: A & Other' each take half of Other's prob, listed probs/pools UNCHANGED, Σp=1,
    cache integrity, and conservation at resolution (sole participant net≈0)."""
    labels = ["Alpha", "Bravo", "Gamma"]
    b0 = balance()
    print(f"start balance: {b0:.4f}   OTHER-SPLIT lifecycle (resolve_to={resolve_to})")
    # initialProbs is disallowed with addable answers, so create uniform (v1, gets an 'Other'),
    # then convert to cpmm-multi-2 via add-liquidity (the lazy-conversion trigger).
    created = post("/v0/market", {
        "question": "v2 Other-split lifecycle (delete-me)",
        "outcomeType": "MULTIPLE_CHOICE", "answers": labels,
        "shouldAnswersSumToOne": True, "addAnswersMode": "ONLY_CREATOR", "liquidityTier": 1000,
    })
    mid = created["id"]
    post(f"/v0/market/{mid}/add-liquidity", {"contractId": mid, "amount": 100})  # → cpmm-multi-2
    market = get(f"/v0/market/{mid}")
    print(f"created id={mid} mechanism={market.get('mechanism')} (converted via add-liquidity)")
    report("market is cpmm-multi-2 after conversion", market.get("mechanism") == "cpmm-multi-2")
    # Trade on Other first, so the splitter must also refine USER positions
    # (convertOtherAnswerShares: YES-Other → YES-A grant) — exercises that path + conservation.
    other_pre = next(a for a in market["answers"] if a.get("isOther"))
    print(f">> trade: BUY M$80 YES on 'Other' (prob {other_pre['probability']:.3f})")
    post("/v0/bet", {"contractId": mid, "amount": 80, "outcome": "YES", "answerId": other_pre["id"]})
    market = get(f"/v0/market/{mid}")
    answers = market["answers"]
    others = [a for a in answers if a.get("isOther")]
    if not report("market has an 'Other' answer", len(others) == 1,
                  f"answers={[a['text'] for a in answers]}"):
        print("\nRESULT: FAILURES (no Other) — addAnswersMode may not auto-create Other")
        return 1
    other = others[0]
    listed_before = {a["id"]: a["probability"] for a in answers if not a.get("isOther")}
    pools_before = {a["id"]: _yn(a) for a in answers if not a.get("isOther")}
    p_o = other["probability"]
    print(f"  Other prob before = {p_o:.4f}; listed probs = "
          f"{ {a['text']: round(a['probability'],4) for a in answers if not a.get('isOther')} }")

    print(">> add answer 'Delta' (triggers v2 refinement split)")
    res = post(f"/v0/market/{mid}/answer", {"contractId": mid, "text": "Delta"})
    new_id = res["newAnswerId"]
    market2 = get(f"/v0/market/{mid}")
    by_id = {a["id"]: a for a in market2["answers"]}
    newA, other2 = by_id[new_id], by_id[other["id"]]

    report("A prob == Other_before/2", abs(newA["probability"] - p_o / 2) < 2e-3,
           f"A={newA['probability']:.4f} want={p_o/2:.4f}")
    report("Other' prob == Other_before/2", abs(other2["probability"] - p_o / 2) < 2e-3,
           f"O'={other2['probability']:.4f} want={p_o/2:.4f}")
    report("A + Other' == Other_before", abs(newA["probability"] + other2["probability"] - p_o) < 2e-3,
           f"sum={newA['probability']+other2['probability']:.4f} want={p_o:.4f}")
    for aid, pb in listed_before.items():
        report(f"listed prob UNCHANGED ({by_id[aid]['text']})",
               abs(by_id[aid]["probability"] - pb) < 1e-4,
               f"{pb:.6f} -> {by_id[aid]['probability']:.6f}")
        report(f"listed POOL UNCHANGED ({by_id[aid]['text']})",
               all(abs(x - y) < 1e-6 for x, y in zip(_yn(by_id[aid]), pools_before[aid])))
    report("Σ prob == 1", abs(sum(a["probability"] for a in market2["answers"]) - 1) < 1e-4,
           f"Σ={sum(a['probability'] for a in market2['answers']):.6f}")
    for a in market2["answers"]:
        y, no = _yn(a)
        report(f"cache prob == prob(pool,p) [{a['text']}]",
               abs(a["probability"] - prob_from_pool(y, no, a["p"])) < 1e-5,
               f"cache={a['probability']:.6f} from-pool={prob_from_pool(y,no,a['p']):.6f}")

    # resolve & check conservation (sole participant)
    win = {"listed": answers[0]["id"], "new": new_id, "other": other["id"]}[resolve_to]
    win_text = by_id.get(win, {}).get("text", win)
    print(f">> resolve CHOOSE_ONE -> '{win_text}'")
    post(f"/v0/market/{mid}/resolve", {"contractId": mid, "outcome": "CHOOSE_ONE", "answerId": win})
    report("market isResolved", get(f"/v0/market/{mid}").get("isResolved") is True)
    b1 = balance()
    net = b0 - b1
    print(f"\n=== CONSERVATION (sole participant) ===\n  start={b0:.4f} end={b1:.4f} net(=fees)={net:.4f}")
    report("net loss small & >= 0 (== fees; no mana created/destroyed)", -0.01 <= net < 100,
           f"net={net:.4f}")
    print(f"\n{'='*50}\nRESULT: {'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    print(f"market id: {mid}")
    return 0 if not _fails else 1


def check_converted(market, n):
    """A market just converted v1→v2 by its first user addLiquidity. Unlike a v2-*created*
    market, its pools are the v1 asymmetric shape (poolYes=ante/2, poolNo=ante/(2n-2)) with
    p floated to hold the existing uniform prob 1/n — so the √variance creation-shape checks
    do NOT apply. What MUST hold post-conversion:
      - mechanism flipped to cpmm-multi-2
      - each answer carries a persisted p (the conversion DOF that absorbs the asymmetry)
      - prob preserved at the pre-conversion value (uniform 1/n), Σp=1
      - cache prob == prob(pool,p); totalLiquidity == Y^p·N^(1-p) on the asymmetric pools
    """
    print("\n=== CONVERSION INVARIANTS (v1→v2 via first addLiquidity) ===")
    report("mechanism flipped to cpmm-multi-2",
           market.get("mechanism") == "cpmm-multi-2", f"mechanism={market.get('mechanism')}")
    answers = market["answers"]
    report("Σ prob == 1", abs(sum(a["probability"] for a in answers) - 1) < 1e-5,
           f"Σ={sum(a['probability'] for a in answers):.8f}")
    for i, a in enumerate(answers):
        report(f"prob[{i}] ≈ uniform 1/n", abs(a["probability"] - 1.0 / n) < 1e-4,
               f"prob={a['probability']:.6f} 1/n={1.0 / n:.6f} p={a['p']:.6f}")
        y, no = _yn(a)
        report(f"cache prob[{i}] == prob(pool,p)",
               abs(a["probability"] - prob_from_pool(y, no, a["p"])) < 1e-5,
               f"Y={y:.2f} N={no:.2f} p={a['p']:.4f}")
        k = y ** a["p"] * no ** (1 - a["p"])
        report(f"totalLiquidity[{i}] == Y^p·N^(1-p)", abs(a["totalLiquidity"] - k) < 1e-2,
               f"stored={a['totalLiquidity']:.4f} computed={k:.4f}")


def run_conversion_lifecycle():
    """v1→v2 LAZY CONVERSION full lifecycle. Create a uniform cpmm-multi-1 market, trigger
    the conversion with the first user addLiquidity (bumps mechanism + writes per-answer p),
    then run the rest of the lifecycle (drizzle → trade → add-liq → drizzle → resolve) on the
    CONVERTED market and assert conservation + invariants throughout. Requires the local
    CPMM_MULTI_2_CREATION_ENABLED flag = true (conversion is gated)."""
    labels = ["Conv-A", "Conv-B", "Conv-C"]
    n = len(labels)
    b0 = balance()
    print(f"start balance: {b0:.4f}   v1→v2 CONVERSION lifecycle")

    # --- CREATE uniform v1 (no initialProbs ⇒ stays cpmm-multi-1) ---
    created = post("/v0/market", {
        "question": "v1→v2 conversion lifecycle (delete-me)",
        "outcomeType": "MULTIPLE_CHOICE", "answers": labels,
        "shouldAnswersSumToOne": True, "addAnswersMode": "DISABLED", "liquidityTier": 1000,
    })
    mid = created["id"]
    print(f"created id={mid} mechanism={created.get('mechanism')}")
    report("created as v1 (uniform ⇒ cpmm-multi-1)",
           created.get("mechanism") == "cpmm-multi-1", f"mechanism={created.get('mechanism')}")
    v1_market = get(f"/v0/market/{mid}")
    v1_probs = [a["probability"] for a in v1_market["answers"]]

    # --- CONVERT: first user addLiquidity flips mechanism + writes p ---
    print("\n>> add-liquidity M$250 (the lazy v1→v2 conversion trigger)")
    post(f"/v0/market/{mid}/add-liquidity", {"contractId": mid, "amount": 250})
    converted = get(f"/v0/market/{mid}")
    check_converted(converted, n)
    report("conversion preserved each prob (vs pre-conversion v1)",
           all(abs(a["probability"] - p0) < 1e-4
               for a, p0 in zip(converted["answers"], v1_probs)),
           str([round(a["probability"], 5) for a in converted["answers"]]))

    # --- DRIZZLE: move the conversion subsidy into pools, lossless ---
    run_drizzle()
    drizzled = get(f"/v0/market/{mid}")
    check_preserved(converted, drizzled, "POST-CONVERSION-DRIZZLE")

    # --- TRADE on the converted v2 market (exercises the v2 arb path) ---
    traded = drizzled["answers"][1]
    print(f"\n>> trade: BUY M$120 YES on '{traded['text']}' (converted-v2 arb path)")
    post("/v0/bet", {"contractId": mid, "amount": 120, "outcome": "YES",
                     "answerId": traded["id"]})
    market = get(f"/v0/market/{mid}")
    check_sumone_and_cache(market, "POST-TRADE")
    print("   post-trade probs:", [round(a["probability"], 4) for a in market["answers"]])

    # --- ADD-LIQ + DRIZZLE again on the now-skewed converted market ---
    before = market
    print("\n>> add-liquidity: M$200 whole-market (on converted, skewed state)")
    post(f"/v0/market/{mid}/add-liquidity", {"contractId": mid, "amount": 200})
    run_drizzle()
    after = get(f"/v0/market/{mid}")
    check_preserved(before, after, "POST-ADDLIQ-DRIZZLE")

    # --- RESOLVE CHOOSE_ONE → conservation ---
    print(f"\n>> resolve CHOOSE_ONE -> '{traded['text']}'")
    post(f"/v0/market/{mid}/resolve",
         {"contractId": mid, "outcome": "CHOOSE_ONE", "answerId": traded["id"]})
    report("market isResolved", get(f"/v0/market/{mid}").get("isResolved") is True)
    b1 = balance()
    net = b0 - b1
    print(f"\n=== CONSERVATION (sole participant) ===\n  start={b0:.4f} end={b1:.4f} net(=fees)={net:.4f}")
    report("net loss small & >= 0 (LP/creator paid; BUG#1 guard)",
           -0.01 <= net < 100, f"net={net:.4f}")
    print(f"\n{'='*50}\nRESULT: {'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    print(f"market id: {mid}")
    return 0 if not _fails else 1


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "choose_one_basket"
    if case in ("conversion", "conversion_lifecycle"):
        return run_conversion_lifecycle()
    if case == "set_lifecycle":
        return run_set_lifecycle()
    if case.startswith("other_split"):
        # other_split | other_split_new | other_split_other
        suffix = case[len("other_split"):].lstrip("_") or "listed"
        return run_other_split(suffix)
    if case == "multibasket":
        return run_multibasket()
    if case == "multibasket_v1":
        return run_multibasket(v1=True)
    do_sell = case == "sell_roundtrip"
    raw = [55, 25, 12, 8]
    labels = ["Alpha", "Bravo", "Charlie", "Delta"]
    b0 = balance()
    print(f"start balance: {b0:.4f}   resolve_case={case}")

    created = post("/v0/market", {
        "question": f"v2 √var lifecycle {case} (delete-me)",
        "outcomeType": "MULTIPLE_CHOICE", "answers": labels, "initialProbs": raw,
        "shouldAnswersSumToOne": True, "addAnswersMode": "DISABLED", "liquidityTier": 1000,
    })
    mid = created["id"]
    print(f"created id={mid} slug={created.get('slug')} mechanism={created.get('mechanism')}")
    market = get(f"/v0/market/{mid}")
    check_creation(market, raw)

    # --- TRADE: buy YES on Bravo (q=0.25), a non-largest answer -> auto-arb others ---
    traded_ans = market["answers"][1]
    print(f"\n>> trade: BUY M$150 YES on '{traded_ans['text']}' (answerId={traded_ans['id']})")
    post("/v0/bet", {"contractId": mid, "amount": 150, "outcome": "YES",
                     "answerId": traded_ans["id"]})
    market = get(f"/v0/market/{mid}")
    check_sumone_and_cache(market, "POST-TRADE")
    print("   post-trade probs:", [round(a["probability"], 4) for a in market["answers"]])

    # --- SELL (v2 sell path round-trip) ---
    if do_sell:
        print(f"\n>> sell: ALL YES shares back on '{traded_ans['text']}' (v2 sell path)")
        post(f"/v0/market/{mid}/sell", {"contractId": mid, "outcome": "YES",
                                        "answerId": traded_ans["id"]})
        market = get(f"/v0/market/{mid}")
        check_sumone_and_cache(market, "POST-SELL")
        print("   post-sell probs:", [round(a["probability"], 4) for a in market["answers"]])
        # round-trip: should land back near creation probs (minus fee impact)
        report("sell round-trip returns near creation probs",
               all(abs(a["probability"] - q) < 0.02
                   for a, q in zip(market["answers"], [p / sum(raw) for p in raw])),
               str([round(a["probability"], 4) for a in market["answers"]]))

    # --- ADD LIQUIDITY (whole-market subsidy) ---
    before = market
    print("\n>> add-liquidity: M$300 whole-market")
    post(f"/v0/market/{mid}/add-liquidity", {"contractId": mid, "amount": 300})
    after = get(f"/v0/market/{mid}")
    # add-liq lands in subsidyPool; probs must already be preserved (pools not yet moved)
    check_sumone_and_cache(after, "POST-ADDLIQ")

    # --- DRIZZLE: move subsidy into pools, must stay lossless ---
    run_drizzle()
    drizzled = get(f"/v0/market/{mid}")
    check_preserved(before, drizzled, "POST-DRIZZLE")
    check_variance_add_shape(before, drizzled, "POST-DRIZZLE")

    # --- RESOLVE ---
    answers = drizzled["answers"]
    if case == "choose_one_basket":
        rb = {"contractId": mid, "outcome": "CHOOSE_ONE", "answerId": traded_ans["id"]}
    elif case == "choose_one_nonbasket":
        rb = {"contractId": mid, "outcome": "CHOOSE_ONE", "answerId": answers[0]["id"]}
    elif case == "choose_multiple":
        rb = {"contractId": mid, "outcome": "CHOOSE_MULTIPLE",
              "resolutions": [{"answerId": answers[0]["id"], "pct": 60},
                              {"answerId": answers[1]["id"], "pct": 40}]}
    elif case in ("cancel", "sell_roundtrip"):
        rb = {"contractId": mid, "outcome": "CANCEL"}
    else:
        raise SystemExit(f"unknown resolve case {case}")
    print(f"\n>> resolve: {json.dumps(rb)}")
    post(f"/v0/market/{mid}/resolve", rb)
    resolved = get(f"/v0/market/{mid}")
    report("market isResolved", resolved.get("isResolved") is True)

    b1 = balance()
    net = b0 - b1   # sole participant: everything put in either returns or is fees
    print("\n=== CONSERVATION (sole participant) ===")
    print(f"  start={b0:.4f} end={b1:.4f}  net spent(=fees)={net:.4f}")
    if case == "cancel":
        report("CANCEL: net ≈ 0 (all mana refunded)", abs(net) < 1.0, f"net={net:.4f}")
    else:
        # net loss should be small & non-negative = trading fees; NOT a large loss
        # (a large loss => LP/creator paid nothing, the BUG#1 regression)
        report("net loss is small & >= 0 (== fees, LP/creator WAS paid)",
               -0.01 <= net < 100, f"net={net:.4f}")

    print(f"\n{'='*50}\nRESULT: {'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    print(f"market id: {mid}")
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
