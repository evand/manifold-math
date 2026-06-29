import json
import urllib.request
import subprocess

API = "http://localhost:8088"
U1 = "test-user-1"
U2 = "IPTOzEqrpkWmEzh6hwvAyY9PqFb2"
VENDOR = "/home/evand/predictions/vendor/manifold"


def req(user, m, p, b=None):
    d = json.dumps(b).encode() if b is not None else None
    r = urllib.request.Request(API + p, data=d, method=m)
    r.add_header("X-Local-User", user)
    if d is not None:
        r.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(r, timeout=60).read().decode())


def bal(u):
    return req(u, "GET", "/v0/me")["balance"]


def drizzle_n(n):
    cmd = "set -a && source .env.local && set +a && npx ts-node src/run-drizzle-once.ts"
    for _ in range(n):
        subprocess.run(
            ["bash", "-lc", cmd],
            cwd=f"{VENDOR}/backend/scheduler",
            capture_output=True,
            text=True,
            timeout=300,
        )


def trial(lp2_adds, resolve):
    b0 = bal(U1)
    m = req(
        U1,
        "POST",
        "/v0/market",
        {
            "question": f"binary attrib lp2={lp2_adds} r={resolve} (delete-me)",
            "outcomeType": "BINARY",
            "initialProb": 50,
            "totalBounty": 0,
            "liquidityTier": 1000,
        },
    )
    mid = m["id"]
    req(
        U2, "POST", "/v0/bet", {"contractId": mid, "amount": 500, "outcome": "YES"}
    )  # U2 trader skews pool up
    if lp2_adds:
        req(
            U2,
            "POST",
            f"/v0/market/{mid}/add-liquidity",
            {"contractId": mid, "amount": 1000},
        )  # U2 = LP2
        drizzle_n(6)
    req(
        U1, "POST", f"/v0/market/{mid}/resolve", {"contractId": mid, "outcome": resolve}
    )  # creator resolves
    return b0 - bal(U1)  # LP1's net LOSS (negative = LP1 profit)


print("LP1=test-user-1 (creator, ante 1000, does nothing else). LP2/trader=adminuser.")
print(
    "LP1 should be INDIFFERENT to LP2's lossless add (no price move, no trade after)."
)
print(
    f"{'resolve':10s}{'baseline(no LP2)':>20s}{'treatment(LP2 adds)':>22s}{'LP1 shift':>12s}"
)
for r in ["YES", "NO"]:
    base = trial(False, r)
    treat = trial(True, r)
    print(f"{r:10s}{base:20.4f}{treat:22.4f}{treat - base:12.4f}")
print(
    "\n(LP1 net LOSS; negative=profit. Nonzero 'shift' that differs by direction => LP1 not indifferent => attribution wonky.)"
)
