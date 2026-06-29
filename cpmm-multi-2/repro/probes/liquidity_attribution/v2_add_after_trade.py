import json
import urllib.request
import subprocess

API = "http://localhost:8088"
USER = "test-user-1"
VENDOR = "/home/evand/predictions/vendor/manifold"


def req(m, p, b=None):
    d = json.dumps(b).encode() if b is not None else None
    r = urllib.request.Request(API + p, data=d, method=m)
    r.add_header("X-Local-User", USER)
    if d is not None:
        r.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(r, timeout=60).read().decode())


def bal():
    return req("GET", "/v0/me")["balance"]


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


def poolsum(m):
    return sum(a["pool"]["YES"] + a["pool"]["NO"] for a in m["answers"])


def run(label, add=0, drizzle=0):
    b0 = bal()
    m = req(
        "POST",
        "/v0/market",
        {
            "question": f"v2 liq+trade {label} (delete-me)",
            "outcomeType": "MULTIPLE_CHOICE",
            "answers": ["X", "Y", "Z"],
            "initialProbs": [40, 30, 30],
            "shouldAnswersSumToOne": True,
            "addAnswersMode": "DISABLED",
            "liquidityTier": 1000,
        },
    )
    mid = m["id"]
    ans = req("GET", f"/v0/market/{mid}")["answers"]
    mech = req("GET", f"/v0/market/{mid}").get("mechanism")
    req(
        "POST",
        "/v0/bet",
        {"contractId": mid, "amount": 300, "outcome": "YES", "answerId": ans[0]["id"]},
    )
    pre = req("GET", f"/v0/market/{mid}")
    if add:
        req(
            "POST",
            f"/v0/market/{mid}/add-liquidity",
            {"contractId": mid, "amount": add},
        )
    if drizzle:
        drizzle_n(drizzle)
    mm = req("GET", f"/v0/market/{mid}")
    req(
        "POST",
        f"/v0/market/{mid}/resolve",
        {"contractId": mid, "outcome": "CHOOSE_ONE", "answerId": ans[0]["id"]},
    )
    net = b0 - bal()
    print(
        f"{label:42s} mech={mech} put_in={1000 + add:6.0f} net_lost={net:9.4f}  poolΣ {poolsum(pre):.1f}->{poolsum(mm):.1f}"
    )
    return net


print(
    "=== V2 (lossless add): create -> bet300 -> [add+drizzle] -> resolve ; single participant ==="
)
a = run("A2: bet -> resolve", add=0)
b = run("B2: bet -> add300 + drizzle x8 -> resolve", add=300, drizzle=8)
print(
    f"\ndelta (B2 - A2) = {b - a:.4f}  <-- v2 should conserve (~0); contrast v1's 78.86"
)
