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


def run(resolve_idx, add):
    b0 = bal()
    m = req(
        "POST",
        "/v0/market",
        {
            "question": f"v1 dir r{resolve_idx} a{add} (delete-me)",
            "outcomeType": "MULTIPLE_CHOICE",
            "answers": ["X", "Y", "Z"],
            "shouldAnswersSumToOne": True,
            "addAnswersMode": "DISABLED",
            "liquidityTier": 1000,
        },
    )
    mid = m["id"]
    ans = req("GET", f"/v0/market/{mid}")["answers"]
    req(
        "POST",
        "/v0/bet",
        {"contractId": mid, "amount": 300, "outcome": "YES", "answerId": ans[0]["id"]},
    )  # bet YES on X(idx0)
    if add:
        req(
            "POST",
            f"/v0/market/{mid}/add-liquidity",
            {"contractId": mid, "amount": add},
        )
        drizzle_n(8)
    req(
        "POST",
        f"/v0/market/{mid}/resolve",
        {
            "contractId": mid,
            "outcome": "CHOOSE_ONE",
            "answerId": ans[resolve_idx]["id"],
        },
    )
    return b0 - bal()


print(
    "bet YES on X(idx0). resolve to each. net_lost = value NOT returned to sole participant."
)
print(f"{'resolve->':12s}{'A:no-add':>12s}{'B:add300':>12s}{'destroyed(B-A)':>16s}")
for i, name in enumerate(["X(bet ans)", "Y", "Z"]):
    a = run(i, 0)
    b = run(i, 300)
    print(f"{name:12s}{a:12.4f}{b:12.4f}{b - a:16.4f}")
