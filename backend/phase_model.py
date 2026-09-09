"""
Phase-by-phase cost distribution and returns for a hosted SentinelAI service.

Three software states (cameras per RTX 5070, from capacity_table.py basis):
  A  code exactly as today: CPU-bound by per-camera JPEG streaming (16 ms/frame x 30 fps)
     and Roboflow latency -> ~16 cameras per 8-core node, GPU barely matters.
  B  after the two-week fixes (JPEG only when watched, Roboflow removed, CUDA check),
     analysing 10 fps as the code does now (FRAME_SKIP=3): Basic 24.6, Plus 12.2, Pro 7.1 per GPU.
  C  optimised (TensorRT, batching, WebRTC), 30 fps: Basic 47.8, Plus 28.8, Pro 19.1 per GPU.
Every INR input is listed below.  python phase_model.py
"""
import math

PRICE = {"Basic": 1000, "Plus": 2000, "Pro": 4000}     # INR per camera per month (genuine pricing)
MIX = {"Basic": 0.7, "Plus": 0.2, "Pro": 0.1}           # camera mix across clients
from capacity_table import table as _cap_table
_gpu, _t = _cap_table()
CAP = {"B": {k: _t[(k, 10)]["today"] for k in ("Basic", "Plus", "Pro")},    # measured, today's code, 10 fps
       "C": {k: _t[(k, 30)]["native"] for k in ("Basic", "Plus", "Pro")}}   # projected, native hot loop + TensorRT, 30 fps
GPUS_PER_NODE = 2
NODE_INR = 194_000
SPARE_NODES = 1
SETUP_BOX_INR, SETUP_FEE_INR = 12_000, 15_000          # per client site: gateway mini-PC; one-time fee charged
CAMS_PER_SITE = 10
OWN_SITE_ONE_TIME = 60_000                             # UPS, switch, router at own premises
OWN_NODE_MONTH, COLO_NODE_MONTH = 13_000, 27_000       # per node: power+bandwidth+space (own) / colo
COLO_FROM_CAMS = 200                                   # move to colocation beyond this
PEOPLE = [(0, 0), (200, 40_000), (500, 120_000), (1000, 250_000)]   # (cams threshold, INR/month) support+ops
OTHER_PCT = 0.10                                       # payment gateway, support tools, misc, of revenue

def cams_per_node(state):
    frac = sum(MIX[t] / CAP[state][t] for t in MIX)    # GPU fraction per mixed camera
    return GPUS_PER_NODE / frac

def people(n):
    return max(p for c, p in PEOPLE if n >= c)

def phase(n, state):
    per_node = cams_per_node(state)
    nodes = math.ceil(n / per_node) + SPARE_NODES
    sites = math.ceil(n / CAMS_PER_SITE)
    colo = n > COLO_FROM_CAMS
    capex = nodes * NODE_INR + OWN_SITE_ONE_TIME + sites * SETUP_BOX_INR
    setup_income = sites * SETUP_FEE_INR
    rev = n * sum(MIX[t] * PRICE[t] for t in MIX)
    infra = nodes * (COLO_NODE_MONTH if colo else OWN_NODE_MONTH)
    opex = infra + people(n) + rev * OTHER_PCT
    net = rev - opex
    return dict(per_node=per_node, nodes=nodes, capex=capex, setup=setup_income, rev=rev,
                infra=infra, people=people(n), other=rev * OTHER_PCT, opex=opex, net=net,
                margin=net / rev, payback=(capex - setup_income) / net if net > 0 else float("inf"),
                colo=colo)

if __name__ == "__main__":
    avg_price = sum(MIX[t] * PRICE[t] for t in MIX)
    print(f"Prices {PRICE}, mix {MIX}, avg {avg_price:.0f} INR/cam/month\n")
    for state in ("B", "C"):
        print(f"=== Software state {state}: {cams_per_node(state):.0f} mixed cameras per dual-5070 node ===")
        print(f"{'cams':>5s} {'nodes':>5s} {'capex':>10s} {'setup fees':>10s} {'revenue/mo':>11s} {'infra/mo':>9s} {'people':>8s} {'other':>7s} {'net/mo':>9s} {'margin':>7s} {'payback':>8s}")
        for n in (25, 50, 100, 200, 500, 1000):
            p = phase(n, state)
            print(f"{n:5d} {p['nodes']:5d} {p['capex']:10,.0f} {p['setup']:10,.0f} {p['rev']:11,.0f} {p['infra']:9,.0f} {p['people']:8,.0f} {p['other']:7,.0f} {p['net']:9,.0f} {p['margin']:6.0%} {p['payback']:5.1f} mo{' colo' if p['colo'] else ''}")
        print()
    print("State A (code exactly as today): ~16 cameras per node regardless of GPU; at 25 cameras that is 3 nodes")
    print(f"  -> capex {3*NODE_INR+OWN_SITE_ONE_TIME:,.0f}, revenue {25*avg_price:,.0f}/mo, infra {3*OWN_NODE_MONTH:,.0f}/mo")
