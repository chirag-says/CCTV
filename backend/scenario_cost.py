"""
Scenario: an organisation with N cameras, surveillance H hours/day.
Compares AWS Mumbai (L4) against on-premises RTX 5070 nodes, per tier.

    python scenario_cost.py --cams 400 --hours 12

All prices INR. Cloud and leased-line figures exclude 18% GST; hardware
figures are retail (GST included). Sources in the accompanying message.
"""
import argparse, json, math
from cost_model import tier_loads, cams_per_gpu, INR_PER_USD, UTIL, FPS

DAYS = 30
MONTHS = 36

# ── cloud: AWS ap-south-1 (Mumbai), Linux on-demand ──────────────────────
L4_USD_HR = 0.966          # g6.xlarge, 1x L4 24 GB
APP_USD_HR = 0.2016        # m6i.xlarge (API + websockets), 24x7
DB_USD_HR = 0.2016         # m6i.xlarge (Postgres), 24x7
DB_EBS_GB, EBS_USD_GB = 500, 0.0912
S3_USD_GB = 0.025
EGRESS_USD_GB = 0.1093
LIVE_STREAMS = 16          # concurrent live streams watched from cloud, all day
STREAM_MBPS = 1.0          # substream bitrate per camera (720p, 5-8 fps)
EVENTS_PER_CAM_DAY = 200
EVENT_MB = 0.2 + 1.25      # snapshot + 10 s clip
RETENTION_DAYS = 30
ILL_INR_MONTH = {300: 90_000, 400: 90_000}   # 500 Mbps internet leased line, mid of quoted range
GATEWAY_INR = 80_000       # on-site relay box that pulls RTSP and pushes to cloud (one-time)

# ── on-prem: dual RTX 5070 node ───────────────────────────────────────────
GPU_INR = 62_000           # RTX 5070 12 GB partner card, mid of Indian retail range
NODE_PLATFORM_INR = 70_000 # Ryzen 7 / i5, 32 GB, 1 TB NVMe, 1000 W PSU, case (holds 2 GPUs)
GPUS_PER_NODE = 2
NODE_WATTS_LOAD, NODE_WATTS_IDLE = 550, 60
DB_SERVER_INR, NAS_INR = 80_000, 120_000     # API+Postgres box; 4-bay NAS with 4x8 TB
INFRA_WATTS = 150          # DB server + NAS, 24x7
UPS_INR_PER_KVA, UPS_HEADROOM = 22_000, 1.3
RACK_INR = 60_000
COOLING_FACTOR = 0.4       # AC energy as fraction of IT energy
TARIFF = 9.0               # INR per kWh, commercial
MAINT_PCT_YEAR = 0.08      # spares, drives, fans

def money(x): return f"{x:,.0f}"

def cloud(n, hours, ms):
    per_gpu = cams_per_gpu(ms, "L4")
    gpus = math.ceil(n / per_gpu)
    gpu_m = gpus * L4_USD_HR * hours * DAYS
    app_m = (APP_USD_HR + DB_USD_HR) * 24 * DAYS + DB_EBS_GB * EBS_USD_GB
    stored_gb = n * EVENTS_PER_CAM_DAY * EVENT_MB / 1024 * RETENTION_DAYS
    s3_m = stored_gb * S3_USD_GB
    egress_gb = LIVE_STREAMS * STREAM_MBPS / 8 * 3600 * hours * DAYS / 1024
    egress_m = max(0, egress_gb - 100) * EGRESS_USD_GB
    usd = gpu_m + app_m + s3_m + egress_m
    inr = usd * INR_PER_USD + ILL_INR_MONTH[n]
    return {"gpus": gpus, "per_gpu": per_gpu, "gpu": gpu_m * INR_PER_USD, "app": app_m * INR_PER_USD,
            "s3": s3_m * INR_PER_USD, "egress": egress_m * INR_PER_USD, "ill": ILL_INR_MONTH[n],
            "month": inr, "uplink_mbps": n * STREAM_MBPS, "gateway": GATEWAY_INR}

def onprem(n, hours, ms):
    per_gpu = cams_per_gpu(ms, "RTX 5070")
    gpus = math.ceil(n / per_gpu)
    nodes = math.ceil(gpus / GPUS_PER_NODE) + 1          # N+1 spare node
    node_inr = GPUS_PER_NODE * GPU_INR + NODE_PLATFORM_INR
    it_kw = (nodes * NODE_WATTS_LOAD + INFRA_WATTS) / 1000
    ups = math.ceil(it_kw * UPS_HEADROOM) * UPS_INR_PER_KVA
    capex = nodes * node_inr + DB_SERVER_INR + NAS_INR + ups + RACK_INR
    kwh_day = ((nodes - 1) * (NODE_WATTS_LOAD * hours + NODE_WATTS_IDLE * (24 - hours))
               + NODE_WATTS_IDLE * 24 + INFRA_WATTS * 24) / 1000
    power_m = kwh_day * (1 + COOLING_FACTOR) * TARIFF * DAYS
    maint_m = capex * MAINT_PCT_YEAR / 12
    return {"gpus": gpus, "per_gpu": per_gpu, "nodes": nodes, "capex": capex, "ups": ups,
            "it_kw": it_kw, "power": power_m, "maint": maint_m, "opex": power_m + maint_m,
            "equiv_month": capex / MONTHS + power_m + maint_m}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench_NVIDIA_GeForce_GTX_1650.json")
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--cams", type=int, nargs="+", default=[300, 400])
    a = ap.parse_args()
    tiers = tier_loads(json.load(open(a.bench)))
    print(f"{a.hours} h/day, {DAYS} days/month, {FPS} analysed fps/camera, {int(UTIL*100)}% GPU utilisation, INR {INR_PER_USD}/USD\n")
    for n in a.cams:
        print(f"=== {n} cameras ===")
        for t, ms in tiers.items():
            c, o = cloud(n, a.hours, ms), onprem(n, a.hours, ms)
            print(f"\n{t}")
            print(f"  CLOUD  AWS Mumbai: {c['gpus']} x g6.xlarge L4 ({c['per_gpu']:.1f} cams/GPU) for {a.hours} h/day")
            print(f"         GPU {money(c['gpu'])} + app/DB {money(c['app'])} + S3 {money(c['s3'])} + egress {money(c['egress'])} + leased line {money(c['ill'])}")
            print(f"         = {money(c['month'])} INR/month  | {money(c['month']*12)} INR/year | {money(c['month']*MONTHS)} over 3 years"
                  f" | one-time gateway {money(c['gateway'])} | uplink needed {c['uplink_mbps']:.0f} Mbps")
            print(f"  ONPREM RTX 5070: {o['gpus']} GPUs ({o['per_gpu']:.1f} cams/GPU) in {o['nodes']} dual-GPU nodes incl. 1 spare, IT load {o['it_kw']:.1f} kW")
            print(f"         capex {money(o['capex'])} (incl. UPS {money(o['ups'])}) | power {money(o['power'])} + maintenance {money(o['maint'])} = opex {money(o['opex'])} INR/month")
            print(f"         = {money(o['equiv_month'])} INR/month equivalent over 3 years | {money(o['capex'] + o['opex']*MONTHS)} total over 3 years")
            print(f"  per camera per month: cloud {c['month']/n:,.0f}  |  on-prem {o['equiv_month']/n:,.0f}")
        print()

if __name__ == "__main__":
    main()
