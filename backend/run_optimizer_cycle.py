"""
Run exactly ONE Gallery Optimizer cycle, in the foreground, and stop.

This is the controlled way to bring the optimizer up on a large backlog:
you watch a single cycle, inspect what it produced, and only then decide
whether to set ENABLE_IDENTITY_REVIEW=true and let the daemon run on its
own schedule.

    python run_optimizer_cycle.py --no-vlm     # clustering + candidates only
    python run_optimizer_cycle.py --no-vlm --vlm-budget 0
    python run_optimizer_cycle.py              # with the configured VLM

--no-vlm is the right first run. It validates cluster quality and candidate
selection without spending thousands of Qwen calls on clusters that may turn
out to be badly cut. Use gallery_optimizer_dryrun.py first if you want the
numbers without writing anything at all — this script DOES write clusters and
review candidates (it never touches a gallery; only a human approval does).
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("urllib3", "httpx", "sqlalchemy.engine"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from app.config import settings
from app.services.identity_review_service import IdentityReviewService


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-vlm", action="store_true",
                    help="disable the VLM for this cycle (recommended first run)")
    ap.add_argument("--vlm-budget", type=int, default=None,
                    help="override IDENTITY_REVIEW_VLM_MAX_PER_CYCLE")
    args = ap.parse_args()

    if args.no_vlm:
        settings.IDENTITY_REVIEW_VLM_PROVIDER = "disabled"
    if args.vlm_budget is not None:
        settings.IDENTITY_REVIEW_VLM_MAX_PER_CYCLE = args.vlm_budget

    # get_vlm_reviewer() reads settings on every call, so the override above
    # takes effect without any singleton to reset.
    from app.services.identity_review_worker import IdentityReviewWorker

    print(f"\nVLM provider      : {settings.IDENTITY_REVIEW_VLM_PROVIDER}")
    print(f"VLM budget        : {settings.IDENTITY_REVIEW_VLM_MAX_PER_CYCLE}")
    print(f"Cluster threshold : {settings.IDENTITY_REVIEW_CLUSTER_THRESHOLD}")
    print(f"Min cluster size  : {settings.IDENTITY_REVIEW_MIN_CLUSTER_SIZE}\n")

    before = IdentityReviewService.get_dashboard_stats()

    worker = IdentityReviewWorker()
    worker._run_one_cycle()

    after = IdentityReviewService.get_dashboard_stats()

    print("\n" + "=" * 58)
    print("  CYCLE RESULT")
    print("=" * 58)
    for key in sorted(after):
        delta = after[key] - before.get(key, 0)
        arrow = f"  ({delta:+d})" if delta else ""
        print(f"  {key:<26} {after[key]:>6}{arrow}")
    print("=" * 58)
    print("  No gallery was modified. Gallery changes require a human")
    print("  approval in the AI Identity Review page.")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    sys.exit(main())
