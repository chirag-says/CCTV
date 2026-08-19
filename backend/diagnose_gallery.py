"""
Face gallery quality report.

Loads the enrolled gallery exactly as a camera pipeline would, replays every
embedding through the recogniser's own decision rules, and reports which people
the system cannot reliably tell apart.

Run BEFORE changing any recognition threshold. Log lines like

    security 1 (distance=0.5323) vs security 3 (distance=0.5476) margin=0.0153
    Recovery rejected: verify_ambiguous_margin

are gallery problems, not threshold problems — this tells you which.

Usage:
    python diagnose_gallery.py                      # report at configured thresholds
    python diagnose_gallery.py -v                   # per-embedding detail
    python diagnose_gallery.py --json               # machine-readable
    python diagnose_gallery.py --json -o report.json
    python diagnose_gallery.py --match-margin 0.15  # what if the margin were stricter?
"""

import argparse
import json
import logging
import sys

sys.path.insert(0, ".")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse enrolled face gallery quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="show per-embedding distances for flagged persons",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit JSON instead of a text table",
    )
    parser.add_argument(
        "-o", "--output", metavar="PATH",
        help="write the report to a file instead of stdout",
    )
    parser.add_argument(
        "--pairs", type=int, default=20, metavar="N",
        help="how many confusable person-pairs to list (default: 20)",
    )
    # Threshold overrides — for asking "what would happen if…" without a restart.
    parser.add_argument("--match-distance", type=float, help="override FACE_MATCH_BALANCED")
    parser.add_argument("--match-margin", type=float, help="override FACE_MATCH_MIN_MARGIN")
    parser.add_argument("--verify-distance", type=float, help="override IDENTITY_VERIFY_MAX_DISTANCE")
    parser.add_argument("--verify-margin", type=float, help="override IDENTITY_VERIFY_MIN_MARGIN")
    args = parser.parse_args()

    # The service layer logs an INFO line per load; it is noise in a report.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    from app.services.person_service import PersonService
    from app.vision.gallery_diagnostics import (
        Thresholds, analyse_gallery, format_report, report_to_dict,
    )

    try:
        encodings = PersonService.get_all_encodings()
    except Exception as e:
        print(f"Could not load encodings from the database: {e}", file=sys.stderr)
        print("Is PostgreSQL running and is .env pointing at it?", file=sys.stderr)
        return 2

    thresholds = Thresholds.from_settings(
        match_distance=args.match_distance,
        match_margin=args.match_margin,
        verify_distance=args.verify_distance,
        verify_margin=args.verify_margin,
    )

    report = analyse_gallery(
        encodings, thresholds=thresholds, max_confusable_pairs=args.pairs
    )

    if args.json:
        text = json.dumps(report_to_dict(report), indent=2)
    else:
        text = format_report(report, verbose=args.verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"Wrote {args.output}")
    else:
        print(text)

    # Exit code carries the verdict, so this can gate a deployment check:
    #   0 = gallery is separable, 1 = someone needs re-enrollment.
    return 1 if report.high_risk else 0


if __name__ == "__main__":
    sys.exit(main())
