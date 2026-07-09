"""
Phase 6 — Realistic load test. Sends N diverse prompts through the running API,
then prints the final cost-savings report — the number that headlines the
portfolio case study.

Usage:
    python scripts/load_test.py --n 500 --base-url http://localhost:8000
"""
import argparse
import random
import sys
import time
import os

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.classifier.dataset import generate_dataset  # reuse the same prompt generator


def run_load_test(base_url: str, n: int, concurrency: int = 10):
    examples = generate_dataset(per_tier=max(1, n // 3))
    random.shuffle(examples)
    examples = examples[:n]

    sent = 0
    errors = 0
    start = time.time()

    with httpx.Client(timeout=30) as client:
        for i, ex in enumerate(examples):
            try:
                r = client.post(
                    f"{base_url}/v1/completions",
                    json={"prompt": ex.prompt, "context": ex.context},
                )
                r.raise_for_status()
                sent += 1
            except Exception as e:
                errors += 1
                print(f"  [{i}] request failed: {e}")
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(examples)} sent")

    elapsed = time.time() - start
    print(f"\nSent {sent} requests ({errors} errors) in {elapsed:.1f}s "
          f"({sent / elapsed:.1f} req/s)")

    # Pull the final report from the API itself
    with httpx.Client(timeout=30) as client:
        stats = client.get(f"{base_url}/v1/stats").json()

    print("\n=== COST SAVINGS REPORT ===")
    print(f"Total requests logged:   {stats['total_requests']}")
    print(f"Total cost (routed):     ${stats['total_cost_usd']:.4f}")
    print(f"Baseline cost (all top-tier model): ${stats['baseline_cost_usd']:.4f}")
    print(f"Savings:                 ${stats['savings_usd']:.4f}  "
          f"({stats['savings_pct']:.1f}%)")
    print(f"Routing distribution:    {stats['routing_distribution']}")
    print(f"Avg quality score:       {stats['avg_quality_score']}")
    print(f"Escalation rate:         {stats['escalation_rate_pct']}%")
    print("\nThis is the headline number for the portfolio case study:")
    print(f"  \"Reduced LLM API costs by {stats['savings_pct']:.0f}% "
          f"while maintaining {(stats['avg_quality_score'] or 0) * 100:.0f}% "
          f"quality agreement with the top-tier model.\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    run_load_test(args.base_url, args.n, args.concurrency)
