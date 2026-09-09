import csv
from collections import Counter
from pathlib import Path

DATA = Path("data/raw/bayan_feedback.csv")


def main():
    with DATA.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    ar_rows = [r for r in rows if r["lang"] == "ar"]
    counts = Counter(r["dialect_region"] for r in ar_rows)
    total = len(ar_rows)

    print(f"Arabic rows: {total}")
    print(f"{'Region':<10} {'Count':>8} {'Share':>8}")
    for region, count in counts.most_common():
        print(f"{region:<10} {count:>8} {100 * count / total:>7.1f}%")


if __name__ == "__main__":
    main()
