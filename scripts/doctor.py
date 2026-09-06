from pathlib import Path
import importlib
import platform
import sys

checks = []

def check(label, ok, detail=""):
    checks.append(ok)
    mark = "✓" if ok else "✗"
    print(f"{mark} {label}" + (f": {detail}" if detail else ""))

check("Python 3.12", sys.version_info[:2] == (3, 12), platform.python_version())
for mod in ["transformers", "spacy", "pandas", "pytest", "sklearn"]:
    try:
        m = importlib.import_module(mod)
        check(mod, True, getattr(m, "__version__", "installed"))
    except Exception as e:
        check(mod, False, str(e))

for p in [
    Path("data/raw/bayan_feedback.csv"),
    Path("data/raw/bayan_raw_sample.csv"),
    Path("data/eval/pii_test_set.csv"),
    Path("data/eval/arabic_normalize_golden.csv"),
    Path("data/models/bayan_ner.conll"),
    Path("data/models/bayan_qa.json"),
    Path("data/search/bayan_cases.csv"),
    Path("data/search/bayan_queries.jsonl"),
]:
    check(str(p), p.exists())

print("\nALL GOOD" if all(checks) else "\nFIX FAILED CHECKS BEFORE THE LAB")
raise SystemExit(0 if all(checks) else 1)
