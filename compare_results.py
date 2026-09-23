"""Check a fresh real-data run against the published results, not prose."""
import argparse
import json
import math
from pathlib import Path


def same(expected, actual, where="root"):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"Different keys at {where}")
        for key in expected:
            same(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"Different lengths at {where}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            same(a, b, f"{where}[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-3):
            raise AssertionError(f"Numeric mismatch at {where}: {expected} versus {actual}")
    elif expected != actual:
        raise AssertionError(f"Mismatch at {where}: {expected!r} versus {actual!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("published", type=Path)
    parser.add_argument("reproduced", type=Path)
    args = parser.parse_args()
    def read(folder, name):
        return json.loads((folder / name).read_text())
    same(read(args.published, "input_manifest.json"), read(args.reproduced, "input_manifest.json"), "inputs")
    for stage in ["develop", "evaluate"]:
        old = read(args.published, f"{stage}.json")
        new = read(args.reproduced, f"{stage}.json")
        for key in ["stage", "protocol_sha256", "compass_sha256", "weighting", "aggregate"]:
            same(old[key], new[key], f"{stage}.{key}")
        same(len(old["cases"]), len(new["cases"]), "case_count")
        for a, b in zip(old["cases"], new["cases"]):
            # Wall-clock timings differ across machines; every scientific
            # result, exclusion and input fingerprint must still agree.
            same({k: v for k, v in a.items() if k != "seconds"},
                 {k: v for k, v in b.items() if k != "seconds"}, f"{stage}.{a['case']}")
        if stage == "evaluate":
            for key in ["confidence_risk", "frozen_threshold_result"]:
                same(old[key], new[key], key)
    old = read(args.published, "selection.json")
    new = read(args.reproduced, "selection.json")
    same({k: v for k, v in old.items() if k != "frozen_utc"},
         {k: v for k, v in new.items() if k != "frozen_utc"}, "selection")
    print("REPRODUCED: all 22 source hashes, all 11 cube results, all development candidates, exclusions, frozen selection and evaluation metrics agree (floating tolerance 0.001).")


if __name__ == "__main__":
    main()
