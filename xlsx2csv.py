"""Convert all xlsx files under data/ to CSV (one CSV per sheet, UTF-8-BOM)."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"d:\mathmodel\SXJM\data")
OUT_DIR = DATA_DIR / "csv"
SUBDIRS = ["raw-datta", "result-data"]

for sub in SUBDIRS:
    src_dir = DATA_DIR / sub
    dst_dir = OUT_DIR / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("*.xlsx")):
        stem = f.stem
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            df = pd.read_excel(f, sheet_name=sheet)
            out = dst_dir / f"{stem}__{sheet}.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  {sub}/{out.name}  shape={df.shape}")
        print(f"[done] {f.name}  -> {len(xl.sheet_names)} csv")

print("\nAll xlsx files converted to CSV.")
