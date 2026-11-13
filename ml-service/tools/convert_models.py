"""Re-save SaansCare model bundles so XGBoost boosters are stored as JSON text (portable across OS/builds)."""
import sys, glob, os, joblib
from pathlib import Path
from xgboost import XGBModel
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob("*.joblib")):
    b = joblib.load(f); xgb = {}
    for key, v in list(b.items()):
        if hasattr(v, "steps"):
            for name, est in v.steps:
                if isinstance(est, XGBModel):
                    xgb[f"{key}::{name}"] = bytes(est.get_booster().save_raw("json")).decode("utf-8")
                    est._Booster = None
    b["xgb_json"] = xgb
    joblib.dump(b, dst / f.name, compress=3)
    print(f.name, "xgb parts:", list(xgb), os.path.getsize(dst / f.name))
