import argparse, os, json
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from src.utils.seed import set_global_determinism
from src.tecem.core import tecem_all_classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--out_dir", type=str, default="outputs/SA_TECEM_cls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_ratio", type=float, default=0.01)  # matches notebook when test_ratio=0.98
    ap.add_argument("--radius", type=int, default=9)
    ap.add_argument("--sam_quantile", type=float, default=90)
    ap.add_argument("--fuse_mode", type=str, default="max", choices=["max","blend"])
    args = ap.parse_args()

    set_global_determinism(args.seed)

    X = sio.loadmat(os.path.join(args.data_dir, "Salinas_corrected.mat"))["salinas_corrected"]
    gt = sio.loadmat(os.path.join(args.data_dir, "Salinas_gt.mat"))["salinas_gt"]

    os.makedirs(args.out_dir, exist_ok=True)

    class_ids = list(range(1, int(gt.max())+1))
    maps = tecem_all_classes(
        X, gt,
        train_ratio=args.train_ratio,
        seed=args.seed,
        radius=args.radius,
        sam_quantile=args.sam_quantile,
        fuse_mode=args.fuse_mode
    )
    for cid, m in maps.items():
        np.save(os.path.join(args.out_dir, f"SA_TECEM_cls{cid}.npy"), m.astype(np.float64))
    meta = vars(args)
    meta.update({"num_classes": int(gt.max()), "shape": list(X.shape)})
    with open(os.path.join(args.out_dir, "tecem_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved {len(maps)} TECEM maps to {args.out_dir}")

if __name__ == "__main__":
    main()
