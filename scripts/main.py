import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse, os, json, time
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

from src.utils.seed import set_global_determinism
from src.data.preprocess import applyPCA, createImageCubes, createImageCubes_2D, splitDataset_Min
from src.models.tetnet import build_tetnet

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--tecem_dir", type=str, default="outputs/SA_TECEM_cls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_ratio", type=float, default=0.98)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_global_determinism(args.seed)


    if args.test_ratio == 0.98:
        validation_split = 0.5
        train_ratio_pct = 1
    else:
  
        validation_split = 0.5
        train_ratio_pct = int(round((1-args.test_ratio)*100))


    windowSize_2D = 15
    windowSize_3D = 15
    bands = 16
    dbands = bands

    X = sio.loadmat(os.path.join(args.data_dir, "Salinas_corrected.mat"))["salinas_corrected"]
    gt = sio.loadmat(os.path.join(args.data_dir, "Salinas_gt.mat"))["salinas_gt"]

    # PCA for 3D/cbam branch
    PCA_data, _ = applyPCA(X, numComponents=bands)

    # Load TECEM maps
    cls_maps = []
    for cid in range(1, bands+1):
        p = os.path.join(args.tecem_dir, f"SA_TECEM_cls{cid}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing TECEM map: {p}. Run scripts/precompute_tecem.py first.")
        cls_maps.append(np.load(p, allow_pickle=True))

    # build patches for each map
    X_cls_patches=[]
    y_ref=None
    for m in cls_maps:
        Xc, yc = createImageCubes_2D(m, gt, windowSize=windowSize_2D, removeZeroLabels=True)
        if y_ref is None:
            y_ref = yc
        else:
            # safety: ensure same labels length/order
            assert len(yc)==len(y_ref)
        X_cls_patches.append(Xc.astype(np.float32))

    # 3D patches from PCA cube
    X_3D, y_3D = createImageCubes(PCA_data, gt, windowSize=windowSize_3D, removeZeroLabels=True)
    assert len(y_3D)==len(y_ref)
    X_3D = X_3D[..., np.newaxis].astype(np.float32)

    # cbam2D branch uses PCA cube as (win,win,PCs)
    X_cbam2D, y_cbam = createImageCubes(PCA_data, gt, windowSize=windowSize_2D, removeZeroLabels=True)
    assert len(y_cbam)==len(y_ref)
    X_cbam2D = X_cbam2D.astype(np.float32)

    N = len(y_ref)
    idx = np.arange(N)

    Xtr_i, Xte_i, ytr, yte = splitDataset_Min(idx, y_ref, args.test_ratio, randomState=args.seed)
    Xtr_i = Xtr_i.astype(int); Xte_i = Xte_i.astype(int)

    def take(arr):
        return arr[Xtr_i], arr[Xte_i]

    train_list=[]; test_list=[]; all_list=[]
    for Xc in X_cls_patches:
        tr, te = take(Xc)
        train_list.append(tr); test_list.append(te); all_list.append(Xc)
    tr3, te3 = take(X_3D); train_list.append(tr3); test_list.append(te3); all_list.append(X_3D)
    tr2, te2 = take(X_cbam2D); train_list.append(tr2); test_list.append(te2); all_list.append(X_cbam2D)

    # categorical labels
    ytrain = np_utils.to_categorical(ytr, bands)
    ytest  = np_utils.to_categorical(yte, bands)
    yall   = np_utils.to_categorical(y_ref, bands)

    model = build_tetnet(windowSize_2D=windowSize_2D, windowSize_3D=windowSize_3D, bands=bands, dbands=dbands)
    opt = Adam(learning_rate=args.lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    run_dir = os.path.join("runs", "SA", f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "best.hdf5")
    callbacks = [ModelCheckpoint(ckpt_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]

    start = time.time()
    history = model.fit(train_list, ytrain, epochs=args.epochs, batch_size=args.batch_size,
                        validation_split=validation_split, callbacks=callbacks, verbose=2)
    train_time = time.time()-start

    loss, acc = model.evaluate(test_list, ytest, verbose=0)

    model.load_weights(ckpt_path)
    Y_pred = model.predict(all_list, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(yall, axis=1)

    oa = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(cm)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, output_dict=False)

    metrics = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "validation_split": validation_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_time_sec": train_time,
    }
    predict_acc = {
        # "eval_loss": float(loss),
        # "eval_acc": float(acc),
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa),
        # "each_class_acc": [float(x) for x in each_acc.tolist()],
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("[DONE] Metrics saved to", run_dir)
    print(metrics)
    print(f"prediction accuracy: = {json.dumps(predict_acc, indent=4)}")

if __name__ == "__main__":
    main()
