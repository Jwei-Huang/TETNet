import numpy as np
from sklearn.decomposition import PCA

def normalize01(a):
    a = a.astype(np.float32)
    mn, mx = np.nanmin(a), np.nanmax(a)
    return (a - mn) / (mx - mn + 1e-8)


def SAM(R, d):
    """
    Spectral Angle Mapper
    R: (N,L)
    d: (L,)
    returns (N,) in radians
    """
    R = np.asarray(R)
    d = np.asarray(d)

    Rn = np.linalg.norm(R, axis=1) + 1e-12
    dn = np.linalg.norm(d) + 1e-12

    cos = (R @ d) / (Rn * dn)
    cos = np.clip(cos, -1.0, 1.0)

    return np.arccos(cos)


def ED(R, d):
    """Euclidean distance"""
    R = np.asarray(R)
    d = np.asarray(d)
    return np.linalg.norm(R - d[None, :], axis=1)



def wbs_cem_score(X, d, weight_mode="SAM", eps=1e-3, use_double=True):

    H, W, L = X.shape
    R = X.reshape(-1, L)

    if use_double:
        R = R.astype(np.float64)
        d = d.astype(np.float64)

    wm = weight_mode.upper()
    if wm == "ED":
        w = ED(R, d)
    elif wm == "SAM":
        w = SAM(R, d)
    else:
        raise ValueError("weight_mode must be 'ED' or 'SAM'")  #or you can define your own weighting methods

    N = R.shape[0]
    Rstar = (R.T * w) @ R / float(N)

    tr = np.trace(Rstar) / L
    lam = eps * (tr + 1e-12)

    v = np.linalg.solve(Rstar + lam * np.eye(L), d)

    denom = float(d @ v) + 1e-12
    score = (R @ v) / denom

    score_map = score.reshape(H, W).astype(np.float32)

    return score_map, w.astype(np.float32), v.astype(np.float32)


def _pca_work_cube(X, n_comp=16, seed=0, pca_sample=60000):

    H, W, L = X.shape
    rng = np.random.default_rng(seed)

    R = X.reshape(-1, L).astype(np.float32)
    idx = rng.choice(R.shape[0], size=min(pca_sample, R.shape[0]), replace=False)

    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(R[idx])

    Z = pca.transform(R).reshape(H, W, n_comp)

    Z_n = np.zeros_like(Z, dtype=np.float32)
    for k in range(n_comp):
        Z_n[..., k] = normalize01(Z[..., k])

    return Z_n.astype(np.float32)

def tecem_for_class(
    X,
    gt,
    class_id,
    train_ratio=0.01,
    seed=0,
    radius=9,
    sam_quantile=90,
    min_train_keep=10,
    expand=0,
    fuse_mode="max",
    alpha=0.7,
):

    H, W, L = X.shape


    X_work = _pca_work_cube(X, n_comp=16, seed=seed)


    coords = np.argwhere(gt == class_id)
    if len(coords) == 0:
        raise ValueError(f"class {class_id} has no pixels")

    rng = np.random.default_rng(seed + class_id * 1000)
    n_train = max(1, int(len(coords) * train_ratio))
    sel = rng.choice(len(coords), size=n_train, replace=False)
    train_yx = coords[sel]

    train_spec = X_work[train_yx[:, 0], train_yx[:, 1], :]
    d_train_raw = train_spec.mean(axis=0)

    sam_train = SAM(train_spec, d_train_raw)
    thr = np.percentile(sam_train, sam_quantile)

    keep_mask = sam_train <= thr
    if keep_mask.sum() < min_train_keep:
        idx = np.argsort(sam_train)[:min_train_keep]
        keep_mask[:] = False
        keep_mask[idx] = True

    train_yx_clean = train_yx[keep_mask]
    train_spec_clean = train_spec[keep_mask]

    d = train_spec_clean.mean(axis=0)


    score_g, _, _ = wbs_cem_score(
        X_work, d, weight_mode="SAM", eps=1e-3, use_double=True
    )
    score_g_n = normalize01(score_g)


    boxes = []
    for (y, x) in train_yx_clean:
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(W - 1, x + radius)
        y2 = min(H - 1, y + radius)
        boxes.append((x1, y1, x2, y2))

    boxes = np.array(boxes)
    n = len(boxes)

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def overlap(b1, b2):
        return not (
            b1[2] < b2[0] or b2[2] < b1[0] or
            b1[3] < b2[1] or b2[3] < b1[1]
        )

    for i in range(n):
        for j in range(i + 1, n):
            if overlap(boxes[i], boxes[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    roi_bboxes = []
    for idxs in groups.values():
        b = boxes[idxs]
        x1 = int(b[:, 0].min())
        y1 = int(b[:, 1].min())
        x2 = int(b[:, 2].max())
        y2 = int(b[:, 3].max())
        roi_bboxes.append((x1, y1, x2, y2))

    tecem = score_g_n.copy()

    for (x1, y1, x2, y2) in roi_bboxes:

        X_roi = X_work[y1:y2 + 1, x1:x2 + 1, :]

        score_l, _, _ = wbs_cem_score(
            X_roi, d, weight_mode="SAM", eps=1e-3, use_double=True
        )

        score_l_n = normalize01(score_l)

        if fuse_mode == "blend":
            base = tecem[y1:y2 + 1, x1:x2 + 1]
            tecem[y1:y2 + 1, x1:x2 + 1] = (
                alpha * score_l_n + (1 - alpha) * base
            )
        else:
            tecem[y1:y2 + 1, x1:x2 + 1] = np.maximum(
                tecem[y1:y2 + 1, x1:x2 + 1],
                score_l_n,
            )

    return tecem.astype(np.float32)

def tecem_all_classes(X, gt, train_ratio=0.01, seed=0, class_ids=None, **kwargs):
    if class_ids is None:
        class_ids = list(range(1, int(gt.max()) + 1))

    maps = {}
    for cid in class_ids:
        maps[cid] = tecem_for_class(
            X,
            gt,
            cid,
            train_ratio=train_ratio,
            seed=seed,
            **kwargs,
        )
    return maps
