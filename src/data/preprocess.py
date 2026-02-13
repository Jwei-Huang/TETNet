import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=X.dtype)
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # prealloc maximum then filter
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=X.dtype)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.int32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex += 1
    patchesData = patchesData[:patchIndex]
    patchesLabels = patchesLabels[:patchIndex]
    if removeZeroLabels:
        mask = patchesLabels > 0
        patchesData = patchesData[mask]
        patchesLabels = patchesLabels[mask] - 1
    return patchesData, patchesLabels

def createImageCubes_2D(X, y, windowSize=5, removeZeroLabels=True):
    # X is (H,W) single-channel map -> output (N,win,win,1)
    X = X[..., np.newaxis]
    return createImageCubes(X, y, windowSize=windowSize, removeZeroLabels=removeZeroLabels)

def splitTrainTestSet(X, y, testRatio, randomState=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testRatio, random_state=randomState, stratify=y
    )
    return X_train, X_test, y_train, y_test

def splitDataset_Min(X, y, testRatio, randomState=1):
    """
    Per-class split that avoids errors on tiny classes.
    Keeps at least 2 samples in TRAIN if possible.
    """
    rng = np.random.default_rng(randomState)
    X = np.asarray(X); y = np.asarray(y)
    classes = np.unique(y)
    train_idx=[]; test_idx=[]
    for cls in classes:
        idx = np.where(y==cls)[0]
        idx = rng.permutation(idx)
        n = len(idx)
        n_test = int(round(n*testRatio))
        n_train = n - n_test
        if n_train < 2 and n >= 2:
            n_train = 2
            n_test = n - n_train
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    # global shuffle
    train_idx = np.array(train_idx); test_idx=np.array(test_idx)
    train_idx = train_idx[rng.permutation(len(train_idx))]
    test_idx = test_idx[rng.permutation(len(test_idx))]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
