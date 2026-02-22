import numpy as np


def test_build_view_augmented_split_repeats_labels_and_flattens_views() -> None:
    from mezzanine.recipes.crystal_spacegroup_distill import _build_view_augmented_split

    K, N, d = 3, 2, 4
    X_views = np.zeros((K, N, d), dtype=np.float32)
    for k in range(K):
        for n in range(N):
            X_views[k, n, :] = k * 100.0 + n * 10.0 + np.arange(d, dtype=np.float32)

    y_soft = np.array([[0.5], [1.5]], dtype=np.float32)
    y_hard = np.array([[10.0], [20.0]], dtype=np.float32)

    idx = np.array([1], dtype=np.int64)
    X_out, y_soft_out, y_hard_out = _build_view_augmented_split(X_views, y_soft, y_hard, idx)

    assert X_out.shape == (K * 1, d)
    assert y_soft_out.shape == (K * 1, 1)
    assert y_hard_out.shape == (K * 1, 1)

    assert np.allclose(X_out[0], X_views[0, 1])
    assert np.allclose(X_out[1], X_views[1, 1])
    assert np.allclose(X_out[2], X_views[2, 1])

    assert np.allclose(y_soft_out[:, 0], y_soft[1, 0])
    assert np.allclose(y_hard_out[:, 0], y_hard[1, 0])

