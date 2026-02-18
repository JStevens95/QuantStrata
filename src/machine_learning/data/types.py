"""
Deprecated data types — kept only for backward compatibility.

These classes have been replaced by:
    - ``sklearn.preprocessing.StandardScaler`` for normalisation
    - ``sklearn.model_selection.train_test_split`` for splitting
    - ``build_tf_dataset()`` (from ``data.dataset``) for tf.data pipelines
    - Plain ``(np.ndarray, np.ndarray)`` or ``SyntheticData`` containers

Migration:
    Old:  dataset = MLDataset(features=X, targets=y); train, test = dataset.split()
    New:  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
import warnings

warnings.warn(
    "machine_learning.data.types is deprecated. "
    "Use sklearn for splitting/scaling and build_tf_dataset() for pipelines.",
    DeprecationWarning,
    stacklevel=2,
)

__all__: list = []
