import matplotlib.pyplot as plt
import numpy as np

# HIERARCHY ############################################################################


def hierarchy_to_lca(hierarchy: np.ndarray) -> np.ndarray:
    """
    Converts a hierarchy to a Least Common Ancestor (LCA) matrix.

    The LCA matrix is a square matrix where each element (i, j) represents
    the level of the least common ancestor for classes i and j.

    Args:
        hierarchy (np.array): A matrix where each row represents the ancestor
            hierarchy of a class.

    Returns:
        A square numpy array containing the LCA matrix.
    """
    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = hierarchy.shape

    lca = np.full((C, C), L, dtype=int)

    for level in hierarchy:
        for row, coarse in zip(lca, level):
            for index, value in enumerate(level):
                if coarse == value:
                    row[index] -= 1
    return lca


def lca_to_hierarchy(lca: np.ndarray) -> np.ndarray:
    """
    Converts a Least Common Ancestor (LCA) matrix to a hierarchy matrix.

    The hierarchy matrix is a matrix where each row represents the ancestor hierarchy
    of a class.

    Args:
        lca (np.array): A square matrix where each element (i, j) represents
            the level of the least common ancestor for classes i and j.

    Returns:
        A numpy array containing the hierarchy matrix.
    """
    # Make a copy to avoid inplace operations
    lca = np.array(lca, dtype=int)

    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = lca.max(), len(lca)

    hierarchy = -np.ones((L, C), dtype=int)

    for level in range(L):
        # Find all siblings at `level`,
        # reverse to be consistence at level 0
        siblings = np.unique(lca == level, axis=0)[::-1]

        # Generate labeler
        labeler = np.arange(len(siblings), dtype=int)

        # Apply labels to siblings with labeler
        labels = labeler @ siblings

        # Add labels to hierarchy
        hierarchy[level] = labels

        # Update lca for next iteration
        lca[lca == level] += 1

    return hierarchy


# PLOTS ################################################################################


def plot_enc(encodings: np.ndarray, hierarchy: np.ndarray | None = None) -> None:
    """
    Plots the encoding matrix with optional hierarchical sorting.

    Args:
        encodings (np.ndarray): The encoding matrix to be plotted.
        hierarchy (np.ndarray | None): A hierarchy matrix representing the
            ancestor hierarchy of the classes. If provided, the encodings will
            be sorted according to the hierarchy before plotting. Default is None.

    Returns:
        None
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 3))

    if hierarchy is not None:
        # Use hierarchy to sort encoding
        idx = np.lexsort(hierarchy)
        encodings = encodings[idx, :][:, idx]

        # Rearrange ticks according to the hierarchical sorting
        ticks = np.arange(len(hierarchy[0]))
        ax.set_xticks(ticks[::20])
        ax.set_yticks(ticks[::20])
        ax.set_xticklabels(labels=idx[::20])
        ax.set_yticklabels(labels=idx[::20])

    ax.imshow(encodings)
    fig.show()


# METRICS ##############################################################################


def accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    corrects = np.any(
        hierarchy[level][top_k_preds] == hierarchy[level][labels],
        axis=1,
    )
    return np.mean(corrects)


def error_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    wrongs = np.all(
        hierarchy[level][top_k_preds] != hierarchy[level][labels],
        axis=1,
    )
    return np.mean(wrongs)


def hier_dist_mistake(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    wrongs = np.all(
        hierarchy[level][top_k_preds] != hierarchy[level][labels],
        axis=1,
    )
    lca = hierarchy_to_lca(hierarchy[level:])
    lca_heights = lca[top_k_preds[wrongs], labels[wrongs]]
    return np.mean(lca_heights)


def hier_dist(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    lca_heights = hierarchy_to_lca(hierarchy[level:])[top_k_preds, labels]
    return np.mean(lca_heights)
