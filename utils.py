import numpy as np


def hierarchy_to_lca(hierarchy: np.ndarray) -> np.ndarray:
    """
    Converts a hierarchy to a Least Common Ancestor (LCA) matrix.

    The LCA matrix is a square matrix where each element (i, j) represents
    the level of the least common ancestor for classes i and j.

    Args:
        hierarchy (np.array or torch.Tensor): A matrix where each row represents
            the ancestor hierarchy of a class.

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
        lca (np.array or torch.Tensor): A square matrix where each element (i, j)
            represents the level of the least common ancestor for classes i and j.

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
