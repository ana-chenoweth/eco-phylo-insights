import math
import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceMatrix


"""Jukes - Cantor distance calculator for two sequences"""
def jukes_cantor_distance(seq1, seq2):
    assert len(seq1) == len(seq2), "The sequences must have the same length"
    differences = sum(1 for a, b in zip(seq1, seq2) if a != b and a != '-' and b != '-')
    valid_sites = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
    if valid_sites == 0:
        return 0.0
    p = differences / valid_sites
    if p >= 0.75:
        return float('inf')  #Cannot calculate JC if p>= 0.75
    return -3/4 * math.log(1 - (4/3)*p)

def is_transition(a, b):
    transitions = [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]
    return (a, b) in transitions

"""Kimura distance calculator for two sequences"""
def kimura_distance(seq1, seq2):
    assert len(seq1) == len(seq2), "The sequences must have the same length"
    transitions = 0
    transversions = 0
    valid_sites = 0
    
    for a, b in zip(seq1.upper(), seq2.upper()):
        if a in "ACGT" and b in "ACGT":
            valid_sites += 1
            if a != b:
                if is_transition(a, b):
                    transitions += 1
                else:
                    transversions += 1

    if valid_sites == 0:
        return 0.0

    P = transitions / valid_sites
    Q = transversions / valid_sites

    if (1 - 2*P - Q) <= 0 or (1 - 2*Q) <= 0:
        return float('inf')  #Can't calculate Kimura in this case

    try:
        distance = -0.5 * math.log(1 - 2*P - Q) - 0.25 * math.log(1 - 2*Q)
    except ValueError:
        distance = float('inf')
    return distance



def compute_distance_matrix(alignment, distance_function):
    ids = [record.id for record in alignment]
    n = len(alignment)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                d = distance_function(str(alignment[i].seq), str(alignment[j].seq))
                matrix[i][j] = d
                matrix[j][i] = d  # symetric
    return ids, matrix



def clean_distance_matrix(matrix, replacement_value=None):
    """Replace the negative distances, inf or nan by a valid value."""
    cleaned = np.copy(matrix)
    
    if replacement_value is None:
        #If it's not specified, we take the maximum finite value in the matrix
        finite_values = cleaned[np.isfinite(cleaned) & (cleaned >= 0)]
        if len(finite_values) == 0:
            replacement_value = 1.0
        else:
            replacement_value = np.max(finite_values)

    n = cleaned.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                if not np.isfinite(cleaned[i][j]) or cleaned[i][j] < 0:
                    cleaned[i][j] = replacement_value
            else:
                cleaned[i][j] = 0.0  # Diagonal must always be zero

    return cleaned


def matrix_to_biopython(ids, np_matrix):
    n = len(ids)
    
    if np_matrix.shape != (n, n):
        raise ValueError("Matrix shape does not match number of IDs")

    #Build lower triangle including the diagonal
    lower_triangle = []
    for i in range(n):
        row = [float(np_matrix[i][j]) for j in range(i + 1)]
        lower_triangle.append(row)

    #Verify that the number of rows matches n
    if len(lower_triangle) != n:
        raise ValueError("Matrix should have n rows")

    for i, row in enumerate(lower_triangle):
        if len(row) != i + 1:
            raise ValueError(f"Row {i} length {len(row)} != {i+1}")

    return DistanceMatrix(names=ids, matrix=lower_triangle)
