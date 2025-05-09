from qiskit.dagcircuit import DAGCircuit
from numpy import ndarray
from typing import List


def heuristic(
    candidate: List,
    dag_circuit: DAGCircuit,
    initial_mapping: dict,
    distance_matrix: ndarray,
    decay_parameter: List,
) -> float:
    return 0.0
