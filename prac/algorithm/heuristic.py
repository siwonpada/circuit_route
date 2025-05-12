from qiskit.dagcircuit import DAGCircuit
from numpy import ndarray
from typing import List

# TODO: implement the decay parameter in sabre algorithm to use the decay parameter


def heuristic(
    candidate: List,
    dag_circuit: DAGCircuit,
    initial_mapping: dict,
    distance_matrix: ndarray,
    decay_parameter: List,
) -> float:
    return 0.0
