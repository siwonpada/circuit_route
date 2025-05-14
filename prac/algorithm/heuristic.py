from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from typing import List


# Heuristic function for the sabre algorithm, be careful this algorithm is look-ahead algorithm in sabre paper
def heuristic(
    swap_qubits: tuple[int, int],
    front_layer: List,
    dag_circuit: DAGCircuit,
    initial_mapping: dict,
    distance_matrix,
    decay_parameter: List,
) -> float:
    extended_successors = create_extended_successors(front_layer, dag_circuit)
    if len(extended_successors) == 0:
        return max(decay_parameter[swap_qubits[0]], decay_parameter[swap_qubits[1]]) * (
            sum(
                [
                    distance_matrix[initial_mapping[node.qargs[0]]][
                        initial_mapping[node.qargs[1]]
                    ]
                    for node in front_layer
                ]
            )
            / len(front_layer)
            + 0.5
        )
    return max(decay_parameter[swap_qubits[0]], decay_parameter[swap_qubits[1]]) * (
        sum(
            [
                distance_matrix[initial_mapping[node.qargs[0]]][
                    initial_mapping[node.qargs[1]]
                ]
                for node in front_layer
            ]
        )
        / len(front_layer)
        + 0.5
        * sum(
            [
                distance_matrix[initial_mapping[node.qargs[0]]][
                    initial_mapping[node.qargs[1]]
                ]
                for node in extended_successors
            ]
        )
        / len(extended_successors)
    )


def create_extended_successors(F: List, dag_circuit: DAGCircuit) -> List:
    """
    Create extended successors for the given front layer.
    """
    extended_successors = []
    for node in F:
        for successor in dag_circuit.successors(node):
            if (
                isinstance(successor, DAGOpNode)
                and successor not in extended_successors
            ):
                extended_successors.append(successor)
    return extended_successors
