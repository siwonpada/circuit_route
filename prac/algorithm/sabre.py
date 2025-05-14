from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit import Qubit

from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from copy import deepcopy

from algorithm.heuristic import heuristic


class OriginalSabreSwap(TransformationPass):
    """Original Sabre Swap pass."""

    def __init__(self, coupling_map: CouplingMap):
        super().__init__()
        self.coupling_map = coupling_map
        if coupling_map is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")

    def run(self, dag: DAGCircuit):
        # We assume that the layout is already done by the user.

        if self.coupling_map is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) != self.coupling_map.size():
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if self.coupling_map.distance_matrix is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")
        # add more constraints if needed

        # Preprocessing
        sabre_dag = deepcopy(dag)
        sabre_dag.name = "sabre_swap"
        delete_nodes = set(sabre_dag.op_nodes()) - set(sabre_dag.two_qubit_ops())
        for node in delete_nodes:
            sabre_dag.remove_op_node(node)

        self.dist_matrix = self.coupling_map.distance_matrix

        dest_dag = DAGCircuit()
        dest_dag.add_qreg(dag.qregs["q"])
        layout = Layout(
            {
                dagInNode.wire: pq._index
                for pq, dagInNode in dag.input_map.items()
                if isinstance(pq, Qubit)
            }
        )  # this is the mapping pi of the paper
        dest_dag.add_creg(
            dag.cregs["c"]
        )  # add classical register, since the layout only needs the qubit register, add creg after making the layout

        # starting point of the sabre swap algorithm
        front_layer = sabre_dag.front_layer()
        decay_parameter = [0.001] * self.coupling_map.size()
        swap_singleton = SwapGate("sabre_swap")
        while len(front_layer) > 0:
            execute_gate_list = []
            current_layout = layout.get_virtual_bits()

            # find the executable gates
            for node in front_layer:
                q1, q2 = (
                    current_layout[node.qargs[0]],
                    current_layout[node.qargs[1]],
                )
                if self.coupling_map.distance(q1, q2) == 1:
                    execute_gate_list.append(node)

            # append the successors of the executable gates to the front layer if their predecessors are all executable
            if len(execute_gate_list) != 0:
                # actually, we choose one swap gate from all the neighbor swap gates however, in the reference code (https://github.com/Kaustuvi/quantum-qubit-mapping/blob/master/quantum_qubit_mapping/sabre_tools/sabre.py) she make a swap gate for each front layer node
                for node in execute_gate_list:
                    successors = sabre_dag.successors(node)
                    sabre_dag.remove_op_node(node)
                    _apply_1_qubit_predecessors(node, dag, dest_dag)
                    dest_dag.apply_operation_back(
                        node.op,
                        (node.qargs[0], node.qargs[1]),
                    )

                    # actually, we just use the method front_layer() after removing the node
                    front_layer.remove(node)
                    for node in successors:
                        if not isinstance(node, DAGOpNode):
                            continue
                        f_flag = True
                        for predcessor in sabre_dag.predecessors(node):
                            if isinstance(predcessor, DAGOpNode):
                                f_flag = False
                        if f_flag:
                            front_layer.append(node)
                decay_parameter = [
                    0.001
                ] * self.coupling_map.size()  # reset the decay parameter

            # if there is no executable gate, we need to swap the qubits
            else:
                # need heuristic to find the best swap
                for node in front_layer:
                    heuristic_score = {}
                    swap_candidate_list = []
                    q1, q2 = (
                        current_layout[node.qargs[0]],
                        current_layout[node.qargs[1]],
                    )
                    for nq in self.coupling_map.neighbors(q1):
                        swap_candidate_list.append((q1, nq))
                    for nq in self.coupling_map.neighbors(q2):
                        swap_candidate_list.append((q2, nq))
                    for swap_candidate in swap_candidate_list:
                        temp_layout = deepcopy(layout)
                        temp_layout.swap(swap_candidate[0], swap_candidate[1])
                        heuristic_score[swap_candidate] = heuristic(
                            swap_candidate,
                            front_layer,
                            sabre_dag,
                            temp_layout.get_virtual_bits(),
                            self.dist_matrix,
                            decay_parameter,
                        )
                    min_score_gate = min(
                        heuristic_score.keys(), key=lambda x: heuristic_score[x]
                    )

                    current_physical_qubit = layout.get_physical_bits()
                    dest_dag.apply_operation_back(
                        swap_singleton,
                        (
                            current_physical_qubit[min_score_gate[0]],
                            current_physical_qubit[min_score_gate[1]],
                        ),
                    )  # apply swap gate to the dest_dag
                    layout.swap(min_score_gate[0], min_score_gate[1])
                    decay_parameter[min_score_gate[0]] += 0.001
                    decay_parameter[min_score_gate[1]] += 0.001

        return dest_dag


def _apply_1_qubit_predecessors(node, dag, dest_dag):
    """Apply all the 1 qubit predecessors of the node to the dest_dag."""
    predecessors = dag.predecessors(node)
    for predecessor in predecessors:
        if isinstance(predecessor, DAGOpNode):
            _apply_1_qubit_predecessors(predecessor, dag, dest_dag)
            dest_dag.apply_operation_back(
                predecessor.op, (predecessor.qargs[0],)
            )  # apply the 1 qubit gate to the dest_dag
            dag.remove_op_node(predecessor)  # remove the 1 qubit gate from the dag


if __name__ == "__main__":
    import qiskit.qasm2

    circuit = qiskit.qasm2.load("data/4gt4-v0_80.qasm")

    coupling_map = CouplingMap.from_grid(4, 4)
    sabre_swap = OriginalSabreSwap(coupling_map)
    dag = circuit_to_dag(circuit)
    sabre_swap.run(dag)

# SABRE를 구현하기 위해서 1 qubit gate는 일단 두고, 2 qubit gate만의 layer를 찾아야 한다. -> 이는 현재 dag에서 1 qubit gate를 모두 제거 함으로 가능하다.
# 그리고 2 qubit gate는 모두 swap을 통해서 mapping을 해줘야 한다. -> 이때 swap은 coupling map에 맞춰서 해야 한다. 이때, 휴리스틱은 따로 구현을 해 주어야 한다. 이는 일단 구현이 되어있는 github의 repo를 참고해서 알고리즘을 가져오자.
# 한번 swap을 했으면, 그에 따른 logical-physical qubit의 mapping을 업데이트 해 주어야 한다. 이를 위해서 우리는 주어지는 gate의 기준이 logical qubit임을 주의하여야 한다. 또한 distance를 구할 때는, logical qubit이 아니라 physical qubit을 기준으로 해야 한다. -> 이때, coupling map을 통해서 distance를 구할 수 있다
