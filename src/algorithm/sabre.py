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
        for start_node in dag.input_map.values():
            _apply_1_qubit_successors(
                start_node, dag, dest_dag, layout.get_virtual_bits()
            )

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
                    dag_node = dag.node(node._node_id)
                    dest_dag.apply_operation_back(
                        node.op,
                        (
                            dest_dag.qregs["q"][current_layout[node.qargs[0]]],
                            dest_dag.qregs["q"][current_layout[node.qargs[1]]],
                        ),
                    )
                    _apply_1_qubit_successors(dag_node, dag, dest_dag, current_layout)

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

                    dest_dag.apply_operation_back(
                        swap_singleton,
                        (
                            dest_dag.qregs["q"][min_score_gate[0]],
                            dest_dag.qregs["q"][min_score_gate[1]],
                        ),
                    )  # apply swap gate to the dest_dag
                    layout.swap(min_score_gate[0], min_score_gate[1])
                    decay_parameter[min_score_gate[0]] += 0.001
                    decay_parameter[min_score_gate[1]] += 0.001

        return dest_dag


def _apply_1_qubit_successors(node, dag, dest_dag, layout):
    """Apply all the 1 qubit successors of the node to the dest_dag."""
    successors = dag.successors(node)
    for successor in successors:
        if isinstance(successor, DAGOpNode) and successor.op.num_qubits == 1:
            dest_dag.apply_operation_back(
                successor.op,
                (dest_dag.qregs["q"][layout[successor.qargs[0]]],),
            )
            _apply_1_qubit_successors(successor, dag, dest_dag, layout)
            dag.remove_op_node(successor)


if __name__ == "__main__":
    import qiskit.qasm2

    circuit = qiskit.qasm2.load("data/4mod5-v1_23.qasm")

    coupling_map = CouplingMap.from_grid(4, 4)
    sabre_swap = OriginalSabreSwap(coupling_map)
    dag = circuit_to_dag(circuit)
    sabre_swap.run(dag)
