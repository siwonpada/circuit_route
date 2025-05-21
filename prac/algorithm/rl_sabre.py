import torch.nn as nn


from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler import CouplingMap
from qiskit.circuit import Qubit

from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from copy import deepcopy


class RLSabreSwap(TransformationPass):
    """Original Sabre Swap pass."""

    def __init__(self, coupling_map: CouplingMap):
        super().__init__()
        self.coupling_map = coupling_map
        if coupling_map is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")

    def run(self, dag: DAGCircuit, training: bool = False):
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

            # if there is no executable gate, we need to swap the qubits
            else:
                for node in front_layer:
                    swap_candidate_list = []
                    q1, q2 = (
                        current_layout[node.qargs[0]],
                        current_layout[node.qargs[1]],
                    )
                    for nq in self.coupling_map.neighbors(q1):
                        swap_candidate_list.append((q1, nq))
                    for nq in self.coupling_map.neighbors(q2):
                        swap_candidate_list.append((q2, nq))
                    # Using RL, we can find the best swap, input:

        return dest_dag


# util function to apply the 1 qubit successors of the node
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


class RLSwap(nn.Module):
    def __init__(
        self,
        n_lookahead_node: int,
    ) -> None:
        super(RLSwap, self).__init__()

        # for accepting dist_matrix
        self.dist_matrix_feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x is tuple of (swap_candidate, sabre_dag, current_layout.get_virtual_bits(), dist_matrix)
        # swap_candidate is a list of tuples (q1, q2)

        # this RL model is used to find the best swap candidate

        return x


def Reward():
    # this function is used to calculate the reward of the RL model
    # the reward is calculated based on the distance between the qubits after the swap and the distance matrix
    # Since minimize the number of swap gates and the number of swap gate layers, and by the outline of the sabre swap algorithm, we don't need to use the distance matrix, just calculate the number of swap gates and the number of swap gate layers.

    pass
