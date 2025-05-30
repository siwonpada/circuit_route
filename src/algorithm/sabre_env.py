from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
import random
import networkx as nx


from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
    TrivialLayout,
    ApplyLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
)
from qiskit.transpiler.layout import Layout
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGCircuit

from copy import deepcopy


class SabreSwapEnv(gym.Env):
    def __init__(
        self,
        circuits: list[QuantumCircuit],
        coupling_map: CouplingMap,
        n_graph: int = 10,
    ) -> None:
        if coupling_map is None:
            raise ValueError("SabreSwapEnv cannot run with coupling_map=None")
        self._coupling_map = coupling_map
        self._circuits = circuits
        self._n_graph = n_graph
        self._swap_singleton = SwapGate()
        max_num_qubits = coupling_map.size()

        self.observation_space = gym.spaces.Dict(
            {
                "swap_candidate": gym.spaces.Sequence(
                    space=gym.spaces.Box(
                        low=0, high=max_num_qubits, shape=(2,), dtype=np.int32
                    )
                ),
                "sabre_dag": gym.spaces.Graph(
                    node_space=gym.spaces.Box(
                        low=0,
                        high=max_num_qubits,
                        shape=(3,),
                        dtype=np.int32,  # first, second, front_Layer
                    ),
                    edge_space=gym.spaces.Discrete(max_num_qubits),  # Wire
                ),
                "current_layout": gym.spaces.Box(
                    low=0,
                    high=max_num_qubits,
                    shape=(max_num_qubits,),
                    dtype=np.int32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=0, high=max_num_qubits, shape=(2,), dtype=np.int32
        )
        return

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._circuit = random.choice(self._circuits)
        self._init_sabre(self._circuit)

        updated_result = self._update_front_layer()

        return (
            {
                "swap_candidate": updated_result[0],
                "sabre_dag": self._convert_digraph(),
                "current_layout": updated_result[1],
            },
            {"distance_matrix": self.dist_matrix},
        )

    def step(
        self, action: tuple[int, int]
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self._apply_swap(action)

        updated_result = self._update_front_layer()
        return (
            {
                "swap_candidate": updated_result[0],
                "sabre_dag": self._convert_digraph(),
                "current_layout": updated_result[1],
            },
            self.reward(updated_result[2], self._swap_depth),
            updated_result[3],
            False,
            {
                "distance_matrix": self.dist_matrix,
            },
        )

    def render(self):
        return self._dest_dag.draw()

    def reward(self, executable_gate_number: int, swap_layer: int) -> float:
        return -1 + 2 * executable_gate_number - 0.5 * swap_layer

    def _init_sabre(self, circuit: QuantumCircuit) -> None:
        # Set up Layout and Ancilla
        apply_ancilla_pass_manager = PassManager(
            [
                TrivialLayout(self._coupling_map),
                FullAncillaAllocation(self._coupling_map),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )
        self._processed_circuit = apply_ancilla_pass_manager.run(circuit)
        self._dag = circuit_to_dag(self._processed_circuit)

        # Set up sabre dag (it contains only 2 qubit gates)
        self._sabre_dag = deepcopy(self._dag)
        self._sabre_dag.name = "sabre_swap"
        delete_nodes = set(self._sabre_dag.op_nodes()) - set(
            self._sabre_dag.two_qubit_ops()
        )
        for node in delete_nodes:
            self._sabre_dag.remove_op_node(node)

        # Set up the distance matrix
        self.dist_matrix = self._coupling_map.distance_matrix

        # Set up the result dag for check the result
        self._dest_dag = DAGCircuit()
        self._dest_dag.add_qreg(self._dag.qregs["q"])
        self._layout = Layout(
            {
                dagInNode.wire: pq._index
                for pq, dagInNode in self._dag.input_map.items()
                if isinstance(pq, Qubit)
            }
        )  # this is the mapping pi of the paper
        self._dest_dag.add_creg(
            self._dag.cregs["c"]
        )  # add classical register, since the layout only needs the qubit register, add creg after making the layout
        for start_node in self._dag.input_map.values():
            _apply_1_qubit_successors(
                start_node, self._dag, self._dest_dag, self._layout.get_virtual_bits()
            )

        # Set up the initial state
        self._front_layer = self._sabre_dag.front_layer()
        self._swap_depth = 0
        return

    def _update_front_layer(self) -> tuple[list, dict, int, bool]:
        # update the front layer
        isTerminated = True
        executable_gate_number = 0
        current_layout = self._layout.get_virtual_bits()
        while len(self._front_layer) > 0:
            execute_gate_list = []
            current_layout = self._layout.get_virtual_bits()

            for node in self._front_layer:
                q1, q2 = (
                    current_layout[node.qargs[0]],
                    current_layout[node.qargs[1]],
                )
                if self._coupling_map.distance(q1, q2) == 1:
                    executable_gate_number += 1
                    execute_gate_list.append(node)

            if len(execute_gate_list) != 0:
                self._swap_depth = 0
                for node in execute_gate_list:
                    successors = self._sabre_dag.successors(node)
                    self._sabre_dag.remove_op_node(node)
                    dag_node = self._dag.node(node._node_id)
                    self._dest_dag.apply_operation_back(
                        node.op,
                        (
                            self._dest_dag.qregs["q"][current_layout[node.qargs[0]]],
                            self._dest_dag.qregs["q"][current_layout[node.qargs[1]]],
                        ),
                    )
                    _apply_1_qubit_successors(
                        dag_node, self._dag, self._dest_dag, current_layout
                    )

                    # actually, we just use the method front_layer() after removing the node
                    self._front_layer.remove(node)
                    for node in successors:
                        if not isinstance(node, DAGOpNode):
                            continue
                        f_flag = True
                        for predcessor in self._sabre_dag.predecessors(node):
                            if isinstance(predcessor, DAGOpNode):
                                f_flag = False
                        if f_flag:
                            self._front_layer.append(node)
            else:
                isTerminated = False
                break

        if isTerminated:
            return [(-1, -1)], current_layout, 0, isTerminated

        swap_candidate_list = []
        for node in self._front_layer:
            q1, q2 = (
                current_layout[node.qargs[0]],
                current_layout[node.qargs[1]],
            )
            for nq in self._coupling_map.neighbors(q1):
                swap_candidate_list.append((q1, nq))
            for nq in self._coupling_map.neighbors(q2):
                swap_candidate_list.append((q2, nq))

        return (
            swap_candidate_list,
            current_layout,
            executable_gate_number,
            isTerminated,
        )

    def _apply_swap(self, action: tuple[int, int]) -> None:
        # apply the swap operation
        before_swap_depth = self._dest_dag.depth()
        self._dest_dag.apply_operation_back(
            self._swap_singleton,
            (
                self._dest_dag.qregs["q"][action[0]],
                self._dest_dag.qregs["q"][action[1]],
            ),
        )
        self._layout.swap(action[0], action[1])
        self._swap_depth += self._dest_dag.depth() - before_swap_depth
        return

    def _convert_digraph(self):
        G = nx.DiGraph()
        count = 0
        process_nodes = deepcopy(self._front_layer)
        while len(process_nodes) > 0 and count < self._n_graph:
            node = process_nodes.pop(0)
            if not isinstance(node, DAGOpNode):
                continue
            process_nodes += self._sabre_dag.successors(node)
            G.add_node(
                node._node_id,
                first=node.qargs[0],
                second=node.qargs[1],
                front_layer=True if count < len(self._front_layer) else False,
            )
            for pred in self._sabre_dag.predecessors(node):
                if isinstance(pred, DAGOpNode):
                    wire = list(filter(lambda x: x in node.qargs, pred.qargs))
                    G.add_edge(pred._node_id, node._node_id, wire=wire[0])
            count += 1
        return G


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
