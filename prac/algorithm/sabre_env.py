from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
import random


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
        self, circuits: list[QuantumCircuit], coupling_map: CouplingMap
    ) -> None:
        if coupling_map is None:
            raise ValueError("SabreSwapEnv cannot run with coupling_map=None")
        self._coupling_map = coupling_map
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
                        low=0, high=max_num_qubits, shape=(2,), dtype=np.int32
                    ),
                    edge_space=gym.spaces.Discrete(max_num_qubits),
                ),
                "current_layout": gym.spaces.Box(
                    low=0,
                    high=max_num_qubits,
                    shape=(max_num_qubits,),
                    dtype=np.int32,
                ),
                "dist_matrix": gym.spaces.Box(
                    low=0,
                    high=max_num_qubits,
                    shape=(max_num_qubits, max_num_qubits),
                    dtype=np.int32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=0, high=max_num_qubits, shape=(2,), dtype=np.int32
        )
        self._circuits = circuits
        self._circuit = random.choice(self._circuits)
        self._init_sabre(self._circuit)
        self._swap_singleton = SwapGate()
        return

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._circuit = random.choice(self._circuits)
        self._init_sabre(self._circuit)
        return super().reset(seed=seed, options=options)

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

    def step(
        self, action: tuple[int, int]
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
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
            {
                "swap_candidate": swap_candidate_list,
                "sabre_dag": self._sabre_dag,
                "current_layout": current_layout,
                "dist_matrix": self.dist_matrix,
            },
            self.reward(executable_gate_number, self._swap_depth),
            isTerminated,
            False,
            {
                "dest_dag": self._dest_dag,
                "sabre_dag": self._sabre_dag,
                "current_layout": current_layout,
            },
        )

    def render(self):
        return self._dest_dag.draw()

    def reward(self, executable_gate_number: int, swap_layer: int) -> float:
        return -0.1 + 0.2 * executable_gate_number - 0.05 * swap_layer


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
