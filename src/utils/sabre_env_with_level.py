from typing import Any, SupportsFloat
from copy import deepcopy
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
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.circuit.random import random_circuit

from .coupling_map_list import coupling_map_list


class LayoutSpace(gym.spaces.Space):
    def __init__(self) -> None:
        """A custom space for the layout for the ibm qiskit environment."""
        super().__init__()  # Initialize the base class

    def sample(self, mask: Any | None = None, probability: int | None = None) -> Layout:
        return Layout({Qubit(i): i for i in range(5)})  # Example layout with 5 qubits

    def contains(self, x: Any) -> bool:
        if isinstance(x, Layout):
            return True
        return False


class SabreSwapEnv(gym.Env):
    def __init__(
        self,
        n_graph: int = 10,
    ) -> None:
        self._n_graph = n_graph
        self._swap_singleton = SwapGate()

        self.observation_space = gym.spaces.Dict(
            {
                "swap_candidate": gym.spaces.Sequence(
                    space=gym.spaces.Box(low=0, high=133, shape=(2,), dtype=np.int32)
                ),
                "front_layer": gym.spaces.Sequence(
                    space=gym.spaces.Tuple(
                        (
                            gym.spaces.Discrete(133),  # first qubit
                            gym.spaces.Discrete(133),  # second qubit
                        )
                    )
                ),
                "sabre_dag": gym.spaces.Graph(
                    node_space=gym.spaces.Box(
                        low=0,
                        high=133,
                        shape=(3,),
                        dtype=np.int32,  # first, second, front_Layer
                    ),
                    edge_space=gym.spaces.Discrete(133),  # Wire
                ),
                "current_layout": LayoutSpace(),
            }
        )
        self.action_space = gym.spaces.Box(low=0, high=133, shape=(2,), dtype=np.int32)
        return

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment to a new circuit and initialize the SABRE algorithm."""
        super().reset(seed=seed, options=options)

        # setting up the circuit
        if options is None:
            options = {"qubit_range": (5, 10), "depth_range": (1, 5)}
        if options["qubit_range"][0] > options["qubit_range"][1]:
            raise ValueError(
                "qubit_range[0] must be less than or equal to qubit_range[1]"
            )
        if options["depth_range"][0] > options["depth_range"][1]:
            raise ValueError(
                "depth_range[0] must be less than or equal to depth_range[1]"
            )
        self._coupling_map: CouplingMap = random.choice(
            list(
                filter(
                    lambda x: len(x.physical_qubits) >= options["qubit_range"][0],
                    coupling_map_list,
                )
            )
        )
        self._distance_matrix = self._coupling_map.distance_matrix
        self._circuit = random_circuit(
            num_qubits=random.randint(
                options["qubit_range"][0],
                min(options["qubit_range"][1], len(self._coupling_map.physical_qubits)),
            ),
            depth=random.randint(*options["depth_range"]),
            max_operands=2,
            measure=False,
            seed=seed,
        )

        self._swap_candidate = []
        self._init_sabre(self._circuit)
        self._update_front_layer()
        return (
            {
                "swap_candidate": self._swap_candidate,
                "front_layer": self._convert_front_layer(),
                "sabre_dag": self._convert_digraph(),
                "current_layout": self._layout,
            },
            {
                "distance_matrix": self._distance_matrix,
                "coupling_map": self._coupling_map,
                "circuit": self._circuit,
                "result_circuit": dag_to_circuit(self._dest_dag),
            },
        )

    def step(
        self, action: tuple[int, int]
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Perform a step in the environment by applying a swap action."""
        isTruncated = self._apply_swap(action)

        executable_gate_number, isTerminated = self._update_front_layer()
        return (
            {
                "swap_candidate": self._swap_candidate,
                "front_layer": self._convert_front_layer(),
                "sabre_dag": self._convert_digraph(),
                "current_layout": self._layout,
            },
            self.reward(
                executable_gate_number, self._swap_depth, isTerminated, isTruncated
            ),
            isTerminated,
            isTruncated,
            {
                "distance_matrix": self._distance_matrix,
                "coupling_map": self._coupling_map,
                "circuit": self._circuit,
                "result_circuit": dag_to_circuit(self._dest_dag),
            },
        )

    def render(self):
        """Draw the destination DAG circuit."""
        return self._dest_dag.draw()

    def reward(
        self,
        executable_gate_number: int,
        swap_layer: int,
        isTerminated: bool,
        isTruncated: bool,
    ) -> float:
        """
        Calculate the reward based on the number of executable gates and swap layers.
        """
        return (
            -0.1
            + 0.2 * executable_gate_number
            - 0.05 * swap_layer
            + (5 if isTerminated else 0)
            - (10 if isTruncated else 0)
        )

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
        if len(self._dag.cregs) > 0:
            self._dest_dag.add_creg(
                self._dag.cregs["c"]
            )  # add classical register, since the layout only needs the qubit register, add creg after making the layout
        for start_node in self._dag.input_map.values():
            self._apply_1_qubit_successors(start_node)

        # Set up the initial state
        self._front_layer = self._sabre_dag.front_layer()
        self._swap_depth = 0
        return

    def _update_front_layer(self) -> tuple[int, bool]:
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
                    self._apply_1_qubit_successors(dag_node)

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

        self._swap_candidate = []
        if isTerminated:
            return 0, isTerminated

        # update the swap candidate list
        for node in self._front_layer:
            q1, q2 = (
                current_layout[node.qargs[0]],
                current_layout[node.qargs[1]],
            )
            for nq in self._coupling_map.neighbors(q1):
                self._swap_candidate.append((q1, nq))
            for nq in self._coupling_map.neighbors(q2):
                self._swap_candidate.append((q2, nq))

        return executable_gate_number, isTerminated

    def _apply_swap(self, action: tuple[int, int]) -> bool:
        # Check if the action is valid
        if action not in self._swap_candidate:
            return True

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
        return False

    def _convert_digraph(self):
        G = nx.DiGraph()
        count = 0
        process_nodes = deepcopy(self._front_layer)
        while len(process_nodes) > 0 and count < self._n_graph:
            node = process_nodes.pop(0)
            if not isinstance(node, DAGOpNode) or node._node_id in G.nodes:
                continue
            process_nodes += self._sabre_dag.successors(node)
            G.add_node(
                node._node_id,
                first=node.qargs[0],
                second=node.qargs[1],
                front_layer=True if count < len(self._front_layer) else False,
            )
            if count >= len(self._front_layer):
                for pred in self._sabre_dag.predecessors(node):
                    if isinstance(pred, DAGOpNode) and pred._node_id in G.nodes:
                        wire = list(filter(lambda x: x in node.qargs, pred.qargs))
                        G.add_edge(pred._node_id, node._node_id, wire=wire[0])
            count += 1
        return G

    def _convert_front_layer(self):
        """Convert the front layer to a list of tuples."""
        return [
            (node.qargs[0], node.qargs[1])
            for node in self._front_layer
            if isinstance(node, DAGOpNode)
        ]

    def _apply_1_qubit_successors(
        self,
        node: Any,
    ) -> None:
        """Apply all the 1 qubit successors of the node to the dest_dag."""
        if not isinstance(node, DAGOpNode):
            return
        successors = self._dag.successors(node)
        layout = self._layout.get_virtual_bits()
        for successor in successors:
            if isinstance(successor, DAGOpNode) and successor.op.num_qubits == 1:
                self._dest_dag.apply_operation_back(
                    successor.op,
                    (self._dest_dag.qregs["q"][layout[successor.qargs[0]]],),
                )
                self._apply_1_qubit_successors(successor)
                self._dag.remove_op_node(successor)
