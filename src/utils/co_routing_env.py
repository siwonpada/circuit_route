from typing import Any, SupportsFloat
from copy import deepcopy
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
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.random import random_circuit
from collections import deque

from .coupling_map_list import coupling_map_list


class CoRoutingEnv(gym.Env):
    def __init__(
        self,
        N: int = 10,
        start_level: int = 2,
        max_window_size: int = 100,
    ) -> None:
        self._N = N
        self._level = start_level
        self._swap_singleton = SwapGate()
        self._reset_failed = 0
        self._result_deque = deque(maxlen=max_window_size)

        self.observation_space = gym.spaces.Dict(
            {
                "nodes": gym.spaces.Box(
                    low=0,
                    high=self._N,
                    shape=(
                        133,
                        self._N,
                    ),
                    dtype=np.int32,
                ),
                "edge_links": gym.spaces.Box(low=0, high=132, shape=(300, 2)),
                "nodes_num": gym.spaces.Box(low=0, high=133, shape=(), dtype=np.int32),
            }
        )

        self.action_space = gym.spaces.MultiDiscrete(np.array([133, 133]))
        return

    def action_masks(self) -> np.ndarray:
        return np.array([True] * self._node_num + [False] * (133 - self._node_num))

    def set_level(self, level: int) -> None:
        if level < 1:
            raise ValueError("The level must be at least 3.")
        self._result_deque.clear()
        self._level = level

    def get_success_rate(self) -> float:
        """Calculate the success rate based on the results deque."""
        if len(self._result_deque) == 0:
            return 0.0
        return sum(self._result_deque) / len(self._result_deque)

    def get_level(self) -> int:
        """Get the current level of the environment."""
        return self._level

    def get_reset_failed(self) -> int:
        """Get the number of times the environment failed to reset."""
        return self._reset_failed

    def front_layer_size(self) -> int:
        """Get the size of the front layer."""
        return len(self._front_layer)

    def render_sabre(self):
        """Render the SABRE DAG circuit."""
        return self._sabre_dag.draw()

    def render_original(self):
        """Render the original DAG circuit."""
        return self._dag.draw()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict[str, Any]]:
        """Reset the environment to a new circuit and initialize the SABRE algorithm."""
        super().reset(seed=seed, options=options)
        self._coupling_map: CouplingMap = random.choice(coupling_map_list)
        self._edge_links = np.array(self._coupling_map.get_edges(), dtype=np.int64)
        self._node_num = self._coupling_map.size()
        self._success = False
        self._truncated = False
        Terminated = True
        i = 0
        while Terminated:
            i += 1
            if i % 1000 == 0:
                print("Resetting environment, attempt:", i)
            self._circuit = random_circuit(
                num_qubits=random.randint(
                    3,
                    min(self._level + 3, len(self._coupling_map.physical_qubits)),
                ),
                depth=self._level + 1,
                max_operands=2,
                measure=False,
                seed=seed,
                num_operand_distribution={1: 0, 2: 1},
            )
            self._init_sabre(self._circuit)
            _, Terminated = self._update_front_layer()
        self._reset_failed = i
        return (
            self._unpack_state(),
            {},
        )

    def step(
        self, action: tuple[int, int]
    ) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Perform a step in the environment by applying a swap action."""
        isTruncated = self._apply_swap(action)
        executable_gate_number, isTerminated = self._update_front_layer()
        if isTruncated:
            self._result_deque.append(False)
        if isTerminated:
            self._result_deque.append(True)
        return (
            self._unpack_state(),
            self.reward(
                executable_gate_number, self._swap_depth, isTerminated, isTruncated
            ),
            isTerminated,
            False,
            {},
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

        # Set up the initial state
        self._front_layer = self._sabre_dag.front_layer()
        self._swap_depth = 0
        return

    def _update_front_layer(self) -> tuple[int, bool]:
        # update the front layer
        isTerminated = True
        executable_gate_number = 0
        while len(self._front_layer) > 0:
            execute_gate_list = []
            layout = self._layout.get_virtual_bits()

            # check if the node in the front layer can be executed
            for node in self._front_layer:
                q1, q2 = (
                    layout[node.qargs[0]],
                    layout[node.qargs[1]],
                )
                if self._coupling_map.distance(q1, q2) == 1:
                    executable_gate_number += 1
                    execute_gate_list.append(node)

            if len(execute_gate_list) != 0:
                self._swap_depth = 0
                # apply the executable gates
                for node in execute_gate_list:
                    self._sabre_dag.remove_op_node(node)
                    self._dest_dag.apply_operation_back(
                        node.op,
                        (
                            self._dest_dag.qregs["q"][layout[node.qargs[0]]],
                            self._dest_dag.qregs["q"][layout[node.qargs[1]]],
                        ),
                    )
                    self._front_layer = self._sabre_dag.front_layer()
            else:
                isTerminated = False
                break

        if isTerminated:
            self._success = True
            return executable_gate_number, isTerminated

        return executable_gate_number, isTerminated

    def _apply_swap(self, action: tuple[int, int]) -> bool:
        # Check if the action is valid
        if (
            action[0] >= self._node_num
            or action[1] >= self._node_num
            or self._coupling_map.distance(action[0], action[1]) != 1
        ):
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

    def _unpack_state(self) -> dict:
        """Unpack the state into a Graph instance."""
        nodes = np.zeros((self._node_num, self._N), dtype=np.int32)
        layers = self._sabre_dag.layers()
        layout = self._layout.get_virtual_bits()
        counter = 0
        for i, layer in enumerate(layers):
            for node in layer["graph"].op_nodes():
                if counter >= self._N:
                    break
                nodes[layout[node.qargs[0]], counter] = i + 1
                nodes[layout[node.qargs[1]], counter] = i + 1
                counter += 1
        test = np.pad(
            nodes,
            ((0, 133 - nodes.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return {
            "nodes": test,
            "edge_links": np.pad(
                self._edge_links,
                ((0, 300 - self._edge_links.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            ),
            "nodes_num": self._node_num,
        }
