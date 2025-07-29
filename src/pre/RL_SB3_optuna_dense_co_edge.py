# %%
from qiskit.transpiler import CouplingMap


def create_Y(n) -> CouplingMap:
    """Create a Y-shaped coupling map with n qubits."""
    if not n % 2 == 0:
        raise ValueError("n must be odd for a Y-shaped coupling map.")
    result = CouplingMap().from_line(n - 1)
    result.add_physical_qubit(n - 1)
    result.add_edge(n - 1, n // 2)
    return result


def create_T(n) -> CouplingMap:
    """Create a T-shaped coupling map with n qubits."""
    result = CouplingMap().from_line(n - 1)
    result.add_physical_qubit(n - 1)
    result.add_edge(n - 1, n - 3)
    return result


coupling_map_list = [
    *[CouplingMap.from_line(n) for n in range(4, 11)],
    CouplingMap.from_ring(12),
    *[create_Y(2 * n) for n in range(2, 6)],
    *[create_T(n) for n in range(5, 11)],
    CouplingMap(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [1, 4],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 5],
            [4, 1],
            [4, 7],
            [5, 3],
            [5, 8],
            [6, 7],
            [7, 4],
            [7, 6],
            [7, 10],
            [8, 5],
            [8, 9],
            [8, 11],
            [9, 8],
            [10, 7],
            [10, 12],
            [11, 8],
            [11, 14],
            [12, 10],
            [12, 13],
            [12, 15],
            [13, 12],
            [13, 14],
            [14, 11],
            [14, 13],
            [14, 16],
            [15, 12],
            [15, 18],
            [16, 14],
            [16, 19],
            [17, 18],
            [18, 15],
            [18, 17],
            [18, 21],
            [19, 16],
            [19, 20],
            [19, 22],
            [20, 19],
            [21, 18],
            [21, 23],
            [22, 19],
            [22, 25],
            [23, 21],
            [23, 24],
            [24, 23],
            [24, 25],
            [25, 22],
            [25, 24],
            [25, 26],
            [26, 25],
        ]
    ),
    CouplingMap(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [1, 4],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 5],
            [3, 30],
            [4, 1],
            [4, 7],
            [5, 3],
            [5, 8],
            [6, 7],
            [7, 4],
            [7, 6],
            [7, 10],
            [8, 5],
            [8, 9],
            [8, 11],
            [9, 8],
            [10, 7],
            [10, 12],
            [11, 8],
            [11, 14],
            [12, 10],
            [12, 13],
            [12, 15],
            [13, 12],
            [13, 14],
            [14, 11],
            [14, 13],
            [14, 16],
            [15, 12],
            [15, 18],
            [16, 14],
            [16, 19],
            [17, 18],
            [18, 15],
            [18, 17],
            [18, 21],
            [19, 16],
            [19, 20],
            [19, 22],
            [20, 19],
            [21, 18],
            [21, 23],
            [22, 19],
            [22, 25],
            [23, 21],
            [23, 24],
            [23, 27],
            [24, 23],
            [24, 25],
            [25, 22],
            [25, 24],
            [25, 26],
            [26, 25],
            [27, 23],
            [27, 28],
            [28, 27],
            [28, 29],
            [29, 28],
            [30, 3],
            [30, 31],
            [31, 30],
            [31, 32],
            [32, 31],
        ]
    ),
    CouplingMap(
        [
            [0, 1],
            [0, 10],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [4, 5],
            [4, 11],
            [5, 4],
            [5, 6],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [8, 7],
            [8, 9],
            [8, 12],
            [9, 8],
            [10, 0],
            [10, 13],
            [11, 4],
            [11, 17],
            [12, 8],
            [12, 21],
            [13, 10],
            [13, 14],
            [14, 13],
            [14, 15],
            [15, 14],
            [15, 16],
            [15, 24],
            [16, 15],
            [16, 17],
            [17, 11],
            [17, 16],
            [17, 18],
            [18, 17],
            [18, 19],
            [19, 18],
            [19, 20],
            [19, 25],
            [20, 19],
            [20, 21],
            [21, 12],
            [21, 20],
            [21, 22],
            [22, 21],
            [22, 23],
            [23, 22],
            [23, 26],
            [24, 15],
            [24, 29],
            [25, 19],
            [25, 33],
            [26, 23],
            [26, 37],
            [27, 28],
            [27, 38],
            [28, 27],
            [28, 29],
            [29, 24],
            [29, 28],
            [29, 30],
            [30, 29],
            [30, 31],
            [31, 30],
            [31, 32],
            [31, 39],
            [32, 31],
            [32, 33],
            [33, 25],
            [33, 32],
            [33, 34],
            [34, 33],
            [34, 35],
            [35, 34],
            [35, 36],
            [35, 40],
            [36, 35],
            [36, 37],
            [37, 26],
            [37, 36],
            [38, 27],
            [38, 41],
            [39, 31],
            [39, 45],
            [40, 35],
            [40, 49],
            [41, 38],
            [41, 42],
            [42, 41],
            [42, 43],
            [43, 42],
            [43, 44],
            [43, 52],
            [44, 43],
            [44, 45],
            [45, 39],
            [45, 44],
            [45, 46],
            [46, 45],
            [46, 47],
            [47, 46],
            [47, 48],
            [47, 53],
            [48, 47],
            [48, 49],
            [49, 40],
            [49, 48],
            [49, 50],
            [50, 49],
            [50, 51],
            [51, 50],
            [51, 54],
            [52, 43],
            [52, 56],
            [53, 47],
            [53, 60],
            [54, 51],
            [54, 64],
            [55, 56],
            [56, 52],
            [56, 55],
            [56, 57],
            [57, 56],
            [57, 58],
            [58, 57],
            [58, 59],
            [59, 58],
            [59, 60],
            [60, 53],
            [60, 59],
            [60, 61],
            [61, 60],
            [61, 62],
            [62, 61],
            [62, 63],
            [63, 62],
            [63, 64],
            [64, 54],
            [64, 63],
        ]
    ),
    CouplingMap(
        [
            [0, 1],
            [0, 15],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [4, 5],
            [4, 16],
            [5, 4],
            [5, 6],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [8, 7],
            [8, 9],
            [8, 17],
            [9, 8],
            [9, 10],
            [10, 9],
            [10, 11],
            [11, 10],
            [11, 12],
            [12, 11],
            [12, 13],
            [12, 18],
            [13, 12],
            [13, 14],
            [14, 13],
            [15, 0],
            [15, 19],
            [16, 4],
            [16, 23],
            [17, 8],
            [17, 27],
            [18, 12],
            [18, 31],
            [19, 15],
            [19, 20],
            [20, 19],
            [20, 21],
            [21, 20],
            [21, 22],
            [21, 34],
            [22, 21],
            [22, 23],
            [23, 16],
            [23, 22],
            [23, 24],
            [24, 23],
            [24, 25],
            [25, 24],
            [25, 26],
            [25, 35],
            [26, 25],
            [26, 27],
            [27, 17],
            [27, 26],
            [27, 28],
            [28, 27],
            [28, 29],
            [29, 28],
            [29, 30],
            [29, 36],
            [30, 29],
            [30, 31],
            [31, 18],
            [31, 30],
            [31, 32],
            [32, 31],
            [32, 33],
            [33, 32],
            [33, 37],
            [34, 21],
            [34, 40],
            [35, 25],
            [35, 44],
            [36, 29],
            [36, 48],
            [37, 33],
            [37, 52],
            [38, 39],
            [38, 53],
            [39, 38],
            [39, 40],
            [40, 34],
            [40, 39],
            [40, 41],
            [41, 40],
            [41, 42],
            [42, 41],
            [42, 43],
            [42, 54],
            [43, 42],
            [43, 44],
            [44, 35],
            [44, 43],
            [44, 45],
            [45, 44],
            [45, 46],
            [46, 45],
            [46, 47],
            [46, 55],
            [47, 46],
            [47, 48],
            [48, 36],
            [48, 47],
            [48, 49],
            [49, 48],
            [49, 50],
            [50, 49],
            [50, 51],
            [50, 56],
            [51, 50],
            [51, 52],
            [52, 37],
            [52, 51],
            [53, 38],
            [53, 57],
            [54, 42],
            [54, 61],
            [55, 46],
            [55, 65],
            [56, 50],
            [56, 69],
            [57, 53],
            [57, 58],
            [58, 57],
            [58, 59],
            [59, 58],
            [59, 60],
            [59, 72],
            [60, 59],
            [60, 61],
            [61, 54],
            [61, 60],
            [61, 62],
            [62, 61],
            [62, 63],
            [63, 62],
            [63, 64],
            [63, 73],
            [64, 63],
            [64, 65],
            [65, 55],
            [65, 64],
            [65, 66],
            [66, 65],
            [66, 67],
            [67, 66],
            [67, 68],
            [67, 74],
            [68, 67],
            [68, 69],
            [69, 56],
            [69, 68],
            [69, 70],
            [70, 69],
            [70, 71],
            [71, 70],
            [71, 75],
            [72, 59],
            [72, 78],
            [73, 63],
            [73, 82],
            [74, 67],
            [74, 86],
            [75, 71],
            [75, 90],
            [76, 77],
            [76, 91],
            [77, 76],
            [77, 78],
            [78, 72],
            [78, 77],
            [78, 79],
            [79, 78],
            [79, 80],
            [80, 79],
            [80, 81],
            [80, 92],
            [81, 80],
            [81, 82],
            [82, 73],
            [82, 81],
            [82, 83],
            [83, 82],
            [83, 84],
            [84, 83],
            [84, 85],
            [84, 93],
            [85, 84],
            [85, 86],
            [86, 74],
            [86, 85],
            [86, 87],
            [87, 86],
            [87, 88],
            [88, 87],
            [88, 89],
            [88, 94],
            [89, 88],
            [89, 90],
            [90, 75],
            [90, 89],
            [91, 76],
            [91, 95],
            [92, 80],
            [92, 99],
            [93, 84],
            [93, 103],
            [94, 88],
            [94, 107],
            [95, 91],
            [95, 96],
            [96, 95],
            [96, 97],
            [97, 96],
            [97, 98],
            [97, 110],
            [98, 97],
            [98, 99],
            [99, 92],
            [99, 98],
            [99, 100],
            [100, 99],
            [100, 101],
            [101, 100],
            [101, 102],
            [101, 111],
            [102, 101],
            [102, 103],
            [103, 93],
            [103, 102],
            [103, 104],
            [104, 103],
            [104, 105],
            [105, 104],
            [105, 106],
            [105, 112],
            [106, 105],
            [106, 107],
            [107, 94],
            [107, 106],
            [107, 108],
            [108, 107],
            [108, 109],
            [109, 108],
            [109, 113],
            [110, 97],
            [110, 116],
            [111, 101],
            [111, 120],
            [112, 105],
            [112, 124],
            [113, 109],
            [113, 128],
            [114, 115],
            [114, 129],
            [115, 114],
            [115, 116],
            [116, 110],
            [116, 115],
            [116, 117],
            [117, 116],
            [117, 118],
            [118, 117],
            [118, 119],
            [118, 130],
            [119, 118],
            [119, 120],
            [120, 111],
            [120, 119],
            [120, 121],
            [121, 120],
            [121, 122],
            [122, 121],
            [122, 123],
            [122, 131],
            [123, 122],
            [123, 124],
            [124, 112],
            [124, 123],
            [124, 125],
            [125, 124],
            [125, 126],
            [126, 125],
            [126, 127],
            [126, 132],
            [127, 126],
            [127, 128],
            [128, 113],
            [128, 127],
            [129, 114],
            [130, 118],
            [131, 122],
            [132, 126],
        ]
    ),
]

# %%
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


# pick optimum edge to swap
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
                "edge_links_num": gym.spaces.Box(
                    low=0, high=300, shape=(), dtype=np.int32
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(
            300
        )  # pick optimum edge to swap (one of the edge_links)
        return

    def action_masks(self) -> np.ndarray:
        return np.array(
            [True] * self._edge_links_num + [False] * (300 - self._edge_links_num)
        )

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

    def get_total_swap(self) -> int:
        """Get the total number of swaps performed."""
        return self._swap_number

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
        self._coupling_map: CouplingMap = random.choice(
            list(
                filter(
                    lambda x: x.size() < self._level + 5 or self._level > 10,
                    coupling_map_list,
                )
            )
        )
        self._edge_links = np.array(self._coupling_map.get_edges(), dtype=np.int64)
        self._edge_links_num = self._edge_links.shape[0]
        self._node_num = self._coupling_map.size()
        self._success = False
        self._truncated = False
        self._swap_number = 0
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
        self, action: int
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

    def _apply_swap(self, action: int) -> bool:
        # Check if the action is valid
        if action > self._edge_links_num - 1:
            return True
        swap_action = self._edge_links[action]

        # apply the swap operation
        before_swap_depth = self._dest_dag.depth()
        self._dest_dag.apply_operation_back(
            self._swap_singleton,
            (
                self._dest_dag.qregs["q"][swap_action[0]],
                self._dest_dag.qregs["q"][swap_action[1]],
            ),
        )
        self._layout.swap(swap_action[0], swap_action[1])
        self._swap_depth += self._dest_dag.depth() - before_swap_depth
        self._swap_number += 1
        return False

    def _unpack_state(self) -> dict:
        """Unpack the state into a Graph instance."""
        nodes = np.zeros((self._node_num, self._N), dtype=np.int32)
        layers = self._sabre_dag.layers()
        layout = self._layout.get_virtual_bits()
        for i, layer in enumerate(layers):
            indicator = 0
            for node in layer["graph"].op_nodes():
                indicator += 1
                nodes[layout[node.qargs[0]], i] = indicator
                nodes[layout[node.qargs[1]], i] = indicator
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
            "edge_links_num": self._edge_links_num,
        }


# %%
from typing import Any
from typing import Dict

import gymnasium as gym
import os
import optuna
import numpy as np
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.ppo_mask import MaskablePPO
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch


# %%
class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, features_dim: int = 0
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.node_space = observation_space["nodes"].shape

        if self.node_space is None:
            raise ValueError(
                "Observation space must contain 'nodes' with a valid shape."
            )

        self.gcn1 = GATConv(self.node_space[1], 64)
        self.gcn2 = GATConv(64, 32)
        self.gcn3 = GATConv(32, 16)

        self.dropout = torch.nn.Dropout(0.1)

        self.feature_liner = torch.nn.Linear(16, features_dim)

    def forward(self, observations) -> torch.Tensor:
        batch_size = observations["nodes"].shape[0]
        nodes_num = observations["nodes_num"].long()
        edge_links_num = observations["edge_links_num"].long()

        # 배치 내 각 그래프에 대해 Data 객체 생성
        data_list = []
        for i in range(batch_size):
            nodes = observations["nodes"][i].float()
            edge_index = observations["edge_links"][i].long()

            filtered_nodes = nodes[: nodes_num[i]]
            filtered_edge_index = edge_index[: edge_links_num[i]]

            data = Data(x=filtered_nodes, edge_index=filtered_edge_index.T)
            data_list.append(data)

        # PyTorch Geometric의 배치 처리
        batch = Batch.from_data_list(data_list)

        if batch.batch.dtype != torch.long:  # type: ignore
            batch.batch = batch.batch.long()  # type: ignore

        # GCN 통과
        x = self.gcn1(batch.x, batch.edge_index)  # type: ignore
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.gcn2(x, batch.edge_index)  # type: ignore
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.gcn3(x, batch.edge_index)  # type: ignore
        x = torch.relu(x)

        # Global mean pooling 적용
        graph_features = global_mean_pool(
            x,
            batch.batch,  # type: ignore
        )

        # 선형 레이어를 통해 최종 특징 차원으로 변환
        graph_features = self.feature_liner(graph_features)  # (batch_size, features_dim

        return graph_features


# %%
ENV_ID = "CoRoutingEnv"

gym.register(id=ENV_ID, entry_point=CoRoutingEnv)  # type: ignore

N_TRIALS = 2000
N_STARTUP_TRIALS = 10
N_EVALUATIONS = 5
N_TIMESTEPS = int(1e6)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 5

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "policy_kwargs": dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    ),
}


# %%
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    gamma: float = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm: float = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda: float = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.1, log=True)
    n_steps: int = 2 ** trial.suggest_int("exponent_n_steps", 3, 11)
    learning_rate: float = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef: float = trial.suggest_float("ent_coef", 0.0000001, 0.1, log=True)
    vf_coef: float = trial.suggest_float("vf_coef", 0.000001, 1.0, log=True)

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
    }


# %%
from typing import Any


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        max_level: int = 1000,
        min_training_epi: int = 1000,
        success_threshold: float = 0.8,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.max_level = max_level
        self.success_threshold = success_threshold
        self.min_training_epi = min_training_epi

        self.level = 1
        self.current_success_rate = 0.0
        self.log_freq = 100
        self.step_count = 0
        self.episode_count = 0

    def on_training_start(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        super().on_training_start(locals_, globals_)
        self.training_env.env_method("set_level", level=self.level)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])

        if any(dones):
            for done in dones:
                if done:
                    self.episode_count += 1
            success_rate = np.mean(self.training_env.env_method("get_success_rate"))
            self.current_success_rate = success_rate

            if self.episode_count >= self.min_training_epi:
                if success_rate >= self.success_threshold:
                    self.episode_count = 0
                    self.level = min(self.level + 1, self.max_level)
                    self.training_env.env_method("set_level", level=self.level)

                    if self.verbose > 0:
                        print(
                            f"Level increased to {self.level} (success rate: {success_rate:.3f})"
                        )

        self.step_count += 1
        if self.step_count % self.log_freq == 0:
            self.logger.record("success_rate", self.current_success_rate)
            self.logger.record("level", self.level)
            self.logger.record("episode_count", self.episode_count)
            self.logger.record(
                "total_swap/mean",
                np.mean(self.training_env.env_method("get_total_swap")),
            )

            if self.step_count % (self.log_freq * 5) == 0:
                front_layer_len = self.training_env.env_method("front_layer_size")
                reset_failed = self.training_env.env_method("get_reset_failed")
                self.logger.record("front_layer_size/mean", np.mean(front_layer_len))
                self.logger.record("reset_failed/mean", np.mean(reset_failed))

        return True


# %%
class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            train_level = self.training_env.env_method("get_level")
            self.eval_env.env_method("set_level", level=train_level[0])
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# %%
EX_NAME = "optuna_sb3_dense_co_edge"


def make_env():
    return Monitor(gym.make(ENV_ID, start_level=1, max_episode_steps=1000))


def objective(trial: optuna.Trial) -> float:
    env = SubprocVecEnv([make_env for _ in range(16)])
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = MaskablePPO(
        **kwargs, env=env, tensorboard_log=f"./{EX_NAME}/", verbose=1, device="cuda"
    )
    # Create env used for evaluation.
    eval_env = Monitor(gym.make(ENV_ID, start_level=1))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )
    nan_encountered = False
    try:
        print("start learning")
        model.learn(
            N_TIMESTEPS,
            callback=[
                eval_callback,
                CurriculumCallback(
                    verbose=1, success_threshold=0.9, min_training_epi=2500
                ),
            ],
        )
        model.save(f"./{EX_NAME}/saves/rl_model_{trial.number}")
        print("Learning end")
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        if model.env is not None:
            model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    if not os.path.exists(f"./{EX_NAME}/study.db"):
        study = optuna.create_study(
            study_name="sb3_co",
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            storage=f"sqlite:///{EX_NAME}/study.db",
        )
    else:
        print("Loading existing study...")
        study = optuna.load_study(
            study_name="sb3_co", storage=f"sqlite:///{EX_NAME}/study.db"
        )
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=60 * 60 * 12)  # 12 hours
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
