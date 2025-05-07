from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library.standard_gates import SwapGate


class OriginalSabreSwap(TransformationPass):
    """Original Sabre Swap pass."""

    def __init__(self, coupling_map: CouplingMap):
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        # We assume that the layout is already done by the user.

        if self.coupling_map is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) != self.coupling_map.size():
            raise TranspilerError("Sabre swap runs on physical circuits only.")
        # add more constraints if needed

        self.dist_matrix = self.coupling_map.distance_matrix

        # 처음 layout을 가져와서 physical qubit과 logical qubit의 mapping을 시켜놓아야 한다. (Swap을 위해서) -> 이것슨 수동으로 구현해야한다. _accelerate에 존재하므로.

        swap_singleton = SwapGate()

        dest_dag = dag
        dag

        return (dest_dag, layout)


# SABRE를 구현하기 위해서 1 qubit gate는 일단 두고, 2 qubit gate만의 layer를 찾아야 한다. -> 이는 현재 dag에서 1 qubit gate를 모두 제거 함으로 가능하다.
# 그리고 2 qubit gate는 모두 swap을 통해서 mapping을 해줘야 한다. -> 이때 swap은 coupling map에 맞춰서 해야 한다. 이때, 휴리스틱은 따로 구현을 해 주어야 한다. 이는 일단 구현이 되어있는 github의 repo를 참고해서 알고리즘을 가져오자.
# 한번 swap을 했으면, 그에 따른 logical-physical qubit의 mapping을 업데이트 해 주어야 한다. 이를 위해서 우리는 주어지는 gate의 기준이 logical qubit임을 주의하여야 한다. 또한 distance를 구할 때는, logical qubit이 아니라 physical qubit을 기준으로 해야 한다. -> 이때, coupling map을 통해서 distance를 구할 수 있다
