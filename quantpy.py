import numpy as np
import random
from matplotlib import pyplot as plt

class Gate(np.ndarray):
    def __new__(cls, name, *args, **kwargs):
        if type(name) == np.ndarray:
            obj = name
        elif type(name) == list:
            obj = np.array(name)
        elif name == 'I':
            obj = np.identity(2)
        elif name == 'H':
            hadamard = np.ones((2, 2)) / np.sqrt(2)
            hadamard[1, 1] *= -1
            obj = hadamard
        elif name == 'X':
            obj = np.ones((2,2)) - np.identity(2)
        elif name == 'Y':
            obj = np.array([[0, -1j], [1j, 0]])
        elif name == 'Z':
            Z = np.identity(2)
            Z[1, 1] *= -1
            obj = Z
        elif name == 'S':
            obj = np.array([[1, 0], [0, 1j]])
        elif name == 'T':
            obj = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]])
        elif name == 'Rx':
            obj = lambda phi: np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        elif name == 'Ry':
            obj = lambda phi: np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        elif name == 'Rz':
            obj = lambda phi: np.array([[np.exp(-1j* phi / 2), 0], [0, np.exp(1j * phi / 2)]])
        elif name == 'CNOT':
            obj = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        elif name == 'invCNOT':
            obj = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        return obj.view(cls)

    def check_valid_operation(self, gate=None):
        if gate is None: gate = self
        return bool(np.all(np.isclose(gate @ np.conjugate(gate).T, np.identity(gate.shape[0]))))

    def check_reverseble(self, gate=None):
        if gate is None: gate = self
        return bool(np.all(np.isclose(gate @ gate, np.identity(2)) == True))

    def check_commuting(self, gate=None):
        if gate is None: gate = self
        if np.all(gate @ self == self @ gate):
            return True
        else:
            return False

    def apply_kronecker_product(self, gate=None):
        if gate is None: gate = self
        return np.kron(self, gate)

    def single_to_double(self, gate=None, index=0):
        if gate is None: gate = self
        if index == 1:
            return np.kron(np.identity(2, dtype=complex), gate)
        else:
            return np.kron(gate, np.eye(2, dtype=complex))

    std = single_to_double

    def apply_quantum_gates(self, list):
        if type(list) == list:
            g = list.pop(0)
            for gate in list:
                g = gate @ g
            return g
        else:
            return self @ list

    def collapse(self):
        if len(self.shape) != 1:
            raise f"Can't collapse a gate, shape: {self.shape}"
        return np.random.binomial(1, np.square(self))

    def beautify(self):
        self = np.round(self, 3)
        self = np.where(self == -0.0, 0.0, self)
        return self


class State(np.ndarray):
    def __new__(cls, name, bra=False, *args, **kwargs):
        if name == '0' or name == '1':
            obj = np.identity(2, dtype=int)[int(name)]
        elif name == '+':
            hadamard = Gate('H')
            obj = hadamard[0]
        elif name == '-':
            hadamard = Gate('H')
            obj = hadamard[1]
        elif name == 'i' or name =='+i' or name =='i+':
            obj = np.array([1,1j]) / np.sqrt(2)
        elif name == 'i' or name =='-i' or name =='i-':
            obj = np.array([1,-1j]) / np.sqrt(2)
        elif name == '00' or name == '01' or name == '10' or name == '11':
            obj = np.identity(4)[int(name, 2)]
        elif name == 'bell00p11' or name == 'bell0' or name == 'bell00':
            obj = np.array([1, 0, 0, 1]) / np.sqrt(2)
        elif name == 'bell00m11' or name == 'bell2' or name == 'bell01':
            obj = np.array([1, 0, 0, -1]) / np.sqrt(2)
        elif name == 'bell01p10' or name == 'bell1' or name == 'bell10':
            obj = np.array([0, 1, 1, 0]) / np.sqrt(2)
        elif name == 'bell01m10' or name == 'bell3' or name == 'bell11':
            obj = np.array([0, 1, -1, 0]) / np.sqrt(2)

        if bra == True:
            obj = np.conjugate(obj)
            cls.bra = True

        return obj.view(cls)

    def is_normalized(self, v=None):
        if v is None: v = self
        length = np.linalg.norm(np.abs(v))
        return bool(np.isclose(length, 1))

    def probability_of_0(self, v=None):
        if v is None: v = self
        return np.abs(v[0]) ** 2

    def probability_of_1(self, v=None):
        if v is None: v = self
        return np.abs(v[1]) ** 2

    def apply_kronecker_product(self, state=None):
        if state is None: state = self
        return np.kron(self, state)

    kp = apply_kronecker_product

    def apply_gate_to_state(self, gate, state=None):
        if state is None: state = self
        return gate @ state

    def braket(self, state=None):
        if state is None: state = self
        return np.inner(np.conj(self), state)

    def braket_probabilty(self, state=None):
        if state is None: state = self
        return np.square(np.abs(np.inner(self, state)))

    def ketbra(self, state=None):
        if state is None: state = self
        return np.outer(self, np.conj(state))

    def Rx(self, phi):
        return np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]) @ self
    def Ry(self, phi):
        return np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]) @ self
    def Rz(self, phi):
        return np.array([[np.exp(-1j* phi / 2), 0], [0, np.exp(1j * phi / 2)]]) @ self

    def collapse(self):
        return np.random.binomial(1, np.square(self))

    def beautify(self):
        self = np.round(self, 3)
        self = np.where(self == -0.0, 0.0, self)
        return self


class Circuit:
    def __init__(self, n_qubit):
        self.n_qubit = n_qubit
        self.circuit_state = np.zeros(2**n_qubit, dtype=np.complex64)
        self.set("0" * n_qubit)

    def identity_kron(self, qubit, gate):
        if qubit == 0:
            s = gate
        else:
            s = np.identity(2)

        for i in range(1, self.n_qubit):
            if i == qubit:
                s = np.kron(s, gate)
            else:
                s = np.kron(s, np.identity(2))
        return s

    def set(self, state="0101"):
        one = int(state, 2)
        self.circuit_state = np.zeros((2 ** self.n_qubit))
        self.circuit_state[one] = 1
        return self.circuit_state

    def Hadamard(self, qubit=0):
        s = self.identity_kron(qubit, Gate("H"))
        self.circuit_state = s @ self.circuit_state
        return s

    H = Hadamard

    def X(self, qubit=0):
        s = self.identity_kron(qubit, Gate("X"))
        self.circuit_state = s @ self.circuit_state
        return s

    def apply_gate(self, qubit=0, gate=None):
        s = self.identity_kron(qubit, gate)
        self.circuit_state = s @ self.circuit_state
        return s

    def CNOT(self, control=(0,), target=1, inplace=True):
        if type(control) == int:
            control = (control,)
        size = 2 ** self.n_qubit
        cnot_matrix = np.eye(size, dtype=complex)

        for i in range(size):
            binary = list(format(i, f'0{self.n_qubit}b'))
            controls_activated = all(binary[c] == '1' for c in control)

            if controls_activated:
                binary[target] = '0' if binary[target] == '1' else '1'
                j = int(''.join(binary), 2)
                cnot_matrix[i, i], cnot_matrix[i, j] = 0, 1
                cnot_matrix[j, j], cnot_matrix[j, i] = 0, 1

        if inplace:
            self.circuit_state = cnot_matrix @ self.circuit_state
        return cnot_matrix

    def multi_qubit_logic_gate(self, input_bits, target_bit, gate_type):
        size = 2 ** self.n_qubit
        gate_matrix = np.eye(size, dtype=complex)

        for i in range(size):
            binary = list(format(i, f'0{self.n_qubit}b'))
            inputs = [binary[b] == '1' for b in input_bits]

            if gate_type == 'AND':
                output = all(inputs)  # AND logic
            elif gate_type == 'OR':
                output = any(inputs)  # OR logic
            else:
                raise ValueError("Invalid gate_type. Choose 'AND' or 'OR'.")

            # Set the target bit based on the logic gate's result
            if output:
                binary[target_bit] = '1'
            else:
                binary[target_bit] = '0'

            j = int(''.join(binary), 2)
            gate_matrix[i, i], gate_matrix[i, j] = 0, 1
            gate_matrix[j, j], gate_matrix[j, i] = 0, 1

        self.circuit_state = gate_matrix @ self.circuit_state
        return gate_matrix

    def AND (self, input_bits, target_bit):
        return self.multi_qubit_logic_gate(input_bits, target_bit, "AND")

    def OR (self, input_bits, target_bit):
        return self.multi_qubit_logic_gate(input_bits, target_bit, "OR")

    def XOR (self, input_bits, target_bit, inplace=True):
        for input_bit in input_bits:
            self.CNOT(input_bit, target_bit, inplace=inplace)
        return self.circuit_state

    def diffusion_operator(self, n_input_bits, n_helper_bits):
        total_bits = n_input_bits + n_helper_bits + 1
        last_qbit_index = total_bits - 1

        for i in range(n_input_bits):
            self.H(i)
            self.X(i)

        self.X(last_qbit_index)
        self.H(last_qbit_index)

        self.CNOT(tuple(range(n_input_bits)), last_qbit_index)

        self.H(last_qbit_index)
        self.X(last_qbit_index)

        for i in range(n_input_bits):
            self.X(i)
            self.H(i)

        return self.circuit_state

    def T_sharp(self, permutation, good_exp):
        return int(np.round(np.pi / (4 * np.sqrt(good_exp / permutation))))

    def Grover_Clauses(self, clauses_list):
        n = np.max(clauses_list) + 1
        qlist_sf_index = [i + n - 1 for i in range(len(clauses_list))]
        nqubits = n + len(clauses_list) + 1
        last_qbit_index = nqubits - 1

        for count, clause in enumerate(clauses_list):
            self.XOR(tuple(clause), count + n)
            self.CNOT(tuple(clause), count + n)

        # Mark the solutions with a -1.
        self.X(last_qbit_index)
        self.H(last_qbit_index)
        self.CNOT(tuple(qlist_sf_index), last_qbit_index)
        self.H(last_qbit_index)
        self.X(last_qbit_index)

        for count, clause in enumerate(clauses_list):
            self.CNOT(tuple(clause), count + n)
            self.XOR(tuple(clause), count + n)

        return self.circuit_state

    def perform_Grover(self, clauses_list, n_solutions, print_probs_only=False, print_text=False):
        n = np.max(clauses_list) + 1
        sf = len(clauses_list)
        nqubits = n + sf + 1

        self.__init__(nqubits)
        for i in range(n):
            self.H(i)

        t_sharp = self.T_sharp(2 ** n, n_solutions)
        for i in range(t_sharp):

            self.Grover_Clauses(clauses_list)
            self.diffusion_operator(n, sf)

            self.propabilities(
                clauses_list=clauses_list,
                n_marked_bits=n,
                print_probs_only=print_probs_only,
                print_text=print_text,
                print_top_k=n_solutions
            )

    def propabilities(self, clauses_list=None, n_marked_bits=1, print_probs_only=False, print_text=True, print_top_k=False):

        probs = np.square(np.abs(self.circuit_state))  # Calculate probabilities
        data_y = []
        data_x = []

        if clauses_list:
            n = np.max(clauses_list) + 1
            for i in range(len(probs)):
                state = format(i, f'0{self.n_qubit}b')
                variables = [int(state[j]) for j in range(n)]
                marked_state = state[:n_marked_bits]

                if marked_state in data_x:
                    ind = data_x.index(marked_state)
                    data_y[ind] += probs[i]
                else:
                    data_y.append(probs[i])
                    data_x.append(marked_state)
        else:
            for i in range(len(probs)):
                if not np.isclose(probs[i], 0):
                    state = format(i, f'0{self.n_qubit}b')
                    data_y.append(probs[i])
                    data_x.append(state)

        if print_text:
            for state, prob in zip(data_x, data_y):
                print(f"{state} = {round(prob, 5)}")

        # Plot the filtered probabilities
        plt.bar(data_x, data_y)
        plt.xticks(range(len(data_x)), data_x, rotation=90)
        plt.ylabel("Probabilities")
        plt.show()

        if print_top_k:
            print(f"Top {print_top_k} States:\n{sorted([x for _, x in sorted(zip(data_y, data_x), reverse=True)[:print_top_k]])}")


    def __str__(self):
        return str(self.circuit_state)

    def __repr__(self):
        return str(self.circuit_state)

    def beautify(self):
        s = np.round(self.circuit_state, 3)
        s = np.where(s == -0.0, 0.0, s)
        return s




i_plus = State("i+")
i_minus = State("i-")
plus = State("+", bra=True)
minus = State("-")
_0 = State("0")
_1 = State("1")
bell00 = State("bell00")
bell11 = State("bell11")
bell01 = State("bell01")
bell10 = State("bell10")

H = Gate("H")
CNOT = Gate("CNOT")
Z = Gate("Z")
X = Gate("X")


__all__ = [
    "i_plus",
    "i_minus",
    "plus",
    "minus",
    "_0",
    "_1",
    "bell00",
    "bell11",
    "bell01",
    "bell10",
    "H",
    "CNOT",
    "Z",
    "X",
    "Circuit",
    "Gate",
    "State"
]

