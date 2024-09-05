"""
Initial state
euclidean norm
normalize
decompose
convert to C++
"""

from qiskit import QuantumCircuit, QuantumRegister, transpile
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def norm(concentration):
    return np.sqrt(np.vdot(concentration, concentration))


def normalize_vector(concentration, euclidean_norm, num_lattice):
    new_state = []

    # finding the normalization
    for j in range(len(concentration)):
        new_state.insert(j, concentration[j] / euclidean_norm)

    check = np.vdot(new_state[:num_lattice], new_state[:num_lattice])

    return new_state


def decomposing(amplitudes, iter, num_lattices):
    num_qubits_lattice = np.log2(num_lattices)

    q = QuantumRegister(num_qubits_lattice, name="qmem")
    qc = QuantumCircuit(q)
    qc.initialize(amplitudes[:num_lattices], q)

    # qcm = qc.decompose(reps=2)
    qcm = transpile(circuits=qc, basis_gates=["cx", "ry"])
    qcm.qasm(filename="encoding_x02.qasm")
    # qcm.draw('mpl', filename='encoding_x02.png')


def conversion(use_1D):
    mainFile = "encoding_x02.cpp"

    if use_1D == True:
        subFile = "encoding_x1D.txt"
    else:
        subFile = "encoding_x2D.txt"

    subFile1 = "encoding_x02.h"
    marker1 = "{"
    marker2 = "}"
    marker3 = "**--------**"
    marker4 = "//Hadamard gate for the second velocity distribution function"

    with open(mainFile, "r") as f1, open(subFile, "r") as f2, open(subFile1, "w") as f3:
        main = f1.read()
        sub = f2.read()

        start = main.find(marker1)
        end = main.find(marker2)

        start1 = sub.find(marker3)
        end1 = sub.find(marker4)

        sub2 = sub[:start1] + main[start + 1 : end] + sub[end1:]

        f3.write(sub2)


def interpolate_initial_1D(
    num_states: int, x_max: int, c_init: float, c_max: float
) -> np.ndarray:
    # Lattice vector
    a = np.zeros(shape=(num_states))

    # Max values: at given lattice point, and it's duplicate on the latter half of the state list
    lattice_end = int(num_states / 2)
    print("Number of Lattice sites = ", lattice_end)

    max_points = [x_max]
    half_points = [x_max - 1, x_max + 1]

    for i in range(lattice_end):
        if i in max_points:
            a[i] = c_max
        elif i in half_points:
            a[i] = c_init + 0.5 * (c_max - c_init)
        else:
            a[i] = c_init

    print("The Initial Vector")
    # print(a[:lattice_end])
    x = np.arange(lattice_end)
    plt.plot(x, a[:lattice_end])

    return a[:lattice_end]


def interpolate_initial_2D(num_qubits_lattice_side, lattice_size, c_max):
    lattice_side = 2 ** (num_qubits_lattice_side)
    print(lattice_side)
    init_state = np.zeros(lattice_size)

    id_max = (lattice_side * 3) + 3

    init_state[int(id_max)] = c_max

    return init_state


def interpolate_initial_2d(num_qubits_lattice_side, lattice_size, c_max):
    lattice_side = 2 ** (num_qubits_lattice_side)
    print(lattice_side)
    init_state = np.zeros(lattice_size)
    print(lattice_size)
    id_max1 = (lattice_side * (lattice_side / 2 - 1)) + (
        lattice_side / 2 - lattice_side / 4
    )
    id_max2 = (lattice_side * (lattice_side / 2 - 1)) + (
        lattice_side / 2 - lattice_side / 4 + 1
    )
    id_max3 = (lattice_side * (lattice_side / 2 - 1)) + (
        lattice_side / 2 - lattice_side / 4 + 2
    )
    id_max4 = (lattice_side * (lattice_side / 2 - 1)) + (
        lattice_side / 2 - lattice_side / 4 - 1
    )
    id_max5 = (lattice_side * (lattice_side / 2 - 1)) + (
        lattice_side / 2 - lattice_side / 4 - 2
    )
    id_max14 = (lattice_side * (lattice_side / 2 - 1)) + (
        lattice_side / 2 - lattice_side / 4 + 3
    )
    id_max15 = (lattice_side * (lattice_side / 2 - 1)) + (
        lattice_side / 2 - lattice_side / 4 - 3
    )

    id_max6 = (lattice_side * (lattice_side / 2 - 2)) + (
        lattice_side / 2 - lattice_side / 4
    )
    id_max7 = (lattice_side * (lattice_side / 2 - 2)) + (
        lattice_side / 2 - lattice_side / 4 + 1
    )
    id_max8 = (lattice_side * (lattice_side / 2 - 2)) + (
        lattice_side / 2 - lattice_side / 4 - 1
    )

    id_max9 = (lattice_side * (lattice_side / 2)) + (
        lattice_side / 2 - lattice_side / 4
    )
    id_max10 = (lattice_side * (lattice_side / 2)) + (
        lattice_side / 2 - lattice_side / 4 + 1
    )
    id_max11 = (lattice_side * (lattice_side / 2)) + (
        lattice_side / 2 - lattice_side / 4 - 1
    )

    id_max12 = (lattice_side * (lattice_side / 2 - 3)) + (
        lattice_side / 2 - lattice_side / 4
    )
    id_max13 = (lattice_side * (lattice_side / 2 + 1)) + (
        lattice_side / 2 - lattice_side / 4
    )

    init_state[int(id_max1)] = c_max
    init_state[int(id_max2)] = c_max
    init_state[int(id_max3)] = c_max
    init_state[int(id_max4)] = c_max
    init_state[int(id_max5)] = c_max
    init_state[int(id_max6)] = c_max
    init_state[int(id_max7)] = c_max
    init_state[int(id_max8)] = c_max
    init_state[int(id_max9)] = c_max
    init_state[int(id_max10)] = c_max
    init_state[int(id_max11)] = c_max
    init_state[int(id_max12)] = c_max
    init_state[int(id_max13)] = c_max
    init_state[int(id_max14)] = c_max
    init_state[int(id_max15)] = c_max

    print(init_state[int(lattice_size / 2 - 200) : int(lattice_size / 2 + 200)])

    return init_state


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def initial_state_circle(num_qubits_lattice_side, radius, c_max):
    num_lattice_pts = int(2 ** (num_qubits_lattice_side))
    cylinder = np.full((num_lattice_pts, num_lattice_pts), 0)
    for y in range(0, num_lattice_pts):
        for x in range(0, num_lattice_pts):
            if distance(num_lattice_pts // 4, num_lattice_pts // 2, x, y) < radius:
                cylinder[y][x] = c_max

    # print("initial array", str(cylinder))

    circle_state_vector = cylinder.flatten()
    # print("New resulting array: ", circle_state_vector)

    return circle_state_vector
