"""
extracting state
plotting
analytic solutions for plotting and comparing
logging
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import sys
import imageio

logger = logging.getLogger(__name__)


def extracting_states(probfiles, norm_factor, num_lattices, use_1D):
    # getting the probabilities from the text file
    imported_probs = []
    for idx, cfile in enumerate(probfiles):
        with open(cfile, "r") as f:
            lines = f.readlines()[:num_lattices]
        probs = []
        for i in range(len(lines)):
            probs.append(float(lines[i].strip("\n")))
        imported_probs = np.asarray(np.sqrt(probs))

    # for 1D
    concentration = []
    if use_1D == True:
        for i in imported_probs:
            concentration.append(i * norm_factor * 2)
            # multiply with 2 because the original paper has 2 Hadamards - 1 encoding(sqrt(2)), 1 ancilla(sqrt(2))
    else:
        for i in imported_probs:
            concentration.append(i * norm_factor * 8)
            # multiply with 8 because the original paper has 6 Hadamards - 3 encoding, 3 ancilla)

    round_concentration = [round(elem, 3) for elem in concentration]
    # print(round_concentration)

    file1 = open('intelresults2d.txt', 'w')
    
    for i in round_concentration:
    	file1.writelines(str(i) + " " + "\n")
    
    file1.close()

    return round_concentration


def plotting_1D(vector, analytic_sol, num_lattice, iteration):
    x = np.arange(num_lattice)
    plt.title("Motion of Gaussian hill using QLBM")

    cx_max = 20

    fig, ax = plt.subplots()
    ax.plot(x[:cx_max], vector[:cx_max], "b", label="QLBM Solution")
    ax.plot(
        x[:cx_max], analytic_sol[:cx_max], "r--", alpha=0.9, label="Analytical Solution"
    )

    # ax.plot(x[:c_max], state_init[:c_max], 'r', alpha=0.5)
    ax.plot(6.0, 1.0, "rx", markersize=12, mew=4, alpha=0.75)
    ax.set_xlabel("x")
    ax.set_ylabel("concentration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.savefig("{}qlbm-1d-results.png".format(iteration), dpi=300, bbox_inches="tight")
    plt.close()


def plotting_2D(vector, latt_size, iteration, analytic_sol):
    latt_dim = int(np.sqrt(latt_size))
    num_sites = latt_dim**2
    x = np.zeros((latt_dim, latt_dim))
    y = np.zeros((latt_dim, latt_dim))
    for i in range(latt_dim):
        for j in range(latt_dim):
            x[i, j] = j
    for i in range(latt_dim):
        for j in range(latt_dim):
            y[i, j] = i

    crop_max = latt_dim

    max_peak = 1.0
    levels = np.linspace(-0.1, 1.05 * max_peak, 20)
    fig, ax = plt.subplots()
    sim_plot = np.reshape(vector, [latt_dim, latt_dim])

    qlbm2d_plot = ax.contourf(
        x[:crop_max, :crop_max],
        y[:crop_max, :crop_max],
        sim_plot[:crop_max, :crop_max],
        levels=levels,
        cmap="Reds",
    )

    ax.contour(
        x[:crop_max, :crop_max],
        y[:crop_max, :crop_max],
        analytic_sol[:crop_max, :crop_max],
        levels=levels,
        cmap="Reds",
        alpha=0.5,
        linestyles="dashed",
    )
    """
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, sim_plot, rstride=1, cstride=1,
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_zlim(0.004, 0.25)
    """

    plt.colorbar(qlbm2d_plot)

    ax.plot(3.0, 3.0, "rx", markersize=6, mew=4, alpha=0.75)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    fig = plt.gcf()
    # fig.set_size_inches(8, 4)
    plt.savefig("{}qlbm-2d-results.png".format(iteration), dpi=300, bbox_inches="tight")
    plt.close()


def analytic1d(parameters, num_states, iteration):
    lattice_end = int(num_states / 2)
    x = np.arange(lattice_end)
    sol = np.zeros(lattice_end)

    c_max = parameters.c_max
    c_init = parameters.c_init
    vel = parameters.velocity
    x_max = parameters.x_max
    steps = iteration + 1

    for i in range(lattice_end):
        sol[i] = c_init + (c_max - c_init) / math.sqrt(0.5 * 3.14 * steps) * math.exp(
            -1.0 * ((x[i] - vel * steps - x_max) ** 2.0) / (0.5 * 4.0 * steps)
        )

    return sol


def analytic2d(parameters, latt_size, iteration):
    c_max = parameters.c_max
    c_init = parameters.c_init
    x_vel = parameters.x_vel
    y_vel = parameters.y_vel
    x_max = parameters.x_max
    y_max = parameters.y_max
    parameters.steps = iteration + 1
    steps = parameters.steps

    latt_dim = int(np.sqrt(latt_size))
    num_sites = latt_dim**2
    x = np.zeros((latt_dim, latt_dim))
    y = np.zeros((latt_dim, latt_dim))
    for i in range(latt_dim):
        for j in range(latt_dim):
            x[i, j] = j
    for i in range(latt_dim):
        for j in range(latt_dim):
            y[i, j] = i

    sol = np.zeros((latt_dim, latt_dim))

    for i in range(latt_dim):
        for j in range(latt_dim):
            sol[i, j] = ((c_max - c_init) / (4.0 * 0.16667 * 3.14 * steps)) * math.exp(
                -1.0
                * (
                    (x[i, j] - x_vel * steps - x_max) ** 2.0
                    + (y[i, j] - y_vel * steps - y_max) ** 2.0
                )
                / (4.0 * 0.16667 * steps)
            ) + c_init

    return sol


def log_file_handler(file: str) -> logging.FileHandler:
    formatter = logging.Formatter(
        "%(asctime)s (%(relativeCreated)d) - [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handle = logging.FileHandler(file, mode="w")
    handle.setFormatter(formatter)
    handle.setLevel(logging.DEBUG)

    return handle


def setup_logging(logfile: str = "main.log"):
    fhandle = log_file_handler(logfile)

    # Skip external info logs
    logger_blocklist = [
        "qiskit",
        "matplotlib",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fhandle)

    return


def image_to_gif(steps):
    vector = np.vectorize(np.int)
    array = np.arange(start=0, stop=steps, step=1)
    x = vector(array)
    with imageio.get_writer("qAVERAGEensambles.gif", mode="I", duration=100) as writer:
        filenames = [f"{i}qlbm-2d-results.png" for i in x]
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
