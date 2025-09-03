import marimo

__generated_with = "0.15.2"
app = marimo.App()


app._unparsable_cell(
    r"""
    # '%matplotlib inline' command supported automatically in marimo
    from math import pi, sqrt
    from qiskit import *
    """,
    name="_"
)


@app.cell
def _(QuantumCircuit, pi):
    # Preparing qubit to be measured
    # Here, we prepare q0 to be (sqrt(3)/2)|0> + (1/2)|1> using the Ry gate
    prep_circuit = QuantumCircuit(3, 3)
    prep_circuit.ry(pi/3, 0)
    prep_circuit.barrier()
    prep_circuit.draw(output="mpl")
    return (prep_circuit,)


@app.cell
def _(QuantumCircuit):
    # Preparing the Bell state
    bell_circuit = QuantumCircuit(3, 3)
    bell_circuit.h(1)  # Superposition by Hadamard gate
    bell_circuit.cx(1, 2)  # Entanglement by CNOT gate
    bell_circuit.barrier()
    bell_circuit.draw(output="mpl")
    return (bell_circuit,)


@app.cell
def _(QuantumCircuit):
    # Create circuit that uses Bell state to teleport
    teleport_circuit = QuantumCircuit(3, 3)

    # Step 0: Apply CNOT and Hadamard
    teleport_circuit.cx(0, 1)
    teleport_circuit.h(0)
    teleport_circuit.barrier()

    # Step 1: Measure first two qubits
    # N.B.: We overwrite the reads because now we do not care much what measurement we got
    # We could keep it to send, but our qubits are close enough just to use CNOT and CZ gates
    teleport_circuit.measure([0, 1], [0, 1])
    teleport_circuit.barrier()

    # Step 2: Apply CNOT and CZ gates to recover original qubit
    teleport_circuit.cx(1, 2)
    teleport_circuit.cz(0, 2)

    # In addition, we measure the qubit to check if it matches our expectations
    teleport_circuit.measure(2, 2)

    teleport_circuit.draw(output="mpl")
    return (teleport_circuit,)


@app.cell
def _(bell_circuit, prep_circuit, teleport_circuit):
    # Finally, we merge all the steps together
    circuit = prep_circuit + bell_circuit + teleport_circuit
    circuit.draw(output="mpl")
    return (circuit,)


@app.cell
def _(Aer, circuit, execute, visualization):
    # We simulate to verify that the qubit was teleported
    simulator = Aer.get_backend("qasm_simulator")
    job = execute(circuit, simulator, shots=2**20)
    result = job.result()
    counts = result.get_counts()
    visualization.plot_histogram(counts)
    return (counts,)


@app.cell
def _(counts, sqrt):
    # Extracting alpha and beta from counts
    alpha_sq = 0
    beta_sq = 0

    for key, value in counts.items():
        if key[0] == "0":
            alpha_sq += value
        else:
            beta_sq += value

    alpha = sqrt(alpha_sq/(2**20))
    beta = sqrt(beta_sq/(2**20))

    print("{}|0> + {}|1>".format(alpha, beta))
    return


@app.cell
def _(qiskit):
    # The results match our expectations! Roughly (sqrt(3)/2)|0> + (1/2)|1>
    # For purposes of reproducibility, the Qiskit version is
    qiskit.__qiskit_version__
    return


if __name__ == "__main__":
    app.run()

