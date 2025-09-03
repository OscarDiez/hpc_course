import marimo

__generated_with = "0.15.2"
app = marimo.App()


app._unparsable_cell(
    r"""
    # Superdense Coding
    # '%matplotlib inline' command supported automatically in marimo
    from qiskit import *
    """,
    name="_"
)


@app.cell
def _(QuantumCircuit):
    # Step 0: Prepare the Bell State
    bell_circuit = QuantumCircuit(2, 2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_circuit.barrier()
    bell_circuit.draw(output="mpl")
    return (bell_circuit,)


@app.cell
def _(QuantumCircuit):
    # Step 1: Prepare Alice circuits depending on b0b1

    # Case 00: the circuit has no gates
    alice_00 = QuantumCircuit(2, 2)
    alice_00.barrier()
    alice_00.draw(output="mpl")
    return (alice_00,)


@app.cell
def _(QuantumCircuit):
    # Case 01: the circuit has the X gate
    alice_01 = QuantumCircuit(2, 2)
    alice_01.x(0)
    alice_01.barrier()
    alice_01.draw(output="mpl")
    return (alice_01,)


@app.cell
def _(QuantumCircuit):
    # Case 10: the circuit has the Z gate
    alice_10 = QuantumCircuit(2, 2)
    alice_10.z(0)
    alice_10.barrier()
    alice_10.draw(output="mpl")
    return (alice_10,)


@app.cell
def _(QuantumCircuit):
    # Case 11: the circuit has the X gate and then Z gate
    alice_11 = QuantumCircuit(2, 2)
    alice_11.x(0)
    alice_11.z(0)
    alice_11.barrier()
    alice_11.draw(output="mpl")
    return (alice_11,)


@app.cell
def _(QuantumCircuit):
    # Step 2: Apply the inverted entanglement circuit, and then measure
    invert_circuit = QuantumCircuit(2, 2)
    invert_circuit.cx(0, 1)
    invert_circuit.h(0)
    invert_circuit.barrier()
    invert_circuit.measure([0, 1], [1, 0])  # Qiskit measures are always reversed, b1b0 not b0b1
    invert_circuit.draw(output="mpl")
    return (invert_circuit,)


@app.cell
def _(alice_11, bell_circuit, invert_circuit):
    # When merged, the whole circuit looks like
    (bell_circuit + alice_11 + invert_circuit).draw(output="mpl")
    return


@app.cell
def _(Aer, execute):
    # Now we simulate each outcome, to do so we create an auxiliary function that runs it
    def simulate_circuit(prep, encoding, decoding):
        """Returns the counts of the circuit that is combination of the three circuits"""
        circuit = prep + encoding + decoding
        simulator = Aer.get_backend("qasm_simulator")
        job = execute(circuit, simulator, shots = 2**16)
        result = job.result()
        count = result.get_counts()
        return count
    return (simulate_circuit,)


@app.cell
def _(alice_00, bell_circuit, invert_circuit, simulate_circuit, visualization):
    # For 00
    count_00 = simulate_circuit(bell_circuit, alice_00, invert_circuit)
    visualization.plot_histogram(count_00)
    return


@app.cell
def _(alice_01, bell_circuit, invert_circuit, simulate_circuit, visualization):
    # For 01
    count_01 = simulate_circuit(bell_circuit, alice_01, invert_circuit)
    visualization.plot_histogram(count_01)
    return


@app.cell
def _(alice_10, bell_circuit, invert_circuit, simulate_circuit, visualization):
    # For 10
    count_10 = simulate_circuit(bell_circuit, alice_10, invert_circuit)
    visualization.plot_histogram(count_10)
    return


@app.cell
def _(alice_11, bell_circuit, invert_circuit, simulate_circuit, visualization):
    # For 11
    count_11 = simulate_circuit(bell_circuit, alice_11, invert_circuit)
    visualization.plot_histogram(count_11)
    return


@app.cell
def _(qiskit):
    # The results match our predictions!
    # For purposes of reproducibility, the Qiskit version is
    qiskit.__qiskit_version__
    return


if __name__ == "__main__":
    app.run()

