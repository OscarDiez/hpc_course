import marimo

__generated_with = "0.15.2"
app = marimo.App()


app._unparsable_cell(
    r"""
    # Deutsch's Algorithm
    # '%matplotlib inline' command supported automatically in marimo
    from qiskit import *
    """,
    name="_"
)


@app.cell
def _(QuantumCircuit):
    # Step 0: prepare the superposition
    prep_circuit = QuantumCircuit(2, 1)
    prep_circuit.x(1)
    prep_circuit.h([0, 1])
    prep_circuit.barrier()
    prep_circuit.draw(output="mpl")
    return (prep_circuit,)


@app.cell
def _(QuantumCircuit):
    # Step 1: send input to blackbox. There are four possible black boxes
    # f0: f(0) = 0 and f(1) = 0, f is constant
    f0_circuit = QuantumCircuit(2, 1)
    f0_circuit.barrier()
    f0_circuit.draw(output="mpl")
    return (f0_circuit,)


@app.cell
def _(QuantumCircuit):
    # f1: f(0) = 0 and f(1) = 1, f is balanced
    f1_circuit = QuantumCircuit(2, 1)
    f1_circuit.cx(0, 1)  # the second bit only becomes 1 when input is 1, hence CNOT gate
    f1_circuit.barrier()
    f1_circuit.draw(output="mpl")
    return (f1_circuit,)


@app.cell
def _(QuantumCircuit):
    # f2: f(0) = 1 and f(1) = 0, f is balanced
    f2_circuit = QuantumCircuit(2, 1)
    f2_circuit.x(1)
    f2_circuit.cx(0, 1)  # the second bit only becomes 1 when input is 0, hence NOT with CNOT gate
    f2_circuit.barrier()
    f2_circuit.draw(output="mpl")
    return (f2_circuit,)


@app.cell
def _(QuantumCircuit):
    # f3: f(0) = 1 and f(1) = 1, f is constant
    f3_circuit = QuantumCircuit(2, 1)
    f3_circuit.x(1)  # f is 1 regardless
    f3_circuit.barrier()
    f3_circuit.draw(output="mpl")
    return (f3_circuit,)


@app.cell
def _(QuantumCircuit):
    # Step 2: apply Hadamard and measure it
    measure_circuit = QuantumCircuit(2, 1)
    measure_circuit.h(0)
    measure_circuit.measure(0, 0)
    measure_circuit.draw(output="mpl")
    return (measure_circuit,)


@app.cell
def _(f2_circuit, measure_circuit, prep_circuit):
    # An example of what the assembled circuit looks like
    (prep_circuit + f2_circuit + measure_circuit).draw(output="mpl")
    return


@app.cell
def _(Aer, execute):
    # Now we simulate for each function, to do so we create an auxiliary function that runs it
    def simulate_circuit(prep, blackbox, measuring):
        """Returns the counts of the circuit that is combination of the three circuits"""
        circuit = prep + blackbox + measuring
        simulator = Aer.get_backend("qasm_simulator")
        job = execute(circuit, simulator, shots = 2**16)
        result = job.result()
        count = result.get_counts()
        return count
    return (simulate_circuit,)


@app.cell
def _():
    # Recall that the measurement is 0 if f is balanced, and 1 if f is constant
    return


@app.cell
def _(
    f0_circuit,
    measure_circuit,
    prep_circuit,
    simulate_circuit,
    visualization,
):
    # For f0, we expect only 0s in measurement because it is constant
    count_f0 = simulate_circuit(prep_circuit, f0_circuit, measure_circuit)
    visualization.plot_histogram(count_f0)
    return


@app.cell
def _(
    f1_circuit,
    measure_circuit,
    prep_circuit,
    simulate_circuit,
    visualization,
):
    # For f1, we expect only 1s in measurement because it is balanced
    count_f1 = simulate_circuit(prep_circuit, f1_circuit, measure_circuit)
    visualization.plot_histogram(count_f1)
    return


@app.cell
def _(
    f2_circuit,
    measure_circuit,
    prep_circuit,
    simulate_circuit,
    visualization,
):
    # For f2, we expect only 1s in measurement because it is balanced
    count_f2 = simulate_circuit(prep_circuit, f2_circuit, measure_circuit)
    visualization.plot_histogram(count_f2)
    return


@app.cell
def _(
    f3_circuit,
    measure_circuit,
    prep_circuit,
    simulate_circuit,
    visualization,
):
    # For f3, we expect only 0s in measurement because it is constant
    count_f3 = simulate_circuit(prep_circuit, f3_circuit, measure_circuit)
    visualization.plot_histogram(count_f3)
    return


@app.cell
def _(qiskit):
    # The results match our predictions!
    # For purposes of reproducibility, the Qiskit version is
    qiskit.__qiskit_version__
    return


if __name__ == "__main__":
    app.run()

