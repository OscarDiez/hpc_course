import marimo

__generated_with = "0.15.2"
app = marimo.App()


app._unparsable_cell(
    r"""
    #Deutsch-Jozsa algorithm
    # '%matplotlib inline' command supported automatically in marimo
    from qiskit import *
    """,
    name="_"
)


@app.cell
def _():
    N = 4  # defining how many qubits will be used
    return (N,)


@app.cell
def _(N, QuantumCircuit):
    # Step 0: Prepare superposition state
    prep_circuit = QuantumCircuit(N + 1, N)
    prep_circuit.x(N)  # working qubit starts with |1>
    prep_circuit.h(range(N+1))  # Generate all superpositions
    prep_circuit.barrier()
    prep_circuit.draw(output="mpl")
    return (prep_circuit,)


@app.cell
def _(N, QuantumCircuit):
    # Step 1: Send input to blackbox 
    # Here, we will experiment with 3 blackboxes
    # The first is the constant function f(x) = 1
    constant_circuit = QuantumCircuit(N+1, N)
    constant_circuit.x(N)
    constant_circuit.barrier()
    constant_circuit.draw(output="mpl")
    return (constant_circuit,)


@app.cell
def _(N, QuantumCircuit):
    # The second is blackbox implements f(x) = x mod 2, which is a balanced function
    mod2_circuit = QuantumCircuit(N+1, N)
    mod2_circuit.cx(0, N)
    mod2_circuit.barrier()
    mod2_circuit.draw(output="mpl")
    return (mod2_circuit,)


@app.function
# The third circuit implements a function that has period 4 and has values {1, 0, 0, 1}
# Before, we code the circuit, here is the representation
def blackbox_3(x):
    if x % 4 == 0 or x % 4 == 3:
        return 1
    else:
        return 0


@app.cell
def _(N, QuantumCircuit):
    periodic_circuit = QuantumCircuit(N+1, N)
    periodic_circuit.cx(0, N)
    periodic_circuit.cx(1, N)
    periodic_circuit.x(N)
    periodic_circuit.barrier()
    periodic_circuit.draw(output="mpl")
    return (periodic_circuit,)


@app.cell
def _(N, QuantumCircuit):
    # Step 2: Apply Hadamard to all qubits and measure
    measure_circuit = QuantumCircuit(N+1, N)
    measure_circuit.h(range(N))
    measure_circuit.measure(range(N), range(N))
    measure_circuit.draw(output="mpl")
    return (measure_circuit,)


@app.cell
def _(measure_circuit, periodic_circuit, prep_circuit):
    # An example of what the assembled circuit looks like
    (prep_circuit + periodic_circuit + measure_circuit).draw(output="mpl")
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
    # Recall that all the measurements are 0 if f is constant, and at least one measurement is 1 if f is balanced
    return


@app.cell
def _(
    constant_circuit,
    measure_circuit,
    prep_circuit,
    simulate_circuit,
    visualization,
):
    # For a constant function, we expect it to be all 0s
    count_constant = simulate_circuit(prep_circuit, constant_circuit, measure_circuit)
    visualization.plot_histogram(count_constant)
    return


@app.cell
def _(
    measure_circuit,
    mod2_circuit,
    prep_circuit,
    simulate_circuit,
    visualization,
):
    # For balanced, we expect at least one measurement to be 1
    count_mod2 = simulate_circuit(prep_circuit, mod2_circuit, measure_circuit)
    visualization.plot_histogram(count_mod2)
    return


@app.cell
def _(
    measure_circuit,
    periodic_circuit,
    prep_circuit,
    simulate_circuit,
    visualization,
):
    # We try again with a balanced function to verify that at least one is zero
    count_periodic = simulate_circuit(prep_circuit, periodic_circuit, measure_circuit)
    visualization.plot_histogram(count_periodic)
    return


@app.cell
def _(qiskit):
    # The results match our predictions!
    # For purposes of reproducibility, the Qiskit version is
    qiskit.__qiskit_version__
    return


if __name__ == "__main__":
    app.run()

