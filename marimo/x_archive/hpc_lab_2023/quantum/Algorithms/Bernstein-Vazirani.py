import marimo

__generated_with = "0.15.2"
app = marimo.App()


app._unparsable_cell(
    r"""
    # Bernstein-Vazirani
    # '%matplotlib inline' command supported automatically in marimo
    from random import randint
    from qiskit import *
    """,
    name="_"
)


@app.cell
def _(randint):
    # Step 0: Generate a secret string
    secret_string = bin(randint(0, 2**7 - 1))[2:]
    N = len(secret_string)
    return N, secret_string


@app.cell
def _(N, QuantumCircuit):
    # Step 1: Prepare the state with superpositions
    prep_circuit = QuantumCircuit(N+1, N)
    prep_circuit.x(N)  # working qubit starts with |1>
    prep_circuit.h(range(N+1))
    prep_circuit.barrier()
    prep_circuit.draw(output="mpl")
    return (prep_circuit,)


@app.cell
def _(N, QuantumCircuit, secret_string):
    # Step 2: send the query to the quantum oracle. Here, we prepare the quantum oracle
    oracle_circuit = QuantumCircuit(N+1, N, name="Blackbox")

    for index, value in enumerate(secret_string):
        if value == "1":
            oracle_circuit.cx(index, N)  # XOR with working qubit

    oracle_circuit.barrier()
    oracle_circuit.draw(output="mpl")
    return (oracle_circuit,)


@app.cell
def _(N, QuantumCircuit):
    # Step 3: Apply Hadamard gates and measure
    measure_circuit = QuantumCircuit(N+1, N)
    measure_circuit.h(range(N))
    measure_circuit.measure(
        list(range(N)), 
        list(reversed(range(N)))
    )
    measure_circuit.draw(output="mpl")
    return (measure_circuit,)


@app.cell
def _(N, QuantumCircuit, measure_circuit, oracle_circuit, prep_circuit):
    # We now merge the circuit together
    circuit = QuantumCircuit(N+1, N)
    circuit += prep_circuit
    circuit.append(oracle_circuit.to_instruction(), range(N+1))  # We do this to "hide" the oracle from the viewer
    circuit.barrier()
    circuit += measure_circuit
    circuit.draw(output="mpl")
    return (circuit,)


@app.cell
def _(Aer, circuit, execute, visualization):
    # We simulate to discover the hidden string
    simulator = Aer.get_backend("qasm_simulator")
    job = execute(circuit, simulator, shots=512)
    results = job.result()
    visualization.plot_histogram(results.get_counts())
    return (results,)


@app.cell
def _(results, secret_string):
    # Was the guess right? Let's discover
    print("Hidden string {}\nGuess {}".format(
        secret_string, list(results.get_counts().keys())[0])
    )
    if list(results.get_counts().keys())[0] == secret_string:
        print("The guess was correct!")
    else:
        print("The guess was not correct")
    return


@app.cell
def _(qiskit):
    # For purposes of reproducibility, the Qiskit version is
    qiskit.__qiskit_version__
    return


if __name__ == "__main__":
    app.run()

