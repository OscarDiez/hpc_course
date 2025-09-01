import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Uncomputation
        """
    )
    return


app._unparsable_cell(
    r"""
    # '%matplotlib inline' command supported automatically in marimo
    from qiskit import *
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To compute $a \wedge b \wedge c$, we need two working qubits. We need to split the computation into two parts: $a \wedge b \wedge c = (a \wedge b) \wedge c$

        Moreover, it is good practice to clean the working qubit that only contains an intermediate result: that is called uncomputation
        """
    )
    return


@app.cell
def _(QuantumCircuit):
    circuit = QuantumCircuit(5, 2)
    return (circuit,)


@app.cell
def _(circuit):
    # Steps to change qubits to 1 for test purposes
    _ = circuit.x(0)
    _ = circuit.x(1)
    _ = circuit.x(2)
    _ = circuit.barrier()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Firstly, we calculate $a \wedge b$ in the first working qubit.
        """
    )
    return


@app.cell
def _(circuit):
    _ = circuit.ccx(0, 1, 3)
    circuit.draw(output="mpl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Secondly, we calculate $(a \wedge b) \wedge c$ using both the third qubit and the first working qubit. The second working qubit is the target.
        """
    )
    return


@app.cell
def _(circuit):
    _ = circuit.ccx(2, 3, 4)
    circuit.draw(output="mpl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We now uncompute $a \wedge b$ in the first working qubit. That way, we can reuse it later if needed.

        To do so, we remember that the Toffoli gate is the inverse of itself: $(a \wedge b) \oplus (a \wedge b) = 0$
        """
    )
    return


@app.cell
def _(circuit):
    _ = circuit.ccx(0, 1, 3)
    circuit.draw(output="mpl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Lastly, we measure both working qubits: one should contain the AND and the other should always be zero
        """
    )
    return


@app.cell
def _(circuit):
    # Measure (x^y^z) and working qubuit in classical bit
    _ = circuit.barrier()
    _ = circuit.measure(4, 0)
    _ = circuit.measure(3, 1)
    return


@app.cell
def _(circuit):
    circuit.draw(output="mpl")
    return


@app.cell
def _(Aer, circuit, execute):
    # Simulate
    simulator = Aer.get_backend("qasm_simulator")
    job = execute(circuit, backend=simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()
    return (counts,)


@app.cell
def _(counts, visualization):
    # Plot results
    visualization.plot_histogram(counts)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

