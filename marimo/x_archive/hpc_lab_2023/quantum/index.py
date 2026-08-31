import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Quantum Notebooks

        Welcome to the quantum notebooks! 

        This repository contains a collection of quantum circuits and quantum algorithms in Python using [Qiskit](https://www.qiskit.org/). Their goal is to provide a hands-on approach to quantum computing: on the Jupyter notebooks, you will find implementations of what is usually only discussed theoretically.

        Lastly, the codes are paired with dicussions on my [blogs about quantum computing](https://ivaniscoding.github.io/tags/quantum/). The discussions on the blog try to build an intuition behind the codes discussed here, so I invite you to check them out. 

        ## List of Codes

        ### Basic Quantum Computing

        Explore the building blocks of quantum, starting with single qubit systems and going up to multiple qubit systems.

        * **[The Qubit](./Basic_Quantum/Qubit.ipynb)**
        * **[Quantum Gates](./Basic_Quantum/Quantum-Gates.ipynb)**
        * **[Quantum Measurements](./Basic_Quantum/Measurements.ipynb)**
        * **[Multiple Qubits](./Basic_Quantum/Multiple-Qubits.ipynb)**
        * **[Uncomputation](./Basic_Quantum/Uncomputation.ipynb)**

        ### Quantum Communication

        Explore how entanglement can be used to send information. 

        * **[Quantum Teleportation](./Quantum_Communication/Teleportation.ipynb)**
        * **[Superdense Coding](./Quantum_Communication/Superdensecoding.ipynb)**

        ### Algorithms

        Explore the quantum advantage: algorithms that show how quantum computers can outperform their classical counterparts.

        * **[Deutsch's Algorithm](./Algorithms/Deutsch.ipynb)**
        * **[Deutsch-Jozsa's Algorithm](./Algorithms/Deutsch-Jozsa.ipynb)**
        * **[Bernstein-Vazirani's Algorithm](./Algorithms/Bernstein-Vazirani.ipynb)**
        * **[Grover's Algorithm](./Algorithms/Grover.ipynb)**
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

