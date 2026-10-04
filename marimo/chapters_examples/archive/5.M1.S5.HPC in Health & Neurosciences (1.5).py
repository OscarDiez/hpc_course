import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Sequence Alignment Using BioPython in Colab

        **Objective**: To perform a basic sequence alignment using the BioPython library, which is commonly used in bioinformatics for sequence analysis.

        **Background**: Sequence alignment is a crucial step in genomic data analysis, where sequences of DNA, RNA, or proteins are arranged to identify regions of similarity that may indicate functional, structural, or evolutionary relationships.

        ## Step 1: Install BioPython

        BioPython is a powerful library that provides tools for working with biological data in Python. Let's start by installing BioPython in the Colab environment.

        ```python
        # Install BioPython
        # (use marimo's built-in package management features instead) !pip install biopython

        """
    )
    return


@app.cell
def _():
    # Install BioPython
    # (use marimo's built-in package management features instead) !pip install biopython
    return


@app.cell
def _():
    # Import necessary modules
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment

    # Define two sequences to align
    seq1 = "GATTACA"
    seq2 = "GCATGCU"

    # Perform global alignment
    alignments = pairwise2.align.globalxx(seq1, seq2)

    # Print the alignments
    for alignment in alignments:
        print(format_alignment(*alignment))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ### Explanation of the Example:

        - **BioPython Installation**: The example begins by installing the BioPython library in the Colab environment, which students can easily do.
  
        - **Sequence Alignment**: The example uses the `pairwise2` module to perform a simple global alignment of two DNA sequences. This is a fundamental bioinformatics task that students can understand and visualize immediately.

        - **Interpretation**: The example includes the interpretation of the alignment, helping students understand the significance of the output.

        ### Why This Example Works:

        - **Simplicity**: The example is simple and doesn't require a deep understanding of HPC or complex bioinformatics tools. It focuses on a core concept—sequence alignment—using a library that is accessible and easy to use in Colab.
  
        - **Practicality**: Students can run this code directly in Colab and immediately see the results, reinforcing the concepts discussed in the chapter.

        - **Scalability**: While the example uses small sequences for simplicity, it introduces students to tools that can be scaled up for more complex tasks, similar to those run on HPC systems.

        This example is a great way to bridge the gap between theory and practice, giving students a hands-on experience with tools used in the field of health and neurosciences.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to GROMACS in HPC

        GROMACS is a versatile package used for molecular dynamics simulations and energy minimization. It is widely employed in High-Performance Computing (HPC) environments for simulating the interactions of particles like proteins, lipids, and other biological molecules.

        In this practical session, you will learn how to install GROMACS in a Google Colab environment, create a simple box of water molecules, scale the dimensions of the box, and understand the steps involved in these processes.

        Let's get started!

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 1: Installing GROMACS

        Before we begin working with GROMACS, we need to install it in our Google Colab environment. The command above installs GROMACS along with its dependencies. Once the installation is complete, GROMACS will be ready for use.

        You should see the output confirming the successful installation.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Install GROMACS
    !apt-get install -y gromacs
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 2: Creating a Water Box

        Now, we will create a cubic box filled with water molecules using GROMACS. This step demonstrates how GROMACS can generate a system that you can later manipulate or simulate.

        - The `gmx solvate` command generates a cubic box filled with water molecules.
        - The `-cs spc216` flag specifies the water model.
        - The `-box 2 2 2` flag defines the size of the box (2x2x2 nm).

        After running this code, the output will show the first few lines of the generated `.gro` file, which includes the coordinates of the water molecules. This gives you a look at how GROMACS structures the data for molecular simulations.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Generate a simple cubic box of water molecules with specified dimensions
    !gmx solvate -cs spc216 -o water_box.gro -box 2 2 2

    # Display the first few lines of the generated water_box.gro file to show its content
    !head -n 10 water_box.gro
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 3: Scaling the Water Box

        We will now scale the water box using the `gmx editconf` command. This command allows us to adjust the dimensions of the box, simulating a scenario where you might need to modify system boundaries.

        - The `gmx editconf` command reads the original `water_box.gro` file.
        - The `-scale 1.2 1.2 1.2` flag scales the box dimensions by 20% in each direction.

        The output will display the first few lines of the scaled `.gro` file, where you can observe the changes in atomic coordinates, reflecting the new, larger dimensions of the water box.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Modify the water box dimensions slightly as an example of changing the system
    !gmx editconf -f water_box.gro -o water_box_scaled.gro -scale 1.2 1.2 1.2

    # Display the first few lines of the modified water_box_scaled.gro file to show the change
    !head -n 10 water_box_scaled.gro
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 4: HPC Concepts in GROMACS

        In High-Performance Computing (HPC), tasks like molecular dynamics simulations are distributed across multiple processors. GROMACS is designed to take advantage of such environments, allowing it to perform large-scale simulations more efficiently.

        For example, when running a GROMACS simulation on an HPC cluster, the workload is divided among multiple CPUs or GPUs. This parallel processing reduces the time needed to simulate complex systems, such as proteins in a solvent.

        ### Additional Example: Running a Parallel Simulation (Hypothetical)

        In an actual HPC environment, you could run a simulation in parallel across multiple processors. Here's a hypothetical command:

        ```python
        # Hypothetical example to run GROMACS in parallel (this requires an actual HPC setup)
        # !mpirun -np 4 gmx_mpi mdrun -s topol.tpr -deffnm simulation

        # Note: The above command would run a simulation using 4 processors in parallel

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Visualizing GROMACS Data: 2D Projection of Water Molecule Positions

        In this section, we will visualize the positions of water molecules generated by GROMACS using a simple 2D scatter plot. This plot will help us understand the spatial distribution of molecules within the box we created and scaled earlier.

        We will use the `MDAnalysis` Python library to parse the `.gro` file containing the molecular data and `matplotlib` to create the plot. The `.gro` file contains the atomic coordinates of the water molecules, which we will extract and plot in two dimensions.

        This visualization is a basic way to get an idea of how the molecules are arranged within the box. For more complex and detailed visualizations, tools like VMD, PyMOL, or UCSF Chimera can be used.

        """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install MDAnalysis matplotlib
    return


@app.cell
def _():
    import MDAnalysis as mda
    import matplotlib.pyplot as plt

    # Load the water box file generated by GROMACS
    u = mda.Universe("water_box_scaled.gro")

    # Extract the positions of the atoms
    positions = u.atoms.positions

    # Plot the coordinates of the atoms in 2D
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], s=10, c='blue', alpha=0.5)
    plt.title("2D Projection of Water Molecule Positions")
    plt.xlabel("X coordinate (nm)")
    plt.ylabel("Y coordinate (nm)")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    from google.colab import files

    # Download the water_box_scaled.gro file
    files.download('water_box_scaled.gro')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        1. **Library Installation**: We start by installing the required Python libraries, `MDAnalysis` for handling molecular data, and `matplotlib` for creating plots.

        2. **Loading the `.gro` File**: The `.gro` file generated by GROMACS contains the coordinates of all atoms in the system. We load this file using `MDAnalysis` to access the atomic positions.

        3. **Extracting Positions**: We extract the atomic coordinates from the loaded `.gro` file. These coordinates represent the positions of the water molecules in the box.

        4. **Plotting the Positions**: Using `matplotlib`, we create a 2D scatter plot of the x and y coordinates of the atoms. This plot gives a visual representation of how the water molecules are distributed within the box.

        The resulting plot shows the arrangement of water molecules in two dimensions, allowing you to visualize the spatial distribution within the simulation box.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

