import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # FPGA AND Gate Simulation with Verilator in Colab

        This guide walks through the process of simulating an AND gate implemented in Verilog and tested using a C++ testbench in Google Colab.

        ### Step 1: Install Verilator

        To simulate Verilog code in Colab, you need to install Verilator, a tool for Verilog HDL simulation.

        ```bash
        !apt-get install verilator
        ```
        Step 2: Create the Verilog AND Gate Module (and_gate.v)
        The following Verilog code defines a simple AND gate. We'll create a file called and_gate.v and write the module code into it.

        ```
        and_gate_verilog = \"\"\"
        module and_gate (
            input wire a,
            input wire b,
            output wire y
        );
            assign y = a & b;
        endmodule
        \"\"\"

        with open("and_gate.v", "w") as f:
            f.write(and_gate_verilog)
        ```
        Step 3: Create the C++ Testbench (main.cpp)
        Next, we create a C++ testbench that will test all possible input combinations for the AND gate. This file will be written to main.cpp.

        Step 4: Compile the Verilog Code with Verilator
        We use Verilator to compile the Verilog file into C++ and link it with the testbench. The --cc flag tells Verilator to generate C++ code, and the --exe flag specifies the C++ testbench.

        ```
        !verilator --cc and_gate.v --exe main.cpp
        ```

        Step 5: Build the Simulation
        After generating the C++ code, we use make to compile the simulation. This command builds the simulation from the generated files.

        ```
        !make -j -C obj_dir -f Vand_gate.mk Vand_gate
        ```

        Step 6: Run the Simulation and Capture the Output
        Finally, we run the compiled simulation and print the results.

        ```
        simulation_output = !./obj_dir/Vand_gate
        print("\n".join(simulation_output))
        ```

        Explanation
        Verilog AND Gate: The Verilog module implements a basic two-input AND gate.
        C++ Testbench: The testbench runs through all input combinations (00, 01, 10, 11) and displays the corresponding output.
        Verilator: This tool translates Verilog into C++ and links it to the testbench for simulation.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Install Verilator
    !sudo apt-get update > /dev/null
    !sudo apt-get install -y verilator > /dev/null
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Verify Verilator installation
    !verilator --version
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Step 2: Create the Verilog AND gate module (and_gate.v)
    and_gate_verilog = \"\"\"
    module and_gate (
        input wire a,
        input wire b,
        output wire y
    );
        assign y = a & b;
    endmodule
    \"\"\"

    with open(\"and_gate.v\", \"w\") as f:
        f.write(and_gate_verilog)

    # Step 3: Create the C++ testbench (main.cpp) with corrected string literals
    main_cpp = \"\"\"
    #include \"Vand_gate.h\"
    #include \"verilated.h\"
    #include <iostream>

    int main(int argc, char **argv) {
        // Initialize Verilated
        Verilated::commandArgs(argc, argv);
        Vand_gate* top = new Vand_gate;

        // Display header
        std::cout << \"Simulating AND Gate:\\n\";
        std::cout << \"a | b | y\\n\";
        std::cout << \"---------\\n\";

        // Iterate through all possible input combinations
        for(int i = 0; i < 4; i++) {
            // Set inputs
            top->a = (i & 0x1);
            top->b = ((i >> 1) & 0x1);

            // Evaluate the model
            top->eval();

            // Display the results
            std::cout << top->a << \" | \" << top->b << \" | \" << top->y << \"\\n\";
        }

        // Finalize simulation
        top->final();
        delete top;
        return 0;
    }
    \"\"\"

    with open(\"main.cpp\", \"w\") as f:
        f.write(main_cpp)

    # Step 4: Compile the Verilog code with Verilator
    # The --cc flag tells Verilator to generate C++ code
    # The --exe flag specifies that we are providing a C++ testbench
    !verilator --cc and_gate.v --exe main.cpp

    # Step 5: Build the simulation using make
    # The -j flag allows make to run in parallel
    !make -j -C obj_dir -f Vand_gate.mk Vand_gate

    # Step 6: Run the simulation and capture the output
    simulation_output = !./obj_dir/Vand_gate

    # Display the simulation results
    print(\"\n\".join(simulation_output))
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ------
        -------
        STOP HERE. Codee below is not working properly in in Colab
        -------
        -------

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenCL Programming for FPGAs

        ## OpenCL Overview
        - **Framework for parallel programming**
        - **Supports heterogeneous systems**

        ## Advantages for FPGAs
        - Portability across hardware
        - High-level abstraction for parallelism
        - Simplifies parallel computation
        - Reusable code across different devices

        ## Use Cases
        - Image processing
        - Data analytics

        ## Tool Support
        - Intel and Xilinx OpenCL implementations

        ## Example: Vector Addition in OpenCL

        ```c
        __kernel void vector_add(__global const int* A, __global const int* B, __global int* C, int N) {
            int id = get_global_id(0);
            if (id < N) {
                C[id] = A[id] + B[id];
            }
        }

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ### Colab Code to Run (OpenCL Setup):
        Unfortunately, Colab does not natively support FPGA execution or OpenCL directly, but you can experiment with OpenCL code on a local machine or through specialized cloud FPGA services like Intel DevCloud or Xilinx Vitis.

        Here’s an example of how you can structure your code for local execution:

        #### Colab: Preparing OpenCL environment
        ```bash
        # Install dependencies (if running locally or on cloud platforms like Intel DevCloud)
        !apt-get install ocl-icd-opencl-dev
        !apt-get install clinfo

        # Check if OpenCL is available
        !clinfo

        """
    )
    return


app._unparsable_cell(
    r"""
    #include <CL/cl.h>
    #include <iostream>
    #include <vector>

    // Size of the vector
    #define N 1024

    // OpenCL kernel for vector addition
    const char* kernelSource = R\"(
    __kernel void vector_add(__global const int* A, __global const int* B, __global int* C, int N) {
        int id = get_global_id(0);
        if (id < N) {
            C[id] = A[id] + B[id];
        }
    }
    )\";

    int main() {
        // Host data
        std::vector<int> A(N, 1); // Initialize A with ones
        std::vector<int> B(N, 2); // Initialize B with twos
        std::vector<int> C(N, 0); // Output array

        // Initialize OpenCL context, queue, and program (skipping detailed error checking for brevity)
        cl_platform_id platform_id;
        cl_device_id device_id;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;

        // Allocate memory on the device and copy data to the device

        // Execute the OpenCL kernel

        // Copy the result back to host and verify the output

        std::cout << \"Vector addition result: \" << C[0] << std::endl; // Example output
        return 0;
    }
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

