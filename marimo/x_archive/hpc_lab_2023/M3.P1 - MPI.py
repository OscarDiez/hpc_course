import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # M3.P1 MPI
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![image.png](attachment:0a8e7611-1dc0-482d-9e24-e88e538518ab.png)

        ## Message Passage Interface 
        The Message Passing Interface (MPI) is a standardized and portable message-passing system that defines the syntax and semantics of a core of library routines useful to a wide range of users writing portable message-passing programs in C, C++, and Fortran.

        ### Programming model
        MPI programming model features:

         - Distributed programming model. Also data parallel
         - Hardware platforms: distributed, shared and hybrid
         - Parallelism is explicit. The programmer is responsible for implementing all parallel constructs.

        The number of tasks dedicated to run a parallel program is static. New tasks can not be dynamically spawned during run time. However, MPI-2 addressed this issue.

        ### Programming model
        MPI benefits are:

        - Portability: There is no need to modify your source code when you port your application to a different platform that supports (and is compliant with) the MPI standard.

        - Standardization: MPI is the only message passing library which can be considered a standard. It is supported on virtually all HPC platforms. Practically, it has replaced all previous message passing libraries.

        - Functionality: Over 115 routines are defined in the MPI-1 alone.

        - Availability: A variety of implementations are available, both vendor and public domain (see below).

        ### Checking the version 
        In this cluster MPI is installed already. Normally this is installed by the supercomputer center in one of the available versions which may be (but not limited to):

        Open MPI
        Intel MPI
        MPICH2
        SGI's MPT
        and so on.

        You need to load the appropriate module (e.g., module load openmpi) prior to running code containing MPI stuffs. If your load was successful, you should be able to type mpiexec --version and see something similar to this:

        ```
        $ mpiexec --version
        mpiexec &#40;OpenRTE&#41; 1.8.8

        Report bugs to http://www.open-mpi.org/community/help/
        MPI Header files:
        C/C++: #include <mpi>
        ForTran: include 'mpif.h'
        Compiler wrappers:
        Intel: (icc -lmpi) and (ifort -lmpi)
        GNU: mpicc, mpif77, mpif90, mpicxx
        ```

        Run the code below to see what version is intalled in this cluster:

        """
    )
    return


@app.cell
def _():
    import subprocess
    subprocess.run('cd ./mpi\nmpiexec --version', shell=True)
    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""



        Depending on the programming language you use, you will need to include the proper headed files and compile the program.


        **MPI Header files:** 
        ```C/C++: #include <mpi>
        ForTran: include 'mpif.h'
        ```

        **Compiler wrappers:**
        ```Intel: (icc -lmpi) and (ifort -lmpi)
        GNU: mpicc, mpif77, mpif90, mpicxx
        ```


        ### MPI program stucture

        MPI uses objects called communicators and groups to define which collection of processes may communicate with each other. Most MPI routines require you to specify a communicator as an argument.

        ![image.png](attachment:c789f932-6ec2-4254-9f85-ee74e8b74074.png)

        #### MPI program schema
        It is very important to learn the MPI program structure before we proceed to coding message passing programs. Following are the common functions used while developing an MPI code:


        **MPI Init:**

        ``MPI_Init (&argc,&argv)
        MPI_INIT (ierr)
        MPI intialisation.
        ``

        **MPI_Comm_size:**

        ``MPI_Comm_size (comm,&size)
        MPI_COMM_SIZE (comm,size,ierr)
        ``
        Determines the number of processes in the group associated with a communicator.

        **MPI_Comm_rank:**

        ``
        MPI_Comm_rank (comm,&rank)
        MPI_COMM_RANK (comm,rank,ierr)
        ``
        Determines the rank (task ID) of the calling process within the communicator. Value 0...p-1

        **MPI_Abort:**

        ``
        MPI_Abort (comm,errorcode)
        MPI_ABORT (comm,errorcode,ierr)
        ``
        Terminates all MPI processes associated with the communicator.

        **MPI_Finalize:**

        ``
        MPI_Finalize ()
        MPI_FINALIZE (ierr)
        ``
        Terminates the MPI execution environment. This function should be the last MPI routine called.


        #### Example 3.1 Hello World!

        In this exercise, we will write  a basic MPI hello world code and also discuss how to run an MPI program. The lesson will cover the basics of initializing MPI and running an MPI job across several processes. The code is already in the folder mpi, and it is called mpi_hello.c:

        ```
        #include <mpi.h>
        #include <stdio.h>

        int main(int argc, char** argv) {
            // Initialize the MPI environment
            MPI_Init(NULL, NULL);

            // Get the number of processes
            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            // Get the rank of the process
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            // Get the name of the processor
            char processor_name[MPI_MAX_PROCESSOR_NAME];
            int name_len;
            MPI_Get_processor_name(processor_name, &name_len);

            // Print off a hello world message
            printf("Hello world from processor %s, rank %d out of %d processors\n",
                   processor_name, world_rank, world_size);

            // Finalize the MPI environment.
            MPI_Finalize();
        }
        ```


        Let's get introduced with the functions and environment veriables used in the code:

        - **MPI_Init**, all of MPI's global and internal variables are constructed.
        - **MPI_Comm_size** returns the size of a communicator and the built-in MPI_COMM_WORLD encloses all of the processes in the job, so this call should return the amount of processes that were requested for the job.
        - **MPI_Comm_rank** returns the rank of a process in a communicator. Each process inside of a communicator is assigned an incremental rank starting from zero.
        - **MPI_Finalize** is used to clean up the MPI environment. No more MPI calls can be made after this one.

        As you can see this is a C program, so we will need to compile it. 

        """
    )
    return


@app.cell
def _(subprocess):
    subprocess.run('cd ./mpi\nmpicc mpi_hello.c -o mpi_hello.out\nmpirun -n 2 -hostfile hostfile mpi_hello.out', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        If you want to change the code you can do it from the left panel. Select your file inside folder mpi and modify it (do not forget to save it with File/save). 
        Do not forget that you will need to compile it. You can reuse the previous code above or run the same commands directlyl in the terminal.

        You can launch a terminal directly from jupyter launcher or just from docker (if from docker you can sue the `bash` command from docker to get a proper bash terminal). 

        To do it from jupyter, start a Terminal in the Launcher tab. You can use the terminal to launchany command, including slurm jobs via commands.

        ![image15.png](attachment:1e20047e-2be2-4a72-8991-ed7ef27df119.png)

        Try to change thenumber of MPI tasks when you run mpirun `-n 4` You do not need tocompile the program again for this.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ## MPI Point to Point Communication
        MPI point-to-point operations typically involve message passing between two, and only two, different MPI tasks. One task is performing a send operation and the other task is performing a matching receive operation. Different types of send and receive routines:

        - Synchronous send
        - Blocking send / blocking receive
        - Non-blocking send / non-blocking receive
        - Buffered send
        - Combined send/receive
        - "Ready" send

        Any type of send routine can be paired with any type of receive routine.

        ### MPI Send / Receive

        MPI point-to-point communication routines generally have an argument list that takes one of the following formats:

        ``MPI_Send (&buf,count,datatype,dest,tag,comm)
        MPI_SEND (buf,count,datatype,dest,tag,comm,ierr)``

        **Buffer**: Program address space that references the data that is to be sent or received. In most cases, this is simply the variable name that is be sent/received. For C programs, this argument is passed by reference and usually must be prepended with an ampersand: &var1.

        **Data Count**: Indicates the number of data elements of a particular type to be sent.

        **Data Type**: For reasons of portability, MPI predefines its elementary data types.

        - `MPI_CHAR` – signed char
        - `MPI_INT` – signed int
        - `MPI_FLOAT` – float
        - `MPI_DOUBLE` – double

        You can also create your own derived data types.

        **Destination**: An argument to send routines that indicates the process where a message should be delivered. Specified as the rank of the receiving process.

        **Source**: An argument to receive routines that indicates the originating process of the message. Specified as the rank of the sending process. This may be set to the wild card MPI_ANY_SOURCE to receive a message from any task.

        **Tag**: Arbitrary non-negative integer assigned by the programmer to uniquely identify a message. Send and receive operations should match message tags. For a receive operation, the wild card MPI_ANY_TAG can be used to receive any message regardless of its tag. The MPI standard guarantees that int can be used as tags, but most implementations allow a much larger range than this.

        **Communicator**: Indicates the communication context, or set of processes for which the source or destination fields are valid. Unless the programmer is explicitly creating new communicators, the predefined communicator MPI_COMM_WORLD is usually used.

        **Status**: For a receive operation, indicates the source of the message and the tag of the message. In C, this argument is a pointer to a predefined structure MPI_Status (ex. stat.MPI_SOURCE stat.MPI_TAG). Additionally, the actual number of bytes received are obtainable from Status via the MPI_Get_count routine.

        **Request**: Used by non-blocking send and receive operations. Since non-blocking operations may return before the requested system buffer space is obtained, the system issues a unique "request number". The programmer uses this system assigned "handle" later (in a WAIT type routine) to determine completion of the non-blocking operation. In C, this argument is a pointer to a predefined structure MPI_Request.

        Blocking sends: `MPI_Send(buffer,count,type,dest,tag,comm)` Non-blocking sends: `MPI_Isend(buffer,count,type,dest,tag,comm,request)` Blocking receive: `MPI_Recv(buffer,count,type,source,tag,comm,status)` Non-blocking receive: `MPI_Irecv(buffer,count,type,source,tag,comm,request)`

        **MPI_Send**:Basic blocking send operation. Routine returns only after the application buffer in the sending task is free for reuse.

        ``MPI_Send (&buf,count,datatype,dest,tag,comm)
        MPI_SEND (buf,count,datatype,dest,tag,comm,ierr)
        MPI_Recv (&buf,count,datatype,source,tag,comm,&status)
        MPI_RECV (buf,count,datatype,source,tag,comm,status,ierr)``

        **Synchronous blocking send**:Send a message and block until the application buffer in the sending task is free for reuse and the destination process has started to receive the message.

        ``MPI_Ssend (&buf,count,datatype,dest,tag,comm)
        MPI_SSEND (buf,count,datatype,dest,tag,comm,ierr)``

        **Buffered blocking send**:permits the programmer to allocate the required amount of buffer space into which data can be copied until it is delivered. Insulates against the problems associated with insufficient system buffer space.

        ``MPI_Bsend (&buf,count,datatype,dest,tag,comm)
        MPI_BSEND (buf,count,datatype,dest,tag,comm,ierr)``

        ### Exercise 3.2 MPI ping-pong with send and receive

        The following code shows an MPI ping-pong program developed by Wes Kendall (reproduced).



        ```
        #include <mpi.h>
        #include <stdio.h>
        #include <stdlib.h>

        int main(int argc, char** argv) {
          const int PING_PONG_LIMIT = 10;

          // Initialize the MPI environment
          MPI_Init(NULL, NULL);
          // Find out rank, size
          int world_rank;
          MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
          int world_size;
          MPI_Comm_size(MPI_COMM_WORLD, &world_size);

          // We are assuming at least 2 processes for this task
          if (world_size != 2) {
            fprintf(stderr, "World size must be two for %s\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
          }

          int ping_pong_count = 0;
          int partner_rank = (world_rank + 1) % 2;
          while (ping_pong_count < PING_PONG_LIMIT) {
            if (world_rank == ping_pong_count % 2) {
              // Increment the ping pong count before you send it
              ping_pong_count++;
              MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
              printf("%d sent and incremented ping_pong_count %d to %d\n",
                     world_rank, ping_pong_count, partner_rank);
            } else {
              MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE);
              printf("%d received ping_pong_count %d from %d\n",
                     world_rank, ping_pong_count, partner_rank);
            }
          }
          MPI_Finalize();
        }
        ```



        """
    )
    return


@app.cell
def _(subprocess):
    subprocess.run('cd ./mpi\nmpicc mpi_pingpong.c -o mpi_pingpong.out\nmpirun -n 2 -hostfile hostfile mpi_pingpong.out', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you want to change the code you can do it from the left panel. Select your file inside folder mpi and modify it (do not forget to save it with File/save). Do not forget that you will need to compile it. You can reuse the previous code above or run the same commands directlyl in the terminal.

        You can launch a terminal directly from jupyter launcher or just from docker (if from docker you can sue the bash command from docker to get a proper bash terminal).

        To do it from jupyter, start a Terminal in the Launcher tab. You can use the terminal to launchany command, including slurm jobs via commands.


        This is the end of the MPI assignment! 

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

