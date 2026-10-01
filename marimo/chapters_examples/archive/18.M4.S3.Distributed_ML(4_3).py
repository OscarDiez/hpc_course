import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Distributed Machine Learning in Colab

        In this notebook, we'll explore how to run distributed machine learning on Google Colab using three popular libraries: **TensorFlow Distributed**, **PyTorch Distributed**, and **Horovod**. We'll walk through the steps to set up each library, run distributed operations, and see the results.

        Since Colab only provides access to a single GPU, we'll simulate distributed environments. Although full multi-node functionality cannot be replicated on Colab, this approach gives you a practical feel for distributed training.

        ---

        ## Section 1: TensorFlow Distributed

        In this code, we are using **TensorFlow’s MirroredStrategy** to distribute training across multiple GPUs. In this setup, the model is mirrored on all available devices (GPUs), and TensorFlow handles splitting the input data and synchronizing the gradients after each batch.

        **MirroredStrategy** works by ensuring that each GPU has the same copy of the model and that all GPUs perform forward and backward propagation in parallel. After each batch, the gradients from each device are averaged using an All-Reduce operation to ensure that the model weights stay consistent across devices.

        This type of distributed training is useful when working with large datasets or models because it reduces the training time by leveraging multiple devices in parallel.

        ### Key Components:
        - **MirroredStrategy**: A TensorFlow strategy that mirrors the model across multiple GPUs and synchronizes after each batch.
        - **All-Reduce**: An operation that sums up the gradients from all GPUs and averages them, ensuring all GPUs have consistent weights.
        - **strategy.scope()**: A context manager that ensures the defined model is properly distributed across devices.

        If you weren't using a distributed strategy like MirroredStrategy, your model would only run on a single device, and all the training steps would be performed sequentially. Using distributed strategies like this one speeds up training by distributing the workload.

        In this Colab notebook, we simulate the use of multiple GPUs, even though Colab only provides access to a single GPU, allowing you to learn the concepts without requiring a multi-GPU setup.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Install TensorFlow

        We need to install TensorFlow first, which is already included in Colab, but it’s good to ensure the latest version is installed.


        """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install tensorflow --upgrade
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Non-Distributed TensorFlow Code

        This code demonstrates a simple neural network model training on the MNIST dataset using TensorFlow. In this version, the code **does not** utilize distributed strategies, meaning that it runs on a single GPU or CPU.

        - **Loading the MNIST dataset**: The dataset is loaded using TensorFlow's built-in `tf.keras.datasets.mnist`, which provides images of handwritten digits. The pixel values of the images are scaled to between 0 and 1 for normalization.
  
        - **Model Architecture**: A basic feed-forward neural network is defined using the `tf.keras.Sequential()` model API. The architecture consists of:
          - A `Flatten` layer that converts each 28x28 image into a one-dimensional array of 784 values.
          - A `Dense` layer with 128 units and ReLU activation, which acts as a hidden layer.
          - A final `Dense` layer with 10 units and softmax activation for classifying the digits (0-9).

        - **Compilation**: The model is compiled with:
          - **Adam optimizer** for adjusting the weights.
          - **Sparse categorical crossentropy** as the loss function since the labels are integers (0-9).
          - **Accuracy metric** to evaluate performance during training.

        - **Training**: The model is trained for 5 epochs using the `.fit()` method on the training dataset (`x_train`, `y_train`). This is where the backpropagation happens to adjust weights.

        - **Evaluation**: The model is evaluated on the test dataset (`x_test`, `y_test`) using `.evaluate()` to check how well the model performs on unseen data.

        #### Why this is a non-distributed version:
        - No use of `tf.distribute.MirroredStrategy` or any distributed strategy API.
        - All computations happen on a single device (CPU or one GPU if available).
        - If you have multiple GPUs, only one will be used in this setup.

        #### Result:
        The model will output the following:
        1. **Training Progress**: During training, the accuracy and loss will be printed after each epoch.
        2. **Final Test Accuracy**: After training, the model will be evaluated on the test set, giving you the loss and accuracy, for example:

        """
    )
    return


@app.cell
def _():
    import tensorflow as tf

    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define a simple neural network without distributed strategy
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model on the dataset
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model
    model.evaluate(x_test, y_test)
    return (tf,)


@app.cell
def _(tf):
    strategy = tf.distribute.MirroredStrategy()
    (x_train_1, y_train_1), (x_test_1, y_test_1) = tf.keras.datasets.mnist.load_data()
    x_train_1, x_test_1 = (x_train_1 / 255.0, x_test_1 / 255.0)
    with strategy.scope():
        model_1 = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
        model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_1.fit(x_train_1, y_train_1, epochs=5)
    model_1.evaluate(x_test_1, y_test_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###Explanation:
        MirroredStrategy distributes training across all available GPUs, but here it's simulated on a single GPU in Colab.

        The MNIST dataset is used for training a simple neural network. After 5 epochs, the model will be evaluated on the test data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise: Modify the TensorFlow Distributed Model to run on multiple nodes

        Now that you have learned about TensorFlow Distributed with `MirroredStrategy`, let’s modify the model to add complexity and adjust some parameters.

        ### Task:
        1. Modify the model by adding an extra **Dense** layer with 64 units and `relu` activation.
        2. Change the number of **epochs** to 3 instead of 5.
        3. Rerun the training process and observe the changes in performance and training time.

        ### Steps:
        1. Locate the section where the model is defined.
        2. Add another layer to the model using `Dense(64, activation='relu')`.
        3. Reduce the number of epochs in the `model.fit()` function to 3.
        4. Run the training and observe the output.

        This exercise will help you understand how the complexity of a model impacts training time and how distributed strategies handle more complex models.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. **Setting up the `TF_CONFIG` environment variable**:
           - The `TF_CONFIG` environment variable is essential when running TensorFlow in a multi-worker setup.
           - It defines the cluster configuration, specifying the IP addresses of the workers in the cluster.
           - Each worker is assigned a role:
             - The `worker` type is responsible for training, and there may be multiple workers in the cluster.
             - The `index` parameter indicates which worker the current machine is (e.g., index 0 for the first worker, index 1 for the second, and so on).
           - The environment is set before restarting the runtime to ensure it takes effect.

        2. **Restarting the runtime**:
           - After defining `TF_CONFIG`, the runtime needs to be restarted manually to apply the cluster settings.
           - This ensures TensorFlow recognizes the distributed setup across multiple workers.

        3. **Initializing `MultiWorkerMirroredStrategy`**:
           - Once the runtime restarts, `MultiWorkerMirroredStrategy` is initialized, which allows the model to be replicated across all workers and synchronizes gradient updates during training.

        4. **Model definition and training**:
           - Inside the `strategy.scope()`, the model is built, compiled, and trained.
           - `strategy.scope()` ensures that the model’s computations are distributed across the cluster of workers, allowing for efficient distributed training.

        5. **Training and evaluation**:
           - The training process runs in parallel across all workers, allowing for faster training.
           - The `evaluate()` method computes accuracy on the test dataset, showing the model’s performance after distributed training.

        This approach is useful in scenarios where training datasets are large, and computation needs to be scaled across multiple nodes or machines.

        """
    )
    return


@app.cell
def _():
    import os
    import json

    # Step 1: Define the TF_CONFIG environment variable
    # Replace the worker addresses with the actual IPs of your nodes
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["worker1_ip:port", "worker2_ip:port"]
        },
        'task': {'type': 'worker', 'index': 0}  # Set 'index': 0 for the first worker, 1 for the second, etc.
    })

    # Restart runtime manually to apply the settings and re-run the next cells
    return (os,)


@app.cell
def _(tf):
    strategy_1 = tf.distribute.MultiWorkerMirroredStrategy()
    (x_train_2, y_train_2), (x_test_2, y_test_2) = tf.keras.datasets.mnist.load_data()
    x_train_2, x_test_2 = (x_train_2 / 255.0, x_test_2 / 255.0)
    with strategy_1.scope():
        model_2 = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
        model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_2.fit(x_train_2, y_train_2, epochs=5)
    model_2.evaluate(x_test_2, y_test_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Exercise

        1. **Modify the `TF_CONFIG`** to include at least 3 worker nodes.
           - Set up a cluster configuration with 3 workers by modifying the `TF_CONFIG` environment variable.
           - Ensure that the `index` and IP addresses for each worker are updated accordingly.

        2. **Experiment with different model architectures** within the `strategy.scope()`.
           - Try changing the architecture (e.g., adding more layers or changing the activation function) and observe how the distributed strategy handles the new model.

        3. **Scaling Learning Rate**:
           - Adjust the learning rate by multiplying the base rate by the number of workers. This is a best practice in distributed training to maintain effective learning across nodes.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ## Section 2: PyTorch Distributed

        In this example, we use **PyTorch's `torch.distributed`** package to simulate distributed training. We are simulating communication between multiple processes using the **Gloo backend**, which is optimized for CPU-based communication (though it can also work with GPUs). Each process in this simulation represents a "worker," and each worker holds its own tensor.

        The key operation we use here is **AllReduce**, which is responsible for summing the tensors from all processes and distributing the result back to each one. In a real-world distributed setup, each process would compute gradients on its own mini-batch of data, and AllReduce would sum these gradients to synchronize the model across all workers.

        In non-distributed PyTorch training, everything runs on a single process and device (GPU or CPU). No communication between processes would be necessary, and operations like AllReduce wouldn't be used. In distributed training, however, such operations are critical to ensure all processes maintain consistent model weights.

        ### Key Components:
        - **torch.distributed**: A PyTorch package that facilitates distributed training across multiple devices or machines.
        - **Gloo Backend**: A backend optimized for communication between CPUs and GPUs, often used for small clusters.
        - **AllReduce**: An operation that sums and distributes data (like tensors or gradients) across all workers.
        - **Process Group**: A group of processes that communicate with each other, initialized using `init_process_group`.

        This example demonstrates how data is synchronized across processes using AllReduce. While this simulation runs on a single device in Colab, it helps you understand how distributed training works when you scale to multiple devices or machines.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Install PyTorch

        Colab comes preinstalled with PyTorch, but we’ll ensure the latest version.



        """
    )
    return


@app.cell
def _():
    #update torch
    # (use marimo's built-in package management features instead) !pip install torch --upgrade
    return


@app.cell
def _():
    import torch

    def run():
        tensor = torch.ones(1) * 0
        print(f'Before operation: The tensor has {tensor.item()}')
        tensor = tensor + 1
        print(f'After operation: The tensor has {tensor.item()}')
    run()
    return (torch,)


@app.cell
def _(os, torch):
    import torch.distributed as dist
    from torch.multiprocessing import Process

    def init_process(rank, size, fn, backend='gloo'):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)
        dist.destroy_process_group()

    def run_1(rank, size):
        tensor = torch.ones(1) * rank
        print(f'Before AllReduce: Rank {rank} has {tensor.item()}')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f'After AllReduce: Rank {rank} has {tensor.item()}')

    def spawn_processes():
        size = 2
        processes = []
        for rank in range(size):
            p = Process(target=init_process, args=(rank, size, run_1))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    spawn_processes()
    return Process, dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###Explanation:
        torch.distributed provides a way to perform distributed communication.

        We use the Gloo backend to simulate an AllReduce operation, where two processes sum their tensors and distribute the result to all participants.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise: Modify the PyTorch Distributed Setup

        Now that you’ve learned about distributed training with **PyTorch** and the **AllReduce** operation, let's expand the example by modifying the number of processes (workers).

        ### Task:
        1. Change the number of **workers** (processes) in the simulation from 2 to 3.
        2. Observe how the tensor values before and after **AllReduce** change when using 3 workers instead of 2.

        ### Steps:
        1. Locate the `spawn_processes()` function in the code.
        2. Change the `size` variable from 2 to 3 to simulate 3 workers.
        3. Run the simulation and observe the printed values before and after AllReduce for each worker.

        This exercise will help you understand how AllReduce works across different numbers of processes and how data synchronization happens in distributed environments.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ----

        #End of the Examples

        -----


        ###Other Examples below needs to be adapted to Colab.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Single GPU Training Explanation

        In this example, we are training a simple feedforward neural network on the MNIST dataset using a **single GPU**.

        ### Key Steps:

        1. **Dataset Preparation**:
           - The MNIST dataset is loaded using `torchvision.datasets.MNIST`. This dataset contains images of handwritten digits and is commonly used for training image classification models.
           - We use the `DataLoader` to load the data in batches of 64 images.

        2. **Model Definition**:
           - We define a simple neural network with two hidden layers. The input size is 28x28 (the size of the MNIST images), and the output layer has 10 units corresponding to the 10 digit classes (0-9).

        3. **Device Setup**:
           - We check if a GPU is available using `torch.cuda.is_available()`. If a GPU is available, the model and data will be moved to the GPU for faster computation.

        4. **Loss Function and Optimizer**:
           - The loss function used is `CrossEntropyLoss`, which is suitable for classification tasks.
           - The optimizer is Adam, which adjusts the weights of the model based on the gradients computed during backpropagation.

        5. **Training Loop**:
           - The training process runs for 5 epochs. In each epoch, the model goes through the training data, performs forward and backward passes, and updates its weights using the optimizer.
           - After each epoch, the average loss is printed to track the model's performance.

        ### Important Points:
        - This code runs on a single GPU (or CPU if a GPU isn't available).
        - It's a basic example of how to use PyTorch for training a neural network on a small dataset like MNIST.

        """
    )
    return


@app.cell
def _(torch):
    print(torch.cuda.device_count())  # Should return 1 in Colab, so it will work only the first example, not the other two.
    return


@app.cell
def _(torch):
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    class SimpleNN(nn.Module):

        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    model_3 = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_3.parameters(), lr=0.001)

    def train(model, loader, criterion, optimizer, device):
        model.train()
        for epoch in range(5):
            running_loss = 0.0
            for images, labels in loader:
                images, labels = (images.to(device), labels.to(device))
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss = running_loss + loss.item()
            print(f'Epoch [{epoch + 1}/5], Loss: {running_loss / len(loader)}')
    train(model_3, train_loader, criterion, optimizer, device)
    return SimpleNN, nn, optim, torchvision, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Distributed Training Explanation

        In this example, we modify the training process to run on **multiple GPUs** using PyTorch's **DistributedDataParallel (DDP)**. This allows us to train the model in parallel, which speeds up training when using large datasets or models.

        ### Key Steps:

        1. **Distributed Setup**:
           - The distributed environment is initialized using `torch.distributed.init_process_group()`, which sets up communication between the GPUs.
           - We use the **NCCL backend**, which is optimized for GPU communication.

        2. **Distributed Sampler**:
           - In distributed training, each GPU needs to work on a different part of the dataset to avoid redundant computations.
           - We use `DistributedSampler` to ensure that each GPU gets a unique portion of the training data.

        3. **Model Parallelism**:
           - The model is wrapped in `torch.nn.parallel.DistributedDataParallel`, which ensures that gradients are synchronized across all GPUs after each training step.
           - Each GPU trains its own mini-batch, and then the gradients are averaged across all GPUs.

        4. **Training Loop**:
           - Similar to the single GPU version, we train the model for 5 epochs. Each GPU processes a subset of the data and updates its version of the model. The updates are synchronized across GPUs at the end of each iteration.

        5. **Multiprocessing**:
           - We use `torch.multiprocessing.spawn()` to launch a separate training process for each GPU. Each process handles the training on one GPU.

        ### Important Points:
        - This version of the code is designed to run on multiple GPUs, with each GPU handling a portion of the dataset.
        - Distributed training helps speed up model training, especially for large-scale tasks.
        - The communication between GPUs is handled by PyTorch's Distributed Data Parallel (DDP) mechanism, ensuring efficient and synchronized training across all devices.

        """
    )
    return


@app.cell
def _(DataLoader_1, SimpleNN, dist, nn, optim, torch, torchvision, transforms):
    import torch.multiprocessing as mp
    from torch.utils.data import DataLoader, DistributedSampler

    class SimpleNN_1(nn.Module):

        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def setup(rank, world_size):
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    def train_1(rank, world_size):
        setup(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader_1(dataset=train_dataset, batch_size=64, shuffle=False, sampler=sampler)
        model = SimpleNN_1().to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(5):
            model.train()
            sampler.set_epoch(epoch)
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = (images.to(device), labels.to(device))
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss = running_loss + loss.item()
            if rank == 0:
                print(f'Epoch [{epoch + 1}/5], Loss: {running_loss / len(train_loader)}')
        cleanup()

    def main():
        world_size = torch.cuda.device_count()
        mp.spawn(train_1, args=(world_size,), nprocs=world_size, join=True)
    if __name__ == '__main__':
        main()
    return DistributedSampler, SimpleNN_1, mp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Multi-node Multi-GPU Training in PyTorch

        ### What Happens:

        - **Runs on multiple nodes**, each with one or more GPUs.
        - A master node coordinates the distributed training across all nodes and GPUs.
        - Each GPU on each node processes a portion of the data and computes gradients.
        - Gradients are synchronized across all GPUs (across nodes), ensuring that model updates are consistent.

        ### Key Steps:

        1. **Master Node Setup**:
           - The master node controls and coordinates communication between nodes.
           - Requires setting the `MASTER_ADDR` and `MASTER_PORT` environment variables.

        2. **World Size and Ranks**:
           - The `world_size` is the total number of processes across all nodes (total number of GPUs).
           - Each GPU on each node has a unique rank, which helps to assign tasks and synchronize data.

        3. **Distributed Data Parallelism**:
           - Each node uses `torch.nn.parallel.DistributedDataParallel` to wrap its model, ensuring synchronized training.
           - Data is split across GPUs both within and across nodes.

        4. **Communication Backend**:
           - Uses the **NCCL** backend for GPU communication across multiple nodes and GPUs.
           - Communication across nodes happens over the network.

        ### Key Points:

        - **Scalability**: Allows for training across multiple machines, drastically speeding up large-scale machine learning tasks.
        - **Communication Overhead**: More communication between nodes (compared to single-node), which can introduce latency but allows for training at a larger scale.
        - **Complex Setup**: Requires coordination across nodes (master node, IP addresses, and ports), but scales well for large datasets and models.
        - **Use Cases**: Suitable for very large datasets and models that require multiple nodes and multiple GPUs for efficient training.

        ### Main Differences from Single-node Training:

        - **Single-node**: Only one machine (node) with multiple GPUs. Faster communication due to shared memory.
        - **Multi-node**: Multiple machines (nodes) with GPUs, requiring network communication between nodes.

        """
    )
    return


@app.cell
def _(
    DataLoader_1,
    DistributedSampler,
    SimpleNN_1,
    dist,
    mp,
    nn,
    optim,
    os,
    torch,
    torchvision,
    transforms,
):
    class SimpleNN_2(nn.Module):

        def __init__(self):
            super(SimpleNN_1, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def setup_1(rank, world_size, master_addr, master_port):
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    def cleanup_1():
        dist.destroy_process_group()

    def train_2(rank, world_size, master_addr, master_port):
        setup_1(rank, world_size, master_addr, master_port)
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader_1(dataset=train_dataset, batch_size=64, shuffle=False, sampler=sampler)
        model = SimpleNN_2().to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(5):
            model.train()
            sampler.set_epoch(epoch)
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = (images.to(device), labels.to(device))
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss = running_loss + loss.item()
            if rank == 0:
                print(f'Epoch [{epoch + 1}/5], Loss: {running_loss / len(train_loader)}')
        cleanup_1()

    def main_1():
        world_size = torch.cuda.device_count() * num_nodes
        master_addr = '192.168.1.1'
        master_port = '12355'
        mp.spawn(train_2, args=(world_size, master_addr, master_port), nprocs=torch.cuda.device_count(), join=True)
    if __name__ == '__main__':
        num_nodes = 2
        main_1()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Distributed Linear Regression Training with PyTorch (Not working in one node only)

        In this example, we demonstrate how to perform distributed training in PyTorch using a simple linear regression model. We will use PyTorch’s `torch.distributed` package to synchronize gradients across multiple processes, simulating a multi-worker distributed environment.

        ## Key Concepts

        - **Linear Regression Model**: A basic model that takes one input and predicts one output. The equation we are modeling is `y = 2x + 1`, and we use synthetic data to train the model.
  
        - **Data Creation**: We generate synthetic training data for this model using the equation. Each process will use the same dataset to compute the loss and gradients.

        - **Distributed Setup**: PyTorch’s `init_process_group` is used to initialize the distributed communication backend (in this case, using the **Gloo** backend, which is optimized for CPU communication). This allows each process to communicate with others.

        - **Training Process**:
            - Each worker initializes a copy of the same model.
            - Each worker computes the gradients based on its local loss.
            - Gradients are **summed** across all workers using the `dist.all_reduce` operation, and then averaged by dividing the summed gradients by the number of workers.
            - The optimizer then updates the model parameters with the averaged gradients, ensuring that all workers’ models are synchronized.

        - **Parallel Training**: In this simulation, we have two workers (processes) that perform parallel training. This simulates a scenario where you distribute the training across different nodes or GPUs.

        ## What Happens in the Code

        1. **Model Initialization**: A linear regression model is defined using PyTorch’s `nn.Linear`. This model is simple, with one input and one output.

        2. **Gradient Synchronization**: After computing gradients locally, each process synchronizes the gradients across all workers using `dist.all_reduce`. This ensures that every process has the same gradients and that the model is updated consistently.

        3. **Training**: The training runs for 20 epochs, with each process printing its initial and final weights. You'll notice that the final weights converge to similar values across both processes, showing how distributed training synchronizes learning.

        4. **Output**: You’ll see that the weights of both processes start from different initial values but converge to the same value after training, illustrating the benefit of distributed gradient aggregation.

        ## Why Distributed?

        In non-distributed training, you only use one process or one machine to train your model, which can be slow for large models or datasets. In distributed training, you split the workload across multiple processes or machines, speeding up training while still ensuring all models stay in sync.

        This example uses the Gloo backend (for CPU communication) and simulates two processes working in parallel to train a model. For real-world applications, you would scale this to more processes and potentially use GPUs.

        ---

        Run the code below to see the distributed training in action. You’ll observe how the model’s weights are synchronized across two workers.

        """
    )
    return


@app.cell
def _(Process, dist, nn, optim, os, torch):
    class LinearRegressionModel(nn.Module):

        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    def init_process_1(rank, size, fn, backend='gloo'):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '23554'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)
        dist.destroy_process_group()

    def train_3(rank, size):
        x_train = torch.tensor([[i] for i in range(10)], dtype=torch.float32)
        y_train = torch.tensor([[2 * i + 1] for i in range(10)], dtype=torch.float32)
        model = LinearRegressionModel()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        print(f'Rank {rank} model initial weights: {model.linear.weight.item()}')
        for epoch in range(20):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data = param.grad.data / size
            optimizer.step()
        print(f'Rank {rank} model final weights: {model.linear.weight.item()}')

    def spawn_processes_1():
        size = 2
        processes = []
        for rank in range(size):
            p = Process(target=init_process_1, args=(rank, size, train_3))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    spawn_processes_1()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ## Section 3: Horovod

        In this example, we use **Horovod**, an open-source distributed training library designed for scaling deep learning models across multiple GPUs and machines. Originally developed by Uber, Horovod supports both TensorFlow and PyTorch. It simplifies the process of distributed training by providing easy integration with minimal code changes.

        Horovod uses the **Ring-AllReduce** algorithm, which efficiently synchronizes gradients between workers (GPUs). Each worker computes its local gradients, then Horovod aggregates them by passing gradients through a ring of GPUs. This minimizes communication overhead, making it possible to scale across many GPUs or even multiple machines.

        Horovod allows us to wrap the standard optimizers (like Adam or SGD) in a **Horovod DistributedOptimizer**, which handles gradient synchronization automatically. This ensures that all workers (GPUs) update their models with the same gradient values, maintaining model consistency across the system.

        ### Key Components:
        - **Horovod**: A distributed training library for TensorFlow and PyTorch, which simplifies scaling deep learning models.
        - **Ring-AllReduce**: A communication algorithm that passes gradients around a ring of workers (GPUs) to aggregate them efficiently.
        - **hvd.DistributedOptimizer**: A wrapper around standard optimizers that automatically synchronizes gradients between workers.
        - **BroadcastGlobalVariablesCallback**: Ensures all workers start with the same initial weights by broadcasting the variables from the first worker to all others.

        Unlike non-distributed training where everything happens on a single GPU or machine, Horovod allows training to be distributed across multiple GPUs or machines. This reduces training time by leveraging more computational resources in parallel.

        Although this Colab example runs on a single GPU, it simulates the behavior of multiple GPUs using Horovod. When scaling to multiple GPUs or nodes, this setup remains the same, and Horovod automatically handles the communication between devices.


        ---
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Install Horovod

        Horovod needs to be installed along with TensorFlow for this example.
        """
    )
    return


app._unparsable_cell(
    r"""
    !apt-get update && apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev
    """,
    name="_"
)


@app.cell
def _(os):
    os.environ['HOROVOD_WITH_TENSORFLOW'] = '1'
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install horovod[tensorflow]
    return


@app.cell
def _():
    import horovod.tensorflow.keras as hvd

    # Initialize Horovod
    hvd.init()

    print(f"Horovod is running with {hvd.size()} process(es).")
    return (hvd,)


@app.cell
def _(hvd, tf):
    hvd.init()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    (x_train_3, y_train_3), (x_test_3, y_test_3) = tf.keras.datasets.mnist.load_data()
    x_train_3 = x_train_3[hvd.rank()::hvd.size()]
    y_train_3 = y_train_3[hvd.rank()::hvd.size()]
    x_train_3, x_test_3 = (x_train_3 / 255.0, x_test_3 / 255.0)
    scaled_lr = 0.001 * hvd.size()
    model_4 = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=scaled_lr)
    optimizer_1 = hvd.DistributedOptimizer(optimizer_1)
    model_4.compile(optimizer=optimizer_1, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), hvd.callbacks.MetricAverageCallback()]
    model_4.fit(x_train_3, y_train_3, batch_size=64, callbacks=callbacks, epochs=5, verbose=1 if hvd.rank() == 0 else 0)
    if hvd.rank() == 0:
        model_4.evaluate(x_test_3, y_test_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise: Modify the Horovod Distributed Training

        Now that you've learned about **Horovod** and its distributed training capabilities, let's modify the training script to see how different settings affect the performance.

        ### Task:
        1. Modify the learning rate scaling. Instead of scaling it linearly with the number of GPUs, scale it by a factor of 0.5 per GPU.
        2. Change the number of **epochs** from 5 to 3 and observe how the training performance changes.

        ### Steps:
        1. Locate the line where the learning rate is scaled: `scaled_lr = 0.001 * hvd.size()`.
        2. Modify this line to `scaled_lr = 0.001 * hvd.size() * 0.5`.
        3. Change the `epochs` argument in the `model.fit()` function from 5 to 3.
        4. Rerun the training and observe the training speed and accuracy.

        This exercise will help you understand how learning rate scaling impacts distributed training and how modifying training parameters can affect performance in a distributed setting.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #Explanation:
        Horovod uses the Ring-AllReduce algorithm to synchronize gradients across processes efficiently.

        We use the MNIST dataset and a simple neural network model for training. The learning rate is scaled by the number of workers to ensure stability.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        GLOO
        """
    )
    return


@app.cell
def _(Process, dist, os, torch):
    def init_process_2(rank, size, fn, backend='gloo'):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)
        dist.destroy_process_group()

    def run_2(rank, size):
        tensor = torch.ones(1) * rank
        print(f'Before AllReduce: Rank {rank} has {tensor.item()}')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f'After AllReduce: Rank {rank} has {tensor.item()}')

    def spawn_processes_2():
        size = 2
        processes = []
        for rank in range(size):
            p = Process(target=init_process_2, args=(rank, size, run_2))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    spawn_processes_2()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

