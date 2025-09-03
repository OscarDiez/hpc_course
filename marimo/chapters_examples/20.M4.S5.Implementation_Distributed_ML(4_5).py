import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Practical Implementation: Distributed Deep Learning on Google Colab

        ## Introduction
        In this exercise, we will walk through the steps to set up and train a deep learning model using **TensorFlow's `tf.distribute.Strategy` API** for distributed training. Google Colab provides free access to GPUs (like the **T4**), and we will leverage this resource for our model.

        In a real-world scenario, this setup can be expanded to train on multiple GPUs or nodes using high-performance computing (HPC) clusters. You will also learn how distributed training improves efficiency, reduces training time, and makes deep learning models scalable.

        ## Learning Objectives:
        - Set up Google Colab to use a **T4 GPU**.
        - Define and configure a **distributed training strategy** using TensorFlow.
        - Implement a deep learning model for image classification using **CIFAR-10** dataset.
        - Train the model on multiple devices and observe performance.

        ## Step 1: Set Up the Environment
        First, make sure you are using a GPU for training. You can verify this by running the following code to check if Colab is connected to a GPU.

        ```
        import tensorflow as tf

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        ```

        Once you verify that a GPU is available, move on to defining your distributed training strategy.

        ##Step 2: Define the Distributed Strategy
        We will use tf.distribute.MirroredStrategy, which performs synchronous training across multiple GPUs on a single machine. Colab supports a single GPU (T4), but this strategy can be extended to multiple GPUs.

        ```
        # Define the distribution strategy
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        ```

        The num_replicas_in_sync property shows the number of devices (in this case, the number of GPUs) being used.

        ##Step 3: Load and Preprocess the Data
        We'll use the CIFAR-10 dataset, a popular dataset for image classification. It's important to preprocess the dataset and ensure that it is distributed efficiently across multiple devices.

        ```
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Define batch size and create a distributed dataset
        batch_size = 64
        BUFFER_SIZE = len(x_train)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        ```

        ##Step 4: Define the Model within the Strategy Scope
        When using distributed strategies in TensorFlow, the model must be defined inside the strategy's scope(). This ensures that the variables are mirrored across devices.

        ```
        with strategy.scope():
            # Define a simple CNN model
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)
            ])

            # Compile the model
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
        ```


        ##Step 5: Train the Model with the Distributed Strategy
        Now, we'll train the model using the distributed strategy. The training process will be parallelized across the available GPUs (or just one in the case of Colab).

        ```
        # Train the model
        history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
        ```

        ##Step 6: Evaluate the Model
        Finally, after training, evaluate the model's performance on the test dataset.

        ```
        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_dataset)
        print('Test accuracy:', test_acc)
        ```

        ##Step 7: Extend to a Multi-GPU/TPU Environment (Optional)
        In a more complex setup, like in HPC environments or when using services like Google Cloud, you can extend this strategy to multiple GPUs or TPUs by adjusting the distributed strategy and using tf.distribute.TPUStrategy or tf.distribute.MultiWorkerMirroredStrategy.
        """
    )
    return


@app.cell
def _(tf):
    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
    _strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(_strategy.num_replicas_in_sync))
    (_x_train, _y_train), (_x_test, _y_test) = tf.keras.datasets.cifar10.load_data()
    _x_train, _x_test = (_x_train / 255.0, _x_test / 255.0)
    batch_size = 64
    BUFFER_SIZE = len(_x_train)
    train_dataset = tf.data.Dataset.from_tensor_slices((_x_train, _y_train)).shuffle(BUFFER_SIZE).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((_x_test, _y_test)).batch(batch_size)
    with _strategy.scope():
        _model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
        _model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = _model.fit(train_dataset, epochs=10, validation_data=test_dataset)
    test_loss, test_acc = _model.evaluate(test_dataset)
    print('Test accuracy:', test_acc)
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
        # Distributed Training with TensorFlow's MultiWorkerMirroredStrategy
        In this exercise, you will explore how distributed training works using TensorFlow's MultiWorkerMirroredStrategy. You will learn how to set up a multi-node training environment, modify the model's architecture, and scale the learning rate appropriately. This exercise will demonstrate how to leverage distributed computing to handle more complex models, reducing training time by spreading the workload across multiple workers.

        ## 1. Setting Up the TF_CONFIG Environment Variable
        The TF_CONFIG environment variable is essential when configuring TensorFlow to work in a multi-worker setup. This variable defines the cluster structure, specifying each worker’s role and how they communicate with each other.

        The cluster section includes the worker nodes, which are responsible for training the model.
        The task section defines the role of each machine, including the worker's type and index (i.e., which worker is being set up on this node).
        For instance:

        ```
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                'worker': ["worker1_ip:port", "worker2_ip:port", "worker3_ip:port"]
            },
            'task': {'type': 'worker', 'index': 0}  # 'index': 0 for first worker, 1 for the second, etc.
        })

        ```
        This configuration simulates a cluster with 3 worker nodes. Each machine will have a different index value based on its role.

        ## 2. Initializing MultiWorkerMirroredStrategy
        Once the cluster environment is set up using TF_CONFIG, the next step is to initialize the distributed strategy. The MultiWorkerMirroredStrategy enables synchronous training across multiple workers, ensuring that the model replicas are synchronized, and gradient updates happen simultaneously across all workers.

        ```
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        ```

        ##3. Building the Model with strategy.scope()
        The model architecture and its computations must be placed inside the strategy.scope() block. This ensures that TensorFlow distributes the model and training process across all available nodes and GPUs.

        Here’s an example using the MNIST dataset:

        ```
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        with strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        ```

        ## 4. Training and Evaluating the Model
        Once the model is defined, you can train it across multiple workers. The MultiWorkerMirroredStrategy distributes the dataset and ensures that each worker receives a different batch of data for training.

        ```
        history = model.fit(x_train, y_train, epochs=5, batch_size=64)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f'Test accuracy: {test_acc}')
        ```

        ## 5. Scaling Learning Rate
        When using distributed training with multiple workers, it's a good practice to scale the learning rate by multiplying the base rate by the number of workers. This ensures that the gradient updates are balanced and the training process remains efficient across all nodes.

        For example, if you have 3 workers:

        ```
        scaled_learning_rate = base_learning_rate * num_workers
        ```

        ## 6. Experimenting with Different Model Architectures
        You can modify the architecture of the model and observe how the distributed strategy handles these changes. For instance, try adding more layers or changing the activation functions:

        ```
        with strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scaled_learning_rate),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        ```

        ## 7. Expected Output
        After running this exercise on a distributed environment, you should observe the following:

        Training time should decrease as more workers are added to the cluster, depending on the complexity of the model and dataset.
        The model's accuracy should remain consistent, provided the batch sizes and learning rate are adjusted correctly.
        The training process should leverage the available computational resources (e.g., GPUs) efficiently.

        ##Exercise Tasks:
        Modify the TF_CONFIG: Add at least 3 workers to the cluster configuration.
        Experiment with Model Architectures: Change the architecture (e.g., add more layers or modify activation functions).
        Adjust the Learning Rate: Scale the learning rate by the number of workers and observe the impact on the model’s convergence speed and accuracy.


        Key Takeaways:

        - Distributed Training allows you to reduce training time by distributing the workload across multiple workers.

        - MultiWorkerMirroredStrategy synchronizes the training process across all workers, ensuring consistency in gradient updates.

        - Scaling Learning Rate is essential for maintaining effective learning when distributing training across multiple workers.

        - Experimenting with Architectures can help understand how more complex models behave in distributed environments and how strategies can efficiently handle them.
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
    return


@app.cell
def _(tf):
    _strategy = tf.distribute.MultiWorkerMirroredStrategy()
    (_x_train, _y_train), (_x_test, _y_test) = tf.keras.datasets.mnist.load_data()
    _x_train, _x_test = (_x_train / 255.0, _x_test / 255.0)
    with _strategy.scope():
        _model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
        _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _model.fit(_x_train, _y_train, epochs=5)
    _model.evaluate(_x_test, _y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Comparing Single GPU, Multiple GPUs, and Multiple Nodes in TensorFlow

        This comparison highlights the key differences between training a model using **single GPU**, **multiple GPUs**, and **multiple nodes** with TensorFlow's `tf.distribute.Strategy`.

        ## 1. Single GPU Setup
        - **Resource Usage**: Runs on a single GPU.
        - **Strategy**: No special strategy needed.
        - **Scope**: No `strategy.scope()` required.
        - **Synchronization**: Not applicable, only one device is used.

        ## 2. Multiple GPUs Setup (MirroredStrategy)
        - **Resource Usage**: Uses multiple GPUs on the same machine.
        - **Strategy**: `MirroredStrategy` synchronizes gradient updates across all GPUs.
        - **Scope**: `strategy.scope()` is required to ensure computations are distributed across GPUs.
        - **Synchronization**: Gradients are synchronized across GPUs using an all-reduce algorithm.

        ## 3. Multiple Nodes Setup (MultiWorkerMirroredStrategy)
        - **Resource Usage**: Uses multiple machines, each with one or more GPUs.
        - **Strategy**: `MultiWorkerMirroredStrategy` distributes the model and synchronizes gradient updates across multiple nodes.
        - **Scope**: `strategy.scope()` is required for distributing computation across nodes and GPUs.
        - **Synchronization**: Gradients are synchronized across nodes using collective communication, requiring configuration of the `TF_CONFIG` environment variable.

        ## Summary of Differences:

        | **Feature**             | **Single GPU**                             | **Multiple GPUs (MirroredStrategy)**                  | **Multiple Nodes (MultiWorkerMirroredStrategy)**      |
        |-------------------------|--------------------------------------------|------------------------------------------------------|------------------------------------------------------|
        | **Execution**            | Single GPU                                 | Multiple GPUs on the same machine                    | Multiple nodes (multiple machines with GPUs)          |
        | **Strategy**             | None                                       | `MirroredStrategy`                                   | `MultiWorkerMirroredStrategy`                        |
        | **Synchronization**      | Not needed (single device)                 | Sync gradients across GPUs using all-reduce           | Sync gradients across workers using collective comm   |
        | **Scope**                | No `strategy.scope()`                      | Requires `strategy.scope()` for GPU distribution      | Requires `strategy.scope()` for multi-node distribution|
        | **Speed/Performance**    | Limited by single GPU                      | Faster, workload distributed across GPUs              | Faster, workload distributed across nodes and GPUs    |

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

