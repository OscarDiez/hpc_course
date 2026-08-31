import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #Assignment Module 4.  Distributed Training with TensorFlow - Multi-GPU and Multi-Node Simulation

        ###Problem Statement:
        You are tasked with implementing and optimizing a neural network for image classification using TensorFlow's distributed strategies. First, you will use the MirroredStrategy for distributed training across multiple GPUs (Colab simulates multi-GPU setups). Then, you'll extend the setup to a multi-node distributed system using MultiWorkerMirroredStrategy to simulate multi-node training.

        You will implement and optimize the training process and compare the performance between the multi-GPU and multi-node setups.

        ###Part 1: Multi-GPU Training using MirroredStrategy
        1. Define a Distributed Strategy: Use tf.distribute.MirroredStrategy() to simulate multi-GPU training.

        2. Dataset: Use the MNIST dataset, ensuring it is preprocessed and normalized.

        3. Model: Build a simple CNN using TensorFlow’s Sequential API.

        4. Training: Train the model using the distributed strategy and compare the performance with non-distributed training.

        5. Evaluation: Evaluate the model on the test set and ensure that the training converges correctly with multiple GPUs.



        ###Part 2: Multi-Node Training using MultiWorkerMirroredStrategy
        1. Simulate a Multi-Node Setup: Set up MultiWorkerMirroredStrategy with appropriate environment variables (TF_CONFIG) for node communication.

        2. Training: Train the same model across simulated nodes and compare the performance.

        3. Evaluation: Evaluate the model after training in the multi-node setup.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###Starter Code for Multi-GPU Training (Part 1):
        """
    )
    return


@app.cell
def _():
    import tensorflow as tf
    strategy = tf.distribute.MirroredStrategy()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = (x_train / 255.0, x_test / 255.0)
    with strategy.scope():
        _model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
        _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _model.fit(x_train, y_train, epochs=5, batch_size=64)
    _model.evaluate(x_test, y_test)
    return strategy, tf, x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Part 1 Instructions:
        1. Modify the model: Change the architecture from a simple feedforward network to a Convolutional Neural Network (CNN) to improve accuracy.
        2.  Experiment with batch size: Try different batch sizes (64, 128, 256) and observe the impact on performance.
        3. Measure training time: Compare the performance of running the training on a single GPU vs. using MirroredStrategy.

        ###Example CNN Architecture:
        """
    )
    return


@app.cell
def _(strategy, tf, x_test, x_train, y_test, y_train):
    with strategy.scope():
        _model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
        _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _model.fit(x_train, y_train, epochs=5, batch_size=64)
    _model.evaluate(x_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Part 2: Multi-Node Training with MultiWorkerMirroredStrategy (Colab Simulation)
        Define the TF_CONFIG Environment Variable: To simulate multi-node training, you need to set the TF_CONFIG environment variable that specifies the cluster configuration (which nodes are workers) and the role of each worker.

        ###Training Code for Multi-Node Setup:
        """
    )
    return


@app.cell
def _(tf):
    import os
    import json
    os.environ['TF_CONFIG'] = json.dumps({'cluster': {'worker': ['localhost:12345', 'localhost:12346']}, 'task': {'type': 'worker', 'index': 0}})
    strategy_1 = tf.distribute.MultiWorkerMirroredStrategy()
    (x_train_1, y_train_1), (x_test_1, y_test_1) = tf.keras.datasets.mnist.load_data()
    x_train_1, x_test_1 = (x_train_1 / 255.0, x_test_1 / 255.0)
    with strategy_1.scope():
        _model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
        _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _model.fit(x_train_1, y_train_1, epochs=5, batch_size=64)
    _model.evaluate(x_test_1, y_test_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Part 2 Instructions:
        1. Run the code on multiple workers: Simulate two workers on different ports by running the code on two different Colab instances or on a local machine with multi-node configuration.

        2. Set up the TF_CONFIG correctly: Ensure each worker is assigned the correct task (task: index in TF_CONFIG) and port.
        3. Experiment with different architectures: Try training larger models and observe how the multi-node setup scales the training.
        4. Checkpointing and saving: Implement a checkpointing system to save the model weights during training.

        ###Deliverables:
        - Modified Multi-GPU Code (Part 1): Submit the modified CNN architecture and experiments using MirroredStrategy.
        - Multi-Node Simulation (Part 2): Submit the code for multi-node distributed training and provide results for different configurations.
        - Report: A short report (1-2 pages) summarizing:
         - Differences in training time between single GPU, multi-GPU, and multi-node setups.
         - Model accuracy and convergence speed in each setup.
         - Challenges faced in implementing the distributed training and how they were overcome.

        ###Grading Rubric:
        - Correct Implementation of Multi-GPU Training (30 points): Proper use of MirroredStrategy and parallel training setup.

        - Multi-Node Setup and Execution (30 points): Correct use of MultiWorkerMirroredStrategy and simulation of multi-node behavior.
        - Performance Analysis (20 points): Evaluation of model performance in different setups.
        - Code Structure and Comments (10 points): Clean, well-documented code with appropriate comments.
        - Report and Findings (10 points): Summary of experiments and insights into distributed training.

        ###Hints:
        - MirroredStrategy works well for multi-GPU setups. Use it to simulate parallelism with one GPU on Colab.
        TF_CONFIG is crucial for multi-node simulations. Understand how workers communicate in a distributed cluster.
        - Use TensorFlow's Checkpoint API to save model weights periodically during training.
        - Test with smaller models first, and then try scaling up with larger batch sizes and deeper networks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ------
        -----

        #The assignments finish here.  
        The third part below is not part of the assigment.

        -----
        -----

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Part 3: Mixed Precision Training with Gradient Accumulation and cuDNN Optimizations (optional exercise to get 20% points)
        ###Optimizing Distributed Training with Mixed Precision and Gradient Accumulation

        In this exercise, you will extend the previous work by incorporating mixed precision training and gradient accumulation to handle large models efficiently on GPUs. You will also explore how to optimize model training by leveraging cuDNN backend for convolution operations.

        The goal is to implement a high-performance training setup that optimally utilizes GPU resources, even in the case of large batch sizes that do not fit into GPU memory. By the end of this exercise, you will learn how to:

        Use mixed precision training to speed up training while reducing memory usage.
        Implement gradient accumulation to simulate training with larger batch sizes.
        Optimize CNN performance using cuDNN accelerations.

        ###Instructions:

        1. Mixed Precision Training:
         - Use TensorFlow’s mixed precision API to convert the model’s computations to use mixed precision (FP16 and FP32) for faster training.
         - Ensure that the model’s optimizer is configured to handle mixed precision.

        2. Gradient Accumulation:

          - Implement gradient accumulation to simulate large batch training, splitting a large batch into smaller sub-batches and accumulating gradients before performing an optimizer step.

        3. cuDNN Optimization:
         - Enable cuDNN optimizations to take full advantage of CUDA for convolutional operations, improving performance on supported GPUs.

        4. Multi-GPU Setup:
         - Use MirroredStrategy to distribute the model across multiple GPUs (if available) and synchronize the gradients across them.


        ###Starter Code:
        """
    )
    return


@app.cell
def _(tf):
    from tensorflow.keras import mixed_precision
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'GPUs Available: {len(gpus)}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print('No GPUs available, training on CPU.')
    mixed_precision.set_global_policy('mixed_float16')
    strategy_2 = tf.distribute.MirroredStrategy()
    (x_train_2, y_train_2), (x_test_2, y_test_2) = tf.keras.datasets.mnist.load_data()
    x_train_2, x_test_2 = (x_train_2 / 255.0, x_test_2 / 255.0)
    x_train_2 = x_train_2.reshape(-1, 28, 28, 1)
    x_test_2 = x_test_2.reshape(-1, 28, 28, 1)
    with strategy_2.scope():
        _model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax', dtype='float32')])
        optimizer = tf.keras.optimizers.Adam()
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        _model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ACCUMULATION_STEPS = 4

    @tf.function
    def train_step(batch_inputs, batch_labels, accumulated_gradients, step):
        with tf.GradientTape() as tape:
            predictions = _model(batch_inputs, training=True)
            loss = _model.compiled_loss(batch_labels, predictions)
        gradients = tape.gradient(loss, _model.trainable_variables)
        for i in range(len(accumulated_gradients)):
            accumulated_gradients[i] = accumulated_gradients[i] + gradients[i]
        if (step + 1) % ACCUMULATION_STEPS == 0:
            optimizer.apply_gradients(zip(accumulated_gradients, _model.trainable_variables))
            for i in range(len(accumulated_gradients)):
                accumulated_gradients[i].assign(tf.zeros_like(accumulated_gradients[i]))
    with strategy_2.scope():
        accumulated_gradients = [tf.Variable(tf.zeros_like(var), trainable=False) for var in _model.trainable_variables]
        steps_per_epoch = len(x_train_2) // (64 * ACCUMULATION_STEPS)
        for epoch in range(5):
            print(f'Epoch {epoch + 1}/{5}')
            step = 0
            for batch_start in range(0, len(x_train_2), 64):
                batch_inputs = x_train_2[batch_start:batch_start + 64]
                batch_labels = y_train_2[batch_start:batch_start + 64]
                train_step(batch_inputs, batch_labels, accumulated_gradients, step)
                step = step + 1
                if step >= steps_per_epoch:
                    break
    test_loss, test_acc = _model.evaluate(x_test_2, y_test_2)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Key Elements:
        1. Mixed Precision Training:

         - We use TensorFlow’s mixed_precision API to convert part of the model to FP16, while ensuring that layers requiring higher precision (e.g., the final dense layer) remain in FP32.
         - The LossScaleOptimizer is used to avoid numerical instability during training with mixed precision.

        2. Gradient Accumulation:

         - Instead of performing an optimizer step after every batch, we accumulate gradients over several batches (ACCUMULATION_STEPS) and apply the gradients only once enough batches have been processed. This simulates training with larger batch sizes without exceeding GPU memory.

        3. cuDNN Optimizations:

          - cuDNN optimizations are enabled for faster convolution operations on supported NVIDIA GPUs.
          - This optimization allows for high-performance execution of convolutional layers, especially on NVIDIA GPUs.

        4. Multi-GPU Setup:
        - The MirroredStrategy is used to distribute training across multiple GPUs (if available), ensuring that the model is replicated on each device and gradients are synchronized across GPUs.

        ## Deliverables:
        1. Complete Code: Submit the complete implementation of the optimized CNN model using mixed precision, gradient accumulation, and cuDNN optimizations.

        2. Report: A short report (1-2 pages) that describes:

          - How mixed precision training and gradient accumulation were implemented.
          - The performance gains observed in terms of training speed and memory usage.
          - How cuDNN optimizations impacted the overall training performance.
        3. Test Results: Include the final test accuracy after 5 epochs of training.

        ###Grading Rubric:
        - Correct Use of Mixed Precision (30 points): Proper implementation of TensorFlow’s mixed_precision API with correct handling of data types in layers.
        - Correct Implementation of Gradient Accumulation (30 points): Proper accumulation of gradients across mini-batches and application of the accumulated gradients at the correct intervals.
        - cuDNN Optimizations (20 points): The code should take advantage of cuDNN optimizations for convolutional layers.
        - Code Quality and Comments (10 points): Clear and well-documented code with appropriate comments explaining the steps.
        - Performance Testing and Analysis (10 points): Report should include observations on training speed and memory usage with mixed precision and gradient accumulation.

        ###Hints:
        - Use mixed_precision.set_global_policy() to enable mixed precision training.
        - Make sure to reset accumulated gradients after applying them to the model.
        - Ensure that you are using cuDNN for optimizing convolution operations on supported GPUs by setting tf.config.experimental.set_memory_growth(gpu, True).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

