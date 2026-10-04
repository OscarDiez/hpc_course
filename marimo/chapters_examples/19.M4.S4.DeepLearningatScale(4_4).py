import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Vanishing Gradient Problem and Solutions

        The Vanishing Gradient Problem is a common issue that arises during training of deep neural networks, where gradients become progressively smaller as they are propagated back through layers. This slows down learning, especially in deep networks like CNNs, RNNs, and Transformers. The gradients shrink so much across layers that the weights stop updating, leading to stagnation in learning.

        ### Why does this happen?
        In deep networks, during backpropagation, gradients are multiplied by small values at each layer. This results in the gradients approaching zero for layers near the input, effectively halting learning. This issue is particularly prevalent in networks using activation functions like the sigmoid or tanh.

        ### Common solutions to the Vanishing Gradient Problem:
        1. **ReLU Activation Function**: ReLU (Rectified Linear Unit) and its variants (Leaky ReLU, Parametric ReLU) allow gradients to propagate more effectively by not squashing the gradient. They introduce non-linearity but avoid the problem of diminishing gradients.
        2. **Batch Normalization**: This technique normalizes inputs to each layer, reducing internal covariate shift, stabilizing gradients, and enabling higher learning rates.
        3. **Residual Networks (ResNets)**: Residual Networks use skip connections, which allow gradients to bypass some layers, preventing them from vanishing completely.

        In this notebook, we will explore these three solutions with practical examples using PyTorch.

        """
    )
    return


@app.cell
def _():
    # First, we need to install PyTorch in case it isn't already installed in the environment
    # (use marimo's built-in package management features instead) !pip install torch
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    def _track_gradients(model):
        grad_norms = []
        for param in _model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        return grad_norms

    class _BaselineModel(nn.Module):

        def __init__(self):
            super(_BaselineModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            return x

    class _ReLUModel(nn.Module):

        def __init__(self):
            super(_ReLUModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    class _BatchNormModel(nn.Module):

        def __init__(self):
            super(_BatchNormModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = torch.relu(self.bn2(self.fc2(x)))
            x = torch.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            return x

    class _ResidualBlock(nn.Module):

        def __init__(self, in_features, out_features):
            super(_ResidualBlock, self).__init__()
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x):
            residual = x
            out = torch.relu(self.fc(x))
            return out + residual

    class _ResNetModel(nn.Module):

        def __init__(self):
            super(_ResNetModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.residual1 = _ResidualBlock(512, 512)
            self.fc2 = nn.Linear(512, 256)
            self.residual2 = _ResidualBlock(256, 256)
            self.fc3 = nn.Linear(256, 128)
            self.residual3 = _ResidualBlock(128, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.residual1(x)
            x = torch.relu(self.fc2(x))
            x = self.residual2(x)
            x = torch.relu(self.fc3(x))
            x = self.residual3(x)
            x = self.fc4(x)
            return x
    _input_data = torch.randn(64, 784)
    _labels = torch.randint(0, 10, (64,))

    def _train_model(model, optimizer, criterion):
        _model.train()
        _optimizer.zero_grad()
        output = _model(_input_data)
        loss = _criterion(output, _labels)
        loss.backward()
        grad_norms = _track_gradients(_model)
        _optimizer.step()
        return grad_norms
    _criterion = nn.CrossEntropyLoss()
    _baseline_model = _BaselineModel()
    _relu_model = _ReLUModel()
    _batchnorm_model = _BatchNormModel()
    _resnet_model = _ResNetModel()
    _optimizer_baseline = optim.Adam(_baseline_model.parameters(), lr=0.001)
    _optimizer_relu = optim.Adam(_relu_model.parameters(), lr=0.001)
    _optimizer_batchnorm = optim.Adam(_batchnorm_model.parameters(), lr=0.001)
    _optimizer_resnet = optim.Adam(_resnet_model.parameters(), lr=0.001)
    _grad_baseline = _train_model(_baseline_model, _optimizer_baseline, _criterion)
    _grad_relu = _train_model(_relu_model, _optimizer_relu, _criterion)
    _grad_batchnorm = _train_model(_batchnorm_model, _optimizer_batchnorm, _criterion)
    _grad_resnet = _train_model(_resnet_model, _optimizer_resnet, _criterion)
    plt.plot(_grad_baseline, label='Baseline (No ReLU/BN/ResNet)')
    plt.plot(_grad_relu, label='ReLU Activation')
    plt.plot(_grad_batchnorm, label='Batch Normalization')
    plt.plot(_grad_resnet, label='Residual Network')
    plt.title('Gradient Norms across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.show()
    return nn, optim, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ####Why Some Models Have More Than 7 Gradient Norms:

        The number of gradient norms corresponds to the number of parameters (weights and biases) in the model that receive gradients during backpropagation. In models like the Batch Normalization model, additional parameters are introduced by the BatchNorm layers (specifically, the learnable scale and shift parameters, gamma and beta). This increases the total number of parameters that have gradients, resulting in more than 7 gradient norms being recorded. Conversely, in models without these additional layers or where some parameters (like biases in certain layers) don't receive gradients due to activation functions like ReLU causing zero outputs, you might see only up to 7 gradient norms.

        ####Why the Batch Normalization Model Has Big Peaks:

        Batch Normalization adjusts the input distribution for each layer, which can amplify gradients during training. This amplification leads to larger gradient updates for certain parameters, appearing as big peaks in your gradient norm graph. These peaks indicate that the BatchNorm layers are actively scaling the gradients, which can help in faster convergence but may also cause instability if not properly managed. The sharp changes are a result of the normalization process interacting with the network's weights and the specific data flowing through the layers.

        ####Best Model in This Context:

        In this small network example, the Residual Network (ResNetModel) appears to be the best choice. Despite having a simple structure, the residual connections help maintain a smooth and stable gradient flow, preventing issues like vanishing gradients. This is particularly beneficial even when the network isn't very deep, as it ensures that all layers learn effectively. The ResNet model balances complexity and performance, making it suitable for networks with fewer layers.

        ####From How Many Layers Does the Vanishing Gradient Problem Occur:

        The vanishing gradient problem typically starts to become significant in networks with more than 5-10 layers, especially when using activation functions like sigmoid or tanh. However, factors like poor weight initialization, lack of normalization, and the choice of activation functions can exacerbate the problem even in shallower networks. Using ReLU activations, Batch Normalization, or residual connections helps mitigate this issue, allowing for deeper networks to be trained effectively.
        """
    )
    return


@app.cell
def _(nn, optim, plt, torch):
    def _track_gradients(model):
        grad_norms = []
        print('\nTracking Gradients:')
        for name, param in _model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms.append(norm)
                print(f'Parameter: {name}, Gradient Norm: {norm}')
            else:
                print(f'Parameter: {name} has no gradient.')
        return grad_norms

    class _BaselineModel(nn.Module):

        def __init__(self):
            super(_BaselineModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            return x

    class _ReLUModel(nn.Module):

        def __init__(self):
            super(_ReLUModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    class _BatchNormModel(nn.Module):

        def __init__(self):
            super(_BatchNormModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = torch.relu(self.bn2(self.fc2(x)))
            x = torch.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            return x

    class _ResidualBlock(nn.Module):

        def __init__(self, features):
            super(_ResidualBlock, self).__init__()
            self.fc = nn.Linear(features, features)

        def forward(self, x):
            residual = x
            out = torch.relu(self.fc(x))
            return out + residual

    class _ResNetModel(nn.Module):

        def __init__(self):
            super(_ResNetModel, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.residual1 = _ResidualBlock(512)
            self.fc2 = nn.Linear(512, 256)
            self.residual2 = _ResidualBlock(256)
            self.fc3 = nn.Linear(256, 128)
            self.residual3 = _ResidualBlock(128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.residual1(x)
            x = torch.relu(self.fc2(x))
            x = self.residual2(x)
            x = torch.relu(self.fc3(x))
            x = self.residual3(x)
            x = self.fc4(x)
            return x
    _input_data = torch.randn(64, 784)
    _labels = torch.randint(0, 10, (64,))

    def _train_model(model, optimizer, criterion):
        _model.train()
        _optimizer.zero_grad()
        output = _model(_input_data)
        loss = _criterion(output, _labels)
        print(f'\nLoss: {loss.item()}')
        loss.backward()
        grad_norms = _track_gradients(_model)
        _optimizer.step()
        return grad_norms
    _criterion = nn.CrossEntropyLoss()
    _baseline_model = _BaselineModel()
    _relu_model = _ReLUModel()
    _batchnorm_model = _BatchNormModel()
    _resnet_model = _ResNetModel()
    _optimizer_baseline = optim.Adam(_baseline_model.parameters(), lr=0.001)
    _optimizer_relu = optim.Adam(_relu_model.parameters(), lr=0.001)
    _optimizer_batchnorm = optim.Adam(_batchnorm_model.parameters(), lr=0.001)
    _optimizer_resnet = optim.Adam(_resnet_model.parameters(), lr=0.001)
    print('Training Baseline Model')
    _grad_baseline = _train_model(_baseline_model, _optimizer_baseline, _criterion)
    print('\nTraining ReLU Model')
    _grad_relu = _train_model(_relu_model, _optimizer_relu, _criterion)
    print('\nTraining BatchNorm Model')
    _grad_batchnorm = _train_model(_batchnorm_model, _optimizer_batchnorm, _criterion)
    print('\nTraining ResNet Model')
    _grad_resnet = _train_model(_resnet_model, _optimizer_resnet, _criterion)
    plt.figure(figsize=(10, 6))
    plt.plot(_grad_baseline, label='Baseline (No ReLU/BN/ResNet)', marker='o')
    plt.plot(_grad_relu, label='ReLU Activation', marker='o')
    plt.plot(_grad_batchnorm, label='Batch Normalization', marker='o')
    plt.plot(_grad_resnet, label='Residual Network', marker='o')
    plt.title('Gradient Norms across Parameters')
    plt.xlabel('Parameter Index')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Code

        ### 1. ReLU and Leaky ReLU
        In this section, we define two types of activation functions:
        - **ReLU**: Returns the maximum of 0 and the input value. This prevents negative values from propagating, which helps to reduce vanishing gradients.
        - **Leaky ReLU**: Similar to ReLU, but allows a small, non-zero gradient when the input is negative. This is beneficial for cases where neurons might be deactivated for all inputs in ReLU.
        We also generate and plot a graph to visualize the behavior of both functions.

        In this example, the **ReLU activation** is used to help mitigate the vanishing gradient problem by allowing gradients to propagate more effectively than activations like sigmoid or tanh.

        ### 2. Batch Normalization
        We define a simple neural network class in PyTorch that includes fully connected (linear) layers followed by batch normalization layers. **Batch normalization** stabilizes the learning process by normalizing the input to each layer and reducing internal covariate shifts. By doing so, it allows gradients to propagate more effectively across layers.

        In the plot, we can see that the batch normalization layers (green line) show more pronounced gradients, especially in the earlier layers, indicating better gradient propagation compared to the baseline and ReLU-only models.

        ### 3. Residual Networks (ResNets)
        We create a simple **residual block** in PyTorch. A residual block "skips" one or more layers by adding the input (residual connection) directly to the output of the layer. This technique helps in addressing the vanishing gradient problem by ensuring that gradients can flow more easily across layers without being diminished.

        In the graph, the **Residual Networks** model (red line) shows very stable gradients across all layers, demonstrating how residual connections help preserve gradient flow through the network.

        ### Gradient Norms Plot
        We plot the gradient norms across the layers for each model:
        - The **baseline model** without any special techniques shows quickly vanishing gradients, especially after the first layer.
        - The **ReLU model** improves gradient flow but still suffers from smaller gradients as we go deeper into the network.
        - The **batch normalization model** exhibits better gradient propagation with sharp peaks, though it can show instability.
        - The **residual networks** model has stable and consistent gradient norms, showing that residual connections are highly effective in solving the vanishing gradient problem.

        This demonstrates how each of these methods—ReLU, Batch Normalization, and Residual Networks—can help alleviate the vanishing gradient problem, improving the learning capability of deep neural networks.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ----

        ### Short Exercise: Experiment with Activation Functions to Observe the Vanishing Gradient Problem

        In the previous sections, we explored the impact of using ReLU, Batch Normalization, and Residual Networks to mitigate the vanishing gradient problem. For this exercise, we want you to modify the current code and observe how a different activation function affects the gradient flow in the network.

        **Task**: Change the ReLU activation function to the Sigmoid activation function in the `SimpleModel` class and re-run the model training. Sigmoid activations are known to exacerbate the vanishing gradient problem because of how they "squash" inputs into a very small range.

        Here’s how you can make the change:
        - Replace `torch.relu(self.fc1(x))` with `torch.sigmoid(self.fc1(x))` in the `forward()` method of the `SimpleModel` class.
        - Observe and record how the loss and gradients behave over multiple epochs.

        **Questions**:
        1. How does the training loss change compared to the original ReLU implementation?
        2. Do the gradients vanish more quickly with the Sigmoid activation function? If so, why?

        Make sure to document your findings and explain the difference in behavior when using Sigmoid vs. ReLU.



        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ----------------
        ### Short Exercise: Modify Batch Normalization and Observe its Impact on Training Stability

        In the previous section, we introduced Batch Normalization as a solution to the vanishing gradient problem. For this exercise, you will modify the current code to **remove** Batch Normalization from the model and observe how its absence impacts training stability.

        **Task**:
        - In the `SimpleModel` class, remove the `BatchNorm1d` layer. Specifically, delete or comment out the `self.bn1 = nn.BatchNorm1d(512)` line and remove the call to `self.bn1(x)` in the `forward()` method.
        - Train the model without Batch Normalization and observe how the training loss changes over the epochs.

        **Questions**:
        1. How does the loss behave compared to when Batch Normalization was included? Is the training less stable or slower?
        2. What impact does removing Batch Normalization have on the gradient norms?

        Document your observations and explain the importance of Batch Normalization in improving training stability.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---


        # Memory Limitations in Large Models

        As models grow in size, such as GPT-3 with 175 billion parameters, memory demands increase significantly. These models can require hundreds of gigabytes of memory, far exceeding the capacity of a single GPU or TPU. Memory limitations are a crucial challenge in training large-scale models effectively.

        ### Memory Demands
        - Large models, such as GPT-3, require immense memory for storing weights, activations, and gradients.
        - Even modern hardware like GPUs and TPUs struggle with fitting these models into memory.

        ### Solutions for Memory Limitations:
        1. **Gradient Accumulation**: Instead of updating the model's weights after every mini-batch, gradients are accumulated over several mini-batches, and the model is updated less frequently. This reduces memory usage without sacrificing learning.
   
        2. **Mixed Precision Training**: Mixed precision refers to using both 16-bit (FP16) and 32-bit (FP32) floating-point numbers. By using 16-bit precision where possible, memory usage can be reduced by almost half while maintaining training performance.

        In this notebook, we will demonstrate both Gradient Accumulation and Mixed Precision Training techniques in PyTorch.

        """
    )
    return


@app.cell
def _():
    # Install PyTorch if it's not installed
    # (use marimo's built-in package management features instead) !pip install torch torchvision
    return


@app.cell
def _(nn, optim, torch):
    import time
    from torch.amp import autocast, GradScaler
    import torch.cuda as cuda

    class _SimpleModel(nn.Module):

        def __init__(self):
            super(_SimpleModel, self).__init__()
            self.fc1 = nn.Linear(4096, 2048)
            self.fc2 = nn.Linear(2048, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc4 = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    _model = _SimpleModel().cuda()
    _optimizer = optim.Adam(_model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    _input_data = torch.randn(128, 4096).cuda()
    _labels = torch.randint(0, 10, (128,)).cuda()

    def get_gpu_memory():
        return cuda.memory_allocated() / 1024 ** 2
    print('\n--- Mixed Precision Training Example ---')
    scaler = GradScaler()
    initial_memory_mp = get_gpu_memory()
    print(f'Initial GPU Memory (Mixed Precision): {initial_memory_mp:.2f} MB')
    start_time_mp = time.time()
    for i in range(4):
        _optimizer.zero_grad()
        with autocast(device_type='cuda'):
            output = _model(_input_data)
            loss = loss_function(output, _labels)
        scaler.scale(loss).backward()
        scaler.step(_optimizer)
        scaler.update()
    final_memory_mp = get_gpu_memory()
    end_time_mp = time.time()
    print(f'Final GPU Memory (Mixed Precision): {final_memory_mp:.2f} MB')
    print(f'Time Taken for Mixed Precision Training: {end_time_mp - start_time_mp:.4f} seconds')
    print('\n--- Gradient Accumulation Example (Standard Precision) ---')
    accumulation_steps = 4
    _optimizer.zero_grad()
    initial_memory = get_gpu_memory()
    print(f'Initial GPU Memory (Standard Precision): {initial_memory:.2f} MB')
    start_time = time.time()
    for i in range(accumulation_steps):
        output = _model(_input_data)
        loss = loss_function(output, _labels)
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            _optimizer.step()
            _optimizer.zero_grad()
    final_memory = get_gpu_memory()
    end_time = time.time()
    print(f'Final GPU Memory (Standard Precision): {final_memory:.2f} MB')
    print(f'Time Taken for Gradient Accumulation: {end_time - start_time:.4f} seconds')
    print(f'\nMemory Reduction using Mixed Precision: {initial_memory - initial_memory_mp:.2f} MB')
    return (time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Code

        ### 1. Gradient Accumulation
        In this part, we demonstrate how to perform gradient accumulation in PyTorch. Instead of updating the model weights after every mini-batch, we accumulate gradients over multiple mini-batches.
        - We divide the loss by the number of accumulation steps (`accumulation_steps`) to average the gradients properly.
        - The model is only updated after the specified number of mini-batches.

        This technique allows us to train with larger effective batch sizes without requiring the GPU memory to hold a large batch in one go.

        ### 2. Mixed Precision Training
        In this section, we apply mixed precision training, where operations are performed using both 16-bit and 32-bit floating-point numbers.
        - The `autocast()` function automatically selects the optimal precision for each operation.
        - The `GradScaler` is used to ensure numerical stability by scaling gradients before backpropagation.

        Mixed precision helps reduce memory usage and speeds up computation without significant loss of accuracy, making it a valuable tool for training large models.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ----

        # Efficient Utilization of CPUs and GPUs

        ## 1. Efficient Utilization of CPUs

        ### Role of CPUs:
        - **Data Preprocessing**: CPUs handle tasks like data augmentation, normalization, and distributing the data to GPUs for training. This allows the GPU to focus on heavy computation, reducing overall training time.
        - **I/O Operations**: CPUs manage input/output operations, ensuring that the GPU is continuously fed with data, preventing GPU idling.
        - **Inference at Scale**: CPUs handle batching inputs and managing pipelines for large-scale inference tasks.

        ### Best Practices:
        - **Parallel Preprocessing**: Use multiple CPU threads (e.g., in PyTorch or TensorFlow) to load and preprocess data in parallel, maximizing data throughput.
        - **Interleaving Tasks**: Use the CPU to handle scheduling and preprocessing tasks in the background while the GPU performs computations. This ensures the GPU is not waiting for data, improving efficiency.

        ## 2. Efficient Utilization of GPUs

        ### GPU Workload:
        - **Optimized for Floating-Point Operations**: GPUs excel at matrix multiplications, backpropagation, and other floating-point operations, which are the core operations in deep learning.
        - **Use of CUDA**: CUDA (Compute Unified Device Architecture) allows for parallel computation on NVIDIA GPUs. Deep learning frameworks like PyTorch and TensorFlow heavily rely on CUDA to accelerate training.

        ### CUDA Example (Matrix Multiplication):
        A simple example of how matrix multiplication is parallelized across GPU cores using CUDA. Each thread processes a portion of the matrix, allowing for faster computation compared to a CPU.

        ```c
        __global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }

        """
    )
    return


@app.cell
def _():
    # Install PyTorch if necessary
    # (use marimo's built-in package management features instead) !pip install torch torchvision
    return


@app.cell
def _(nn, optim, plt, time, torch):
    from torch.utils.data import DataLoader, TensorDataset
    print(f'cuDNN Available: {torch.backends.cudnn.enabled}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    torch.backends.cudnn.enabled = True

    class CNNModel(nn.Module):

        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 14 * 14, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def train_comparison(model, dataloader, optimizer, criterion, epochs=2):
        _model.train()
        total_loss = []
        total_time = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            start_time = time.time()
            for i, (inputs, _labels) in enumerate(dataloader):
                inputs, _labels = (inputs.cuda(), _labels.cuda())
                _optimizer.zero_grad()
                outputs = _model(inputs)
                loss = _criterion(outputs, _labels)
                loss.backward()
                _optimizer.step()
                epoch_loss += loss.item()
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time
            total_loss.append(epoch_loss)
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.4f} seconds')
        return (total_time, total_loss)
    _batch_size = 64
    _input_data = torch.randn(_batch_size * 10, 1, 28, 28)
    _labels = torch.randint(0, 10, (_batch_size * 10,))
    _dataset = TensorDataset(_input_data, _labels)
    _model = CNNModel().cuda()
    _optimizer = optim.Adam(_model.parameters(), lr=0.001)
    _criterion = nn.CrossEntropyLoss()
    print('\n--- Case 1: Without Proper CPU/GPU Utilization ---')
    dataloader_slow = DataLoader(_dataset, batch_size=_batch_size, shuffle=True, num_workers=0)
    time_slow, loss_slow = train_comparison(_model, dataloader_slow, _optimizer, _criterion)
    print('\n--- Case 2: With Proper CPU/GPU Utilization (cuDNN Enabled) ---')
    dataloader_fast = DataLoader(_dataset, batch_size=_batch_size, shuffle=True, num_workers=4)
    time_fast, loss_fast = train_comparison(_model, dataloader_fast, _optimizer, _criterion)
    print(f'\n--- Comparison ---')
    print(f'Total Time Without Proper Utilization: {time_slow:.4f} seconds')
    print(f'Total Time With Proper Utilization: {time_fast:.4f} seconds (cuDNN Enabled)')
    torch.backends.cudnn.enabled = False
    print(f'\n--- Case 3: Without cuDNN ---')
    time_no_cudnn, loss_no_cudnn = train_comparison(_model, dataloader_fast, _optimizer, _criterion)
    print(f'Total Time Without cuDNN: {time_no_cudnn:.4f} seconds')
    print(f'Total Time With cuDNN: {time_fast:.4f} seconds')
    plt.plot(loss_slow, label='Without Efficient Utilization')
    plt.plot(loss_fast, label='With Efficient Utilization (cuDNN)')
    plt.plot(loss_no_cudnn, label='Without cuDNN')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        # Comparison of Training Efficiency: CPU and GPU Utilization with cuDNN

        ## Results Summary:
        In the plot, we compare the training loss per epoch for three different setups:
        1. **Without Efficient Utilization**: Single-threaded data loading and inefficient CPU-GPU utilization.
        2. **With Efficient Utilization (cuDNN)**: Multiple CPU threads for data loading and cuDNN optimizations for GPU operations.
        3. **Without cuDNN**: Multiple CPU threads for data loading, but cuDNN is explicitly disabled to demonstrate the performance impact.

        ## Key Observations:

        ### 1. Without Efficient Utilization (Blue Line):
        - **Training Time**: This setup is the slowest. Without proper CPU parallelism, the GPU spends significant time idling while waiting for data from the CPU.
        - **Loss Reduction**: The loss decreases, but at a slower rate compared to the other setups, as the model is not utilizing the hardware efficiently.
        - **Main Bottleneck**: Single-threaded data loading causes a CPU bottleneck, limiting GPU utilization.

        ### 2. With Efficient Utilization (cuDNN) (Orange Line):
        - **Training Time**: This is faster than the "Without Efficient Utilization" case. Multiple CPU threads handle data loading and preprocessing in parallel, ensuring the GPU remains fed with data. Additionally, **cuDNN** accelerates convolutional and pooling operations.
        - **Loss Reduction**: The loss decreases at a faster rate than the inefficient setup, indicating better model training due to consistent GPU utilization.
        - **Main Improvement**: The parallel data loading reduces CPU bottlenecks, and cuDNN optimizations significantly speed up convolutional operations.

        ### 3. Without cuDNN (Green Line):
        - **Training Time**: Interestingly, this setup shows a faster loss reduction compared to the cuDNN-enabled case. However, this could be due to variations in batch processing speed or other small optimizations.
        - **Loss Reduction**: Even though cuDNN is disabled, the model still benefits from multiple CPU threads for data loading, which ensures that the GPU is efficiently utilized.
        - **Main Trade-off**: While cuDNN provides optimizations for deep learning operations, disabling it still maintains some efficiency due to proper CPU-GPU


        ----
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Gradient Accumulation in PyTorch

        ## What is Gradient Accumulation?
        Gradient accumulation is a technique that allows us to simulate larger batch sizes than our memory can handle by splitting a large batch into smaller mini-batches and accumulating the gradients over those mini-batches. The model parameters are updated only after a specified number of mini-batches.

        This technique is particularly useful when:
        - You have memory constraints (e.g., limited GPU memory).
        - You want to use a larger effective batch size for better generalization without actually increasing the batch size.

        ### How Gradient Accumulation Works:
        1. **Process Mini-Batches Sequentially**: Instead of processing a large batch at once, we break it into smaller mini-batches.
        2. **Accumulate Gradients**: For each mini-batch, we perform a forward pass and a backward pass but do **not** update the model parameters immediately.
        3. **Apply Update After Several Mini-Batches**: After processing a specified number of mini-batches, we update the model parameters once, simulating a larger effective batch size.

        ### Code Example for Gradient Accumulation in PyTorch:

        """
    )
    return


@app.cell
def _(nn, optim, time, torch):
    class _SimpleModel(nn.Module):

        def __init__(self):
            super(_SimpleModel, self).__init__()
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    data_size = 1024
    _input_data = torch.randn(data_size, 512)
    _labels = torch.randint(0, 10, (data_size,))
    _batch_size = 64
    _dataset = torch.utils.data.TensorDataset(_input_data, _labels)
    dataloader = torch.utils.data.DataLoader(_dataset, batch_size=_batch_size, shuffle=True)
    _model = _SimpleModel()
    _optimizer = optim.Adam(_model.parameters(), lr=0.001)
    _criterion = nn.CrossEntropyLoss()

    def train_with_accumulation(model, dataloader, accumulation_steps=8, epochs=5):
        _model.train()
        _optimizer.zero_grad()
        total_time = 0
        for epoch in range(epochs):
            running_loss = 0.0
            start_time = time.time()
            for i, (inputs, targets) in enumerate(dataloader):
                outputs = _model(inputs)
                loss = _criterion(outputs, targets)
                loss = loss / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    _optimizer.step()
                    _optimizer.zero_grad()
                running_loss += loss.item()
            end_time = time.time()
            total_time += end_time - start_time
            print(f'Epoch {epoch + 1}, Loss: {running_loss:.4f}')
        return total_time

    def train_without_accumulation(model, dataloader, epochs=5):
        _model.train()
        total_time = 0
        for epoch in range(epochs):
            running_loss = 0.0
            start_time = time.time()
            for i, (inputs, targets) in enumerate(dataloader):
                _optimizer.zero_grad()
                outputs = _model(inputs)
                loss = _criterion(outputs, targets)
                loss.backward()
                _optimizer.step()
                running_loss += loss.item()
            end_time = time.time()
            total_time += end_time - start_time
            print(f'Epoch {epoch + 1}, Loss: {running_loss:.4f}')
        return total_time
    print('\n--- Training Without Gradient Accumulation ---')
    model_no_accum = _SimpleModel()
    optimizer_no_accum = optim.Adam(model_no_accum.parameters(), lr=0.001)
    time_no_accum = train_without_accumulation(model_no_accum, dataloader)
    print('\n--- Training With Gradient Accumulation ---')
    model_accum = _SimpleModel()
    optimizer_accum = optim.Adam(model_accum.parameters(), lr=0.001)
    time_accum = train_with_accumulation(model_accum, dataloader, accumulation_steps=8)
    print(f'\n--- Time Comparison ---')
    print(f'Training Time Without Accumulation: {time_no_accum:.2f} seconds')
    print(f'Training Time With Accumulation: {time_accum:.2f} seconds')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Results and Benefits of Gradient Accumulation

        ## 1. Results Overview:

        ### Training Without Gradient Accumulation:
        - **Epoch Loss**:
          - Epoch 1, Loss: 37.3898
          - Epoch 2, Loss: 37.3898
          - Epoch 3, Loss: 37.3899
          - Epoch 4, Loss: 37.3898
          - Epoch 5, Loss: 37.3898
        - **Training Time**:
          - 0.30 seconds for 5 epochs.
  
        In this case, the model is updating its parameters after every mini-batch. The loss remains relatively high, and the time taken per epoch is quick due to frequent updates. However, with this setup, the batch size remains constrained by memory limitations.

        ### Training With Gradient Accumulation:
        - **Epoch Loss**:
          - Epoch 1, Loss: 4.6273
          - Epoch 2, Loss: 4.6273
          - Epoch 3, Loss: 4.6273
          - Epoch 4, Loss: 4.6273
          - Epoch 5, Loss: 4.6273
        - **Training Time**:
          - 0.15 seconds for 5 epochs.
  
        With gradient accumulation, the model simulates a larger batch size by accumulating gradients over 8 mini-batches before updating the model parameters. The loss is significantly lower, indicating more stable training. The time per epoch is also shorter, as the model updates the parameters less frequently (every 8 mini-batches).

        ---

        ## 2. Benefits of Using Gradient Accumulation

        ### a) **Memory Efficiency**:
        Gradient accumulation reduces memory demand by allowing you to train with effectively larger batch sizes. Instead of updating model parameters after every mini-batch, you can accumulate gradients and apply the update after processing multiple mini-batches. This is especially useful for:
        - **Memory-constrained environments** (e.g., small GPUs).
        - **Large-scale models** that would otherwise exceed memory capacity with large batch sizes.

        ### b) **Better Gradient Stability**:
        Larger batch sizes often result in smoother and more stable gradient updates, improving convergence. By accumulating gradients over several mini-batches, gradient accumulation simulates the effect of a large batch size, leading to:
        - **Lower loss**: As seen in the results, the loss was significantly lower with gradient accumulation.
        - **Improved model generalization**: Larger batch sizes typically lead to better generalization during training.

        ### c) **Training Time Efficiency**:
        Although the model updates parameters less frequently, we saw a faster overall training time with gradient accumulation (0.15 seconds vs. 0.30 seconds). This is because:
        - **Fewer update steps**: The model performs fewer weight updates, which reduces the computational overhead associated with frequent parameter updates.
        - **Larger effective batch size**: Fewer updates are required to achieve the same or better performance, leading to shorter training times.

        ### d) **Scaling for Larger Datasets**:
        Gradient accumulation is particularly beneficial when scaling models and datasets. For example, in distributed environments like TPUs or multi-GPU systems, using a larger effective batch size helps achieve faster convergence while keeping memory usage in check. When scaling up to more complex models or larger datasets, gradient accumulation helps:
        - **Efficiently utilize hardware resources**.
        - **Reduce training time for large-scale models**.

        ---

        ## 3. When to Use Gradient Accumulation?
        - **When your model cannot fit large batch sizes into memory**: This method allows you to accumulate gradients and simulate larger batch sizes without memory overflow.
        - **When training on distributed systems**: Accumulating gradients over multiple mini-batches helps maintain stability and leverage hardware more efficiently, especially when scaling up on multiple GPUs or TPUs.
        - **When aiming for better convergence**: By simulating larger batch sizes, gradient accumulation can lead to smoother gradients, better convergence, and improved generalization.
  
        In summary, gradient accumulation is an effective strategy to reduce memory usage, speed up training, and improve model performance when scaling to larger datasets or limited hardware environments.

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
        -----

        -----

        Please change the type of the machine to a TPU. You can do this on the top of the screen.

        ------
        ------
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Using TPUs in Google Colab with TensorFlow

        To take advantage of Google's **TPUs** (Tensor Processing Units) for faster and more efficient deep learning, follow these steps:

        ## Step 1: Select TPU as the Runtime
        1. **Go to the Runtime Menu** at the top of the Colab notebook.
        2. Select **Change runtime type**.
        3. Under **Hardware Accelerator**, choose **TPU** from the dropdown list.
        4. Click **Save**.

        This will enable the Colab environment to use a TPU for training.

        ## Step 2: Verify TPU Setup in TensorFlow

        Now, let's set up and verify that the TPU is available in TensorFlow. The following code will connect to the TPU and distribute the model for efficient training.


        """
    )
    return


@app.cell
def _():
    import tensorflow as tf

    def create_model():
        return tf.keras.Sequential([tf.keras.layers.Dense(512, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    with strategy.scope():
        _model = create_model()
        _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    import numpy as np
    _dataset = tf.data.Dataset.from_tensor_slices((np.random.randn(10000, 512), np.random.randint(0, 10, 10000))).batch(64)
    _model.fit(_dataset, epochs=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ----
        ----
        Answers to the short exercises above
        ---------

        ----
        -------



        ### Answer: Impact of Sigmoid Activation Function on Vanishing Gradients

        When we change the activation function from ReLU to Sigmoid, we notice a significant difference in the behavior of the model’s training process.

        **Observation**:
        - **Loss Behavior**: With Sigmoid, the loss decreases very slowly compared to ReLU, and in some cases, the loss may stagnate or stop decreasing altogether. This is a clear indication of the vanishing gradient problem.
        - **Gradient Flow**: The gradients with Sigmoid become very small, especially for deeper layers. This is because the Sigmoid function squashes input values into a range between 0 and 1, leading to very small gradients when backpropagating through layers. The further back you propagate, the smaller the gradients become, eventually approaching zero.

        **Why Does This Happen?**
        - Sigmoid’s derivative is small for values far from zero, meaning that for inputs significantly larger or smaller than 0, the gradient will be very close to 0. As we go deeper into the network, these small gradients multiply, causing them to approach zero. This effectively halts learning for the earlier layers, preventing them from updating their weights.
        - In contrast, ReLU does not suffer from this issue because it doesn’t squash values to a narrow range, allowing gradients to propagate more effectively.

        **Conclusion**:
        - The Sigmoid activation function is prone to vanishing gradients, especially in deep networks. This is why modern architectures tend to use ReLU or its variants, which avoid this issue.
        - This exercise demonstrates the importance of selecting appropriate activation functions when designing deep neural networks to avoid the vanishing gradient problem and ensure efficient training.


        -----

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Answer: Impact of Removing Batch Normalization on Training

        When Batch Normalization is removed from the model, the following changes are observed in the training process:

        **Observation**:
        - **Loss Behavior**: Without Batch Normalization, the training loss tends to decrease more slowly and can become unstable. There might be oscillations or sharp increases in loss at certain points during training, indicating less stability in the learning process.
        - **Gradient Norms**: Without Batch Normalization, the gradient norms can vary more widely across layers. This variance in gradients can lead to exploding gradients in some layers and vanishing gradients in others, particularly in deeper networks.

        **Why Does This Happen?**
        - **Internal Covariate Shift**: Batch Normalization helps to mitigate internal covariate shift, which is the change in the distribution of layer inputs during training. Without it, the network's activations become less predictable, making it harder for the model to learn stable representations.
        - **Gradient Flow**: By normalizing the inputs to each layer, Batch Normalization helps maintain a more consistent gradient flow across the network. Without it, deeper layers may struggle to receive strong enough gradients for effective learning.

        **Conclusion**:
        - Removing Batch Normalization leads to a less stable and slower training process. This exercise demonstrates the importance of using Batch Normalization to improve gradient flow and stabilize the learning process in deep networks, especially in deep architectures where gradient instability can become a major issue.

        -----
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

