class HyperParameters:
    # Parameters
    num_steps = 10000
    batch_size = 64

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    num_input = 28*28  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)
