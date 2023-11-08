import torch
import math

# TODO: Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.


class Layer:
    def __init__(self):
        """
        TODO: Add arguments and initialize instance attributes here.
        """
        self.output_grad = None
        self.output_grad_reset = False
        self.output = None

    def accumulate_grad(self, grad: torch.Tensor):
        """
        TODO: Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """
        # if self.output.size() != grad.size():
        #     raise ValueError("Output gradient size cannot change")
        if self.output_grad is None:
            self.output_grad = grad
        else:
            self.output_grad += grad

    def clear_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        if self.output_grad is not None:
            self.output_grad = torch.zeros(self.output_grad.size())

    def step(self, _):
        """
        TODO: Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    def __init__(self, trainable: bool = False, output: torch.Tensor = None, size: torch.Size = None):
        super().__init__()

        self.trainable = trainable
        self.output_set = False
        if output is not None:
            self.output = output
            self.output_set = True
        elif size is not None:
            self.output = torch.zeros(size)
            self.randomize()
        else:
            raise ValueError("Must provide either output or size")


    def set(self, value: torch.Tensor):
        """
        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        if self.output.size()[0] != value.size()[0]:
            raise ValueError("Output size cannot change")
        self.output = value
        self.output_set = True

    def randomize(self):
        """
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.randn(self.output.size()) * 0.1
        self.output_set = True

    def forward(self):
        """
        This method does nothing as the Input layer should already have its output set.
        """
        pass

    def backward(self):
        """
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self, step_size: float = 0.1):
        """
        TODO: Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """
        if self.trainable:
            if self.output_grad is  None:
                raise ValueError("Output gradient is not set")
            else:
                if math.isnan(self.output_grad[0,0]):
                    thing = 10
                self.output -= self.output_grad * step_size
        
class Linear(Layer):
    def __init__(self, weight_layer: Input, bias_layer: Input, input_layer: Layer):
        super().__init__()

        self.weight_layer = weight_layer
        self.bias_layer = bias_layer
        self.input_layer = input_layer

    def forward(self):
        self.output = torch.matmul(self.weight_layer.output, self.input_layer.output).add(self.bias_layer.output)

    def backward(self):
        self.weight_layer.accumulate_grad(torch.matmul(self.output_grad, self.input_layer.output.T))
        self.bias_layer.accumulate_grad(self.output_grad)
        self.input_layer.accumulate_grad(torch.matmul(self.weight_layer.output.T, self.output_grad))


class ReLU(Layer):
    def __init__(self, input: Layer):
        super().__init__()

        self.input = input

    def forward(self):
        """
        Run the ReLU function on the input and return the result.
        """
        self.output = torch.abs(self.input.output * (self.input.output > 0))

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.input.accumulate_grad(self.output_grad * (self.input.output > 0))


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, y_hat: Layer, y: Input):
        super().__init__()

        self.y_hat = y_hat
        self.y = y

    def forward(self):
        sum_ = torch.sum(((self.y.output - self.y_hat.output) ** 2))
        self.output = sum_ / self.y.output.size()[1]

    def backward(self):
        self.y_hat.accumulate_grad(2 * (self.y_hat.output - self.y.output) / self.y.output.size()[1])


class Regularization(Layer):
    def __init__(self, weight_layer: Input):
        super().__init__()

        self.weight_layer = weight_layer

    def forward(self):
        self.output = torch.sum(self.weight_layer.output ** 2)

    def backward(self):
        self.weight_layer.accumulate_grad(2 * self.weight_layer.output * self.output_grad)


class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    TODO: Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, input_layer: Layer, y: Input, epsilon=1e-8):
        super().__init__()

        self.classifications = None # classification output
        self.input_layer = input_layer
        self.y = y
        self.epsilon = epsilon

    def forward(self):
        small = self.input_layer.output - torch.max(self.input_layer.output)

        # Perform softmax
        exp = torch.exp(small)
        self.classifications = torch.div(exp, torch.sum(exp, 0))

        # Compute cross-entropy loss
        self.output = torch.sum(-1 * self.y.output * torch.log(self.classifications + self.epsilon))

    def backward(self):
        self.input_layer.accumulate_grad(self.classifications - self.y.output)

class Sum(Layer):
    def __init__(self, *input_layers: Layer, lambda_=0.00001):
        super().__init__()

        self.input_layers = input_layers
        self.lambda_ = lambda_

    def forward(self):
        self.output = torch.sum(torch.stack([layer.output for layer in self.input_layers]), 0) * self.lambda_

    def backward(self):
        for layer in self.input_layers:
            layer.accumulate_grad(self.output_grad * self.lambda_)
