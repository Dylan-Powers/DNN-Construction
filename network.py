import torch
from layers import Input, Layer

class Network:
    def __init__(self):
        """
        TODO: Initialize a `layers` attribute to hold all the layers in the gradient tape.
        """
        self.layers = []
        
        # this layer is expected to be the first layer in the network and eventually be populated with the input
        self.input_layer = None

        # this layer is expected to contain the classifications attribute
        self.output_layer = None

        # this layer is expected to have its output set to the loss of the network
        self.loss_layer = None

        # this layer is expected to eventually contain the true label for that batch
        self.true_label_layer = None

    def add(self, layer: Layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        if self.input_layer is None:
            raise ValueError("Input must be set before adding layers")
        self.layers.append(layer)

    def set_input(self, input: Input):
        """
        :param input: The sublayer that represents the signal input (e.g., the image to be classified)
        """
        if self.input_layer is not None:
            raise ValueError("Input already set")
        elif len(self.layers) > 0:
            raise ValueError("Input must be set before adding layers")
        else:
            self.layers.append(input)
            self.input_layer = input

    def set_output(self,output: Layer):
        """
        :param output: SubLayer that produces the useful output (e.g., clasification decisions) as its output.
        """
        # This becomes messier when your output is the variable o from the middle of the Softmax
        # layer -- I used try/catch on accessing the layer.classifications variable.
        # when trying to access read the output layer's variable -- and that ended up being in a
        # different method than this one.        
        self.layers.append(output)
        self.output_layer = output

    def set_true_label(self, true_label: Layer):
        """
        Should be called after set_loss
        :param true_label: SubLayer that produces the true label as its output.
        """
        self.layers.append(true_label)
        self.true_label_layer = true_label
    
    def set_loss(self, loss: Layer):
        """
        Should be called after set_output
        :param loss: SubLayer that produces the loss value as its output.
        """
        if self.output_layer is None:
            raise ValueError("Output must be set before adding loss")
        else:
            self.layers.append(loss)
            self.loss_layer = loss

    def forward(self, input: torch.Tensor, expected_output: torch.Tensor):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward

        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.

        # This assumes that the input layer is the first layer in the network.
        # This is guaranteed by the checks in add() and set_input()
        self.input_layer.set(input)
        self.true_label_layer.set(expected_output)
        
        for layer in self.layers:
            layer.forward()
        
        return self.output_layer.output, self.loss_layer.output

    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation backward through the 
        gradient tape.

        """
        self.loss_layer.output_grad = torch.tensor(1.0)

        for layer in reversed(self.layers):
            # print(f"Backward: {layer}")
            layer.backward()

    def step(self, step_size: float):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 

        """
        for layer in self.layers:
            # print(f"Step: {layer}")
            layer.step(step_size)

    def clear_grad(self):
        """
        Reset the gradient of all learnable parameters to zero
        """
        for layer in self.layers:
            layer.clear_grad()

