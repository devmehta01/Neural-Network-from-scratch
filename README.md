# Neural-Network-from-scratch
This repository contains code that solves the entire forward and back propagation of Neural Networks. This code seeks the hyperparameter value from the user and hence an N layer network with N' hidden layers can be created.

The 2 files train_catvnoncat.h5 and test_catvnoncat.h5 are used to train and test the model.

Different modules are created as follows:

1. param_init(dim_layer) - initializes the weight matrix W and the bias b based upon the number of layers entered by the user. W is initialized with random numbers and b is initialized with 0s to break symmetry.

2. sigmoid(z) - The sigmoid activation function. a = 1/(1+exp(-z))

3. relu(z) - The REctifies Linear Unit activation funciton. a = maximum(0,z)

4. 3 modules for forward propagation {forward(a, W, b); forward_prop_activation(a_prev, W, b, act_fn); forward_prop(X, params)} - forward_prop starts the feed forward mechanism. It calls the forward_prop_activation module for the 1st layer to the 2nd last layer with relu activation function and for the last layer with sigmoid activation function. forward_prop_activation first calls the forward module which calculates the dot product as z = W.dot(a) + b and then the activation is applied to z.

5. error_calc(yhat, y) - Module to find the error between the predicted y values and the actual y values to then pass on to the backprop module. 2 types of loss functions are provided: mean square error and crossentropy error.

6. 5 modules for back propagation {back_prop(yhat, Y, prevs); back_prop_activation(da, prev, act_fn); back(dz, prev); back_prop_relu(da, prev); back_prop_sigmoid(da, prev)} - back_prop starts the back propagation mechanism and calls back_prop_activation first for the sigmoid layer and then for the relu layers in the reversed manner. back_prop_activation first calls back_prop_relu or back_prop_sigmoid based on the activation function and then calls back. back_prop_relu and back_prop_sigmoid calculate the gradients with respect to the activation function. back calculates the gradients of W and b.

7. weight_update(params, gradients, alpha_lr) - Once the gradients are calculated, we need to update the weights. Weight matrix is updated using the gradients and the learning rate.

8. load_data() - module to load the data. Loads the train_catvnoncat.h5 into train_x_orig and train_y, and test_catvnoncat.h5 into test_x_orig and test_y.

9. model_train(X, Y, dim_layers, alpha_lr, epochs) - Module that starts the entire train procedure, i.e. the forward propagation and the backward propagation. It first calls param_init with the dim_layers to intialize the parameters. Next, it calls the forward_prop module to start the forward propagation, then calls error_calc to calculate the error and back_prop to calculate the gradients. It then calls weight_update to calculate the new weights using the gradients. It runs this entire process from the forward prop to the backprop (weight initialization is only done once) for the number of epochs specified.
