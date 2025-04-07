import torch
import torch.autograd as autograd


def calculate_jacobian(model, input_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_data)
    num_outputs = output.size()[1]  # Number of output features
    num_inputs = input_data.size()[1]  # Number of input features
    jacobian_matrix = torch.zeros(num_outputs, num_inputs)
    for i in range(num_outputs):
        jacobian_matrix[i] = autograd.grad(output[0, i], input_data, retain_graph=True)[0]
  
jacobian_matrix = calculate_jacobian(model, x_values)

# Print the Jacobian matrix
print("Jacobian Matrix:")