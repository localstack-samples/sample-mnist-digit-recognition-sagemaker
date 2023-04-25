import torch
from model import Net

model = Net()
model.load_state_dict(torch.load("ml/results/model.pth"))

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 28, 28)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("ml/results/model.pt")
