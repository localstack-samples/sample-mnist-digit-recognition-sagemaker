
import torch
import json
import numpy as np


def output_fn(prediction_output, accept):
    torch.argmax(prediction_output, dim=1).item()
    number = torch.argmax(prediction_output, dim=1).item()
    print("returning response: ", number)
    return number

def predict_fn(input_data, model):
    tensor = torch.tensor(input_data)

    with torch.no_grad():
        print("starting inference")
        result = model.forward(tensor)
        print("result: ", result)

        return result

def input_fn(input_data, content_type):
    print("content_type: ", content_type)
    to_json = json.loads(input_data)
    print("to_json: ", to_json, type(to_json))
    to_np = np.array([to_json], dtype=np.float32)
    print("numpy input: ", to_np, to_np.shape)
    return to_np


def model_fn(model_dir):
    print("model_dir: ", model_dir)
    model = torch.jit.load("model.pt")
    return model

