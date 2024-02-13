# Importing essential libraries
import gradio as gr
import torch
from torch import nn
import numpy as np
from PIL import Image

# Creating Neural Model class used for training and validation
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Intializing and loading the saved model
model = NeuralNetwork()
state_dict = torch.load('model.pt',    map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# Preprocessing input value for giving to function
def preprocess_input(input):
    input = input['composite']

    # Convert the image data to a PIL Image
    input = Image.fromarray(input)

    # Resize the image to match the input size expected by your model
    input = input.resize((28, 28))

    # Convert the image to grayscale
    input = input.convert('L')

    # Flatten the pixel values
    input = np.array(input)

    return input

# Define a predict function
def predict(img):
    x = torch.tensor(preprocess_input(img), dtype=torch.float32)
    with torch.no_grad():
        return model(x.unsqueeze(0)).argmax().item()
    
# Design UI
gr.Interface(fn=predict,
             inputs="sketchpad",
             outputs="label",
             live=True).launch(share=True)