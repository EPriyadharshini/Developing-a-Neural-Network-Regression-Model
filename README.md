# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1131" height="800" alt="image" src="https://github.com/user-attachments/assets/7debdd98-96a7-4fe7-a009-cbc7cd3bba45" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: PRIYADHARSHINI E

### Register Number: 212223230159

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```
dataset1 = pd.read_csv('/content/CSV FILE.csv')
X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
```
```
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```
```
# Name:PRIYADHARSHINI E
# Register Number:212223230159
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
        
  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```
# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)
```
```
# Name:PRIYADHARSHINI E
# Register Number:212223230159
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()




        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
```
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
```
```
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
```
```
loss_df = pd.DataFrame(ai_brain.history)
```
```
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```
```
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```




### Dataset Information
<img width="1061" height="659" alt="image" src="https://github.com/user-attachments/assets/b11abbd6-b613-4e10-9d54-e14d2678ea1f" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="819" height="767" alt="image" src="https://github.com/user-attachments/assets/767fadcc-3d30-455b-b71e-5c60fa80103a" />


### New Sample Data Prediction
<img width="995" height="162" alt="image" src="https://github.com/user-attachments/assets/bd9e8ccb-8bcf-42a0-bff7-186d3cfe6fed" />
<img width="737" height="171" alt="image" src="https://github.com/user-attachments/assets/559aa5e1-eb5e-43d0-b7a9-bac7ad21fa78" />
<img width="277" height="781" alt="image" src="https://github.com/user-attachments/assets/7f4a2240-fda0-494c-88a9-afe74328209b" />
<img width="813" height="320" alt="image" src="https://github.com/user-attachments/assets/125ef8da-8875-4ba6-8dfb-3931cff55acc" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
