import torch
import torch.nn as nn
from torch import optim
import copy
from Utilities.data_utilities import calculate_accuracy
import numpy as np

class LearnableBasisNet_Simple(nn.Module):
  def __init__(self, num_sensors=4, time_steps=200, grid_size=32):
    super(LearnableBasisNet_Simple, self).__init__()
    self.grid_size = grid_size
    self.time_steps = time_steps

    self.hidden_dim = 128
    self.learnable_transform = nn.Linear(time_steps, self.hidden_dim)
    self.norm1 = nn.LayerNorm(self.hidden_dim)
    input_dim = num_sensors * self.hidden_dim

    self.spectral_mixer = nn.Sequential(
        nn.Linear(input_dim, 2048),
        nn.ReLU(),
        nn.LayerNorm(2048),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, grid_size * grid_size)
    )

  def forward(self, x):
    batch_size = x.size(0)
    x_flat = x.view(-1, self.time_steps)

    x_freq = self.learnable_transform(x_flat)
    x_freq = torch.relu(x_freq)
    x_freq = self.norm1(x_freq)
    x_freq = x_freq.view(batch_size, -1)
    x_spatial = self.spectral_mixer(x_freq)

    output_map = x_spatial.view(batch_size, 1, self.grid_size, self.grid_size)
    output_map = torch.sigmoid(output_map)
    return output_map

# smaller model architecture
class LearnableBasisNet_Simple_small(nn.Module):
  def __init__(self, num_sensors=4, time_steps=200, grid_size=32):
    super(LearnableBasisNet_Simple_small, self).__init__()
    self.grid_size = grid_size
    self.time_steps = time_steps

    self.hidden_dim = 128
    self.learnable_transform = nn.Linear(time_steps, self.hidden_dim)
    self.norm1 = nn.LayerNorm(self.hidden_dim)
    input_dim = num_sensors * self.hidden_dim

    self.spectral_mixer = nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.ReLU(),
        nn.LayerNorm(1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, grid_size * grid_size)
    )

  def forward(self, x):
    batch_size = x.size(0)
    x_flat = x.view(-1, self.time_steps)

    x_freq = self.learnable_transform(x_flat)
    x_freq = torch.relu(x_freq)
    x_freq = self.norm1(x_freq)
    x_freq = x_freq.view(batch_size, -1)
    x_spatial = self.spectral_mixer(x_freq)

    output_map = x_spatial.view(batch_size, 1, self.grid_size, self.grid_size)
    return output_map
  

def train_fno(model, train_loader, val_loader, test_loader, epochs=30, lr=0.001, save_best=False, patience=10, early_stopping=True):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  opt_fno = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
  criterion = nn.MSELoss()
  acc = []
  best_loss = float('inf')
  best_params = copy.deepcopy(model.state_dict())
  epochs_no_improve = 0 # for early stopping

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, _, _, target_heatmaps, _ in train_loader:
      inputs, target_heatmaps = inputs.to(device), target_heatmaps.to(device)

      random_factor = np.random.uniform(0.001, 0.008)
      noise = torch.randn_like(inputs) * random_factor
      inputs = inputs + noise

      opt_fno.zero_grad()
      pred = model(inputs)
      loss = criterion(pred, target_heatmaps)
      loss.backward()
      opt_fno.step()

      running_loss += loss.item()


    current_test_error = calculate_accuracy(model, test_loader, device)
    acc.append(current_test_error)

    # Print loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"FNO Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.6f}")

    if save_best or early_stopping:
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
        for _, inputs, _, target_heatmaps, _ in val_loader:
          inputs, target_heatmaps = inputs.to(device), target_heatmaps.to(device)
          pred = model(inputs)
          val_loss += criterion(pred, target_heatmaps).item()

      avg_val_loss = val_loss / len(val_loader)

      if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_params = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
      else:
        epochs_no_improve += 1
        if early_stopping and epochs_no_improve >= patience:
          print(f"Early stopping at epoch {epoch+1}")
          break

  if save_best:
    model.load_state_dict(best_params) # load best params at the end of training

  return acc