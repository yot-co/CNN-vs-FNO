import torch
import torch.nn as nn
import torch.optim as optim
import copy
from Utilities.data_utilities import calculate_accuracy
import numpy as np

# model architecture
class WaveToMapNet(nn.Module):
    def __init__(self):
        super(WaveToMapNet, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )

        # linear layer
        self.fc = nn.Linear(3200, 128 * 8 * 8)

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.decoder(x)
        return x

# smaller model architecture
class WaveToMapNet_small(nn.Module):
    def __init__(self):
        super(WaveToMapNet_small, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )

        # linear layer
        self.bottleneck = nn.Sequential(
            nn.Linear(3200, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.decoder(x)
        return x
    
  
# at the start we should rely more heavily on the teacher, and over time (epochs), the studnet would rely more on itself
# this helper function implements this logic
def get_alpha(epoch, epochs):
  starting_alpha = 0.8
  min_alpha = 0.1 # the lowest we will go
  epoch_decay_limit = int(0.8 * epochs)
  if epoch >= epoch_decay_limit:
    return min_alpha

  progress = epoch / epoch_decay_limit
  return starting_alpha - (progress * (starting_alpha - min_alpha))

# training function for the student model with knowledge distillation
def train_student(model, teacher_model, train_loader, val_loader, test_loader, epochs=30, lr=0.001, save_best=False, patience=10, early_stopping=True):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss() # using MSE loss for heatmap regression
  best_params = copy.deepcopy(model.state_dict())
  min_loss = float('inf')
  teacher_model.eval()
  acc = []
  epochs_no_improve = 0 # for early stopping
  for epoch in range(epochs):
    alpha = get_alpha(epoch, epochs)
    model.train()
    running_loss = 0.0

    for inputs, teacher_inputs, _, target_heatmaps, n_src in train_loader:
      inputs, teacher_inputs, n_src, target_heatmaps = inputs.to(device), teacher_inputs.to(device), n_src.to(device), target_heatmaps.to(device)

      # add noise to student inputs
      random_factor = np.random.uniform(0.001, 0.008)
      noise = torch.randn_like(inputs) * random_factor
      student_inputs = inputs + noise

      # teacher evaluation
      with torch.no_grad():
        teacher_preds = teacher_model(teacher_inputs)

      student_preds = model(student_inputs)

      # calculate loss
      loss_truth = criterion(student_preds, target_heatmaps)
      loss_distill = criterion(student_preds, teacher_preds)

      loss = (1 - alpha) * loss_truth + alpha * loss_distill

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    current_test_error = calculate_accuracy(model, test_loader, device)
    acc.append(current_test_error)

    if (epoch+1) % 5 == 0:
        print(f"Distill Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.6f}")

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
      for inputs, _, _, target_heatmaps, n_src in val_loader:
        inputs, n_src, target_heatmaps = inputs.to(device), n_src.to(device), target_heatmaps.to(device)
        val_loss += criterion(model(inputs), target_heatmaps).item()

    avg_val_loss = val_loss / len(val_loader)
    # save best params during training
    if avg_val_loss < min_loss:
      best_params = copy.deepcopy(model.state_dict())
      min_loss = avg_val_loss
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1
      if early_stopping and epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

  print(f"Student Validation Loss: {avg_val_loss:.6f}")

  if save_best:
    model.load_state_dict(best_params) # load best params

  return acc