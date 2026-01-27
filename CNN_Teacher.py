import torch
import torch.nn as nn
import torch.optim as optim
import copy

# model architecture
class TeacherWaveToMapNet(nn.Module):
    def __init__(self):
        super(TeacherWaveToMapNet, self).__init__()
        self.coord_adder = AddCoords()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # pooling
        self.global_pool = nn.AdaptiveAvgPool1d(25)

        # linear layer
        self.fc = nn.Linear(256 * 25, 128 * 16 * 16)

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.coord_adder(x)
        x = self.encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 16, 16)
        x = self.decoder(x)
        return x
    
# helper layer to help the model avoid ghosting sources
class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        batch_size, _, length = input_tensor.size()
        coords = torch.linspace(-1, 1, length).to(input_tensor.device)
        coords = coords.view(1, 1, length).expand(batch_size, 1, length)

        return torch.cat([input_tensor, coords], dim=1)

# training function for the teacher model
def train_teacher(model, train_loader, val_loader, epochs=30, lr=0.001):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
  criterion = nn.MSELoss() # usinng MSE loss for heatmap regression
  best_params = copy.deepcopy(model.state_dict())
  min_loss = float('inf')
  for epoch in range(epochs):
      model.train()
      running_loss = 0.0

      for _, inputs, _, target_heatmaps, n_src in train_loader:
          inputs, n_src, target_heatmaps = inputs.to(device), n_src.to(device), target_heatmaps.to(device)

          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, target_heatmaps)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      avg_loss = running_loss / len(train_loader)
      if (epoch+1) % 5 == 0:
          print(f"Teacher Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

  print("Training complete, Validating now")

  val_loss = 0.0
  model.eval()
  with torch.no_grad():
      for _, inputs, _, target_heatmaps, n_src in val_loader:
          inputs, n_src, target_heatmaps = inputs.to(device), n_src.to(device), target_heatmaps.to(device)
          val_loss += criterion(model(inputs), target_heatmaps).item()

  avg_val_loss = val_loss / len(val_loader)
  # save best params if this is the best validation loss
  if avg_val_loss < min_loss:
      best_params = copy.deepcopy(model.state_dict())
      min_loss = avg_val_loss

  print(f"Teacher Validation Loss: {avg_val_loss:.6f}")
  model.load_state_dict(best_params) # load best params