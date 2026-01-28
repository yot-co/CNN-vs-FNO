# data generation
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# a function to generate data for all models
def generate_unified_data(num_samples=5000, grid_size=32, noise_level=0.001, dropout_prob=0.3, min_vel=280.0, max_vel=320.0, two_source_input_prob=0.5, wave_freq=5.0, time_steps=200):
    dt = 0.01
    x_max, y_max = float(grid_size), float(grid_size)
    sensor_locs = np.array([[0, 0], [0, y_max], [x_max, 0], [x_max, y_max]]) # sensor locations at the 4 Corners


    X_clean = np.zeros((num_samples, 4, time_steps))
    y_locs = np.full((num_samples, 2, 2), -1.0)
    n_sources = np.zeros(num_samples, dtype=int)

    t_wave = np.arange(time_steps) * dt
    f0 = wave_freq
    wavelet_base = (1.0 - 2.0*(np.pi**2)*(f0**2)*(t_wave**2)) * np.exp(-(np.pi**2)*(f0**2)*(t_wave**2))
    wavelet_base = wavelet_base / np.max(np.abs(wavelet_base))


    for i in range(num_samples):
        vel = np.random.uniform(min_vel, max_vel)
        # randomly generate 1 or 2 sources sample
        num_src = 1 if np.random.rand() > two_source_input_prob else 2
        n_sources[i] = num_src

        current_signal = np.zeros((4, time_steps))

        for src_idx in range(num_src):
            src = np.random.rand(2) * [x_max, y_max]
            y_locs[i, src_idx] = src

            for s in range(4):
                dist = np.sqrt(np.sum((src - sensor_locs[s])**2))
                arrival_time = dist / vel
                shift_steps = int(arrival_time / dt)
                amplitude = 1.0 / (dist + 1.0)

                if shift_steps < time_steps:
                    valid_len = time_steps - shift_steps
                    current_signal[s, shift_steps:] += wavelet_base[:valid_len] * amplitude

        X_clean[i] = current_signal

    # teacher data part
    teacher_noise_tensor = np.random.normal(0, noise_level, X_clean.shape)
    X_teacher = X_clean + teacher_noise_tensor

    # student data part
    student_noise_tensor = np.random.normal(0, noise_level, X_clean.shape)
    X_student = X_clean + student_noise_tensor

    for i in range(num_samples):
        if np.random.rand() < dropout_prob:
            dead_sensor_idx = np.random.randint(0, 4)
            X_student[i, dead_sensor_idx, :] = 0.0

    return (
        torch.tensor(X_student, dtype=torch.float32),
        torch.tensor(X_teacher, dtype=torch.float32),
        torch.tensor(y_locs, dtype=torch.float32),
        torch.tensor(n_sources, dtype=torch.long)
    )


# gaussian map creator function
def create_mixed_gaussian_map(source_locs, grid_size=32, sigma=3.0):

    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    maps = []

    for i in range(len(source_locs)):
        # source 1
        cx1, cy1 = source_locs[i, 0]

        gauss1 = np.exp(-((xx - cx1)**2 + (yy - cy1)**2) / (2 * sigma**2))

        # source 2 if exists
        cx2, cy2 = source_locs[i, 1]

        # if there's only 1 source
        if cx2 == -1:
            combined = gauss1

        # if there are 2 sources
        else:
            gauss2 = np.exp(-((xx - cx2)**2 + (yy - cy2)**2) / (2 * sigma**2))
            combined = np.maximum(gauss1, gauss2)

        maps.append(combined)

    return np.array(maps)

# utility to get min value and its index from a histogram
def get_min_and_index(hist):
  for i, item in enumerate(hist):
    if item == np.min(hist):
      return i, item
    
# getting the peak location from heatmap
def get_peak_location(heatmap, grid_size=32):
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    pixel_size = float(grid_size) / heatmap.shape[0]
    pred_y = idx[0] * pixel_size
    pred_x = idx[1] * pixel_size
    return np.array([pred_x, pred_y])

# calculate accuracy of the model on a dataset provided by loader
def calculate_accuracy(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for inputs, _, true_coords, _, _ in loader:
            inputs = inputs.to(device)
            # true_coords shape: (batch_size, 2, 2)
            # dim 1 is source index (source 0, source 1)
            # dim 2 is (x, y)

            outputs = model(inputs).cpu().numpy()
            true_coords = true_coords.numpy()

            for i in range(len(inputs)):
                # find the actual peak
                pred_loc = get_peak_location(outputs[i, 0])

                # find distance to first true source
                dist1 = np.linalg.norm(true_coords[i, 0] - pred_loc)
                # save it
                valid_dists = [dist1]

                # check if the second source exists
                # if it does, calculate distance to it as well
                if true_coords[i, 1, 0] != -1:
                    dist2 = np.linalg.norm(true_coords[i, 1] - pred_loc)
                    valid_dists.append(dist2)

                # take the minimum distance as error
                errors.append(min(valid_dists))

    return np.mean(errors)

# creating data and dataloaders
def create_data_and_data_loaders(num_samples=5000, grid_size=32, noise_level=0.001, dropout_prob=0, min_vel=280.0, max_vel=320.0, two_source_input_prob=0.5, wave_freq=5.0, time_steps=200, sigma=3.0):
    X_s_raw, X_t_raw, Y_coords, n_sources = generate_unified_data(num_samples=num_samples, grid_size=grid_size, noise_level=noise_level,
                                                                dropout_prob=dropout_prob, min_vel=min_vel, max_vel=max_vel, two_source_input_prob=two_source_input_prob, wave_freq=wave_freq, time_steps=time_steps)

    Y_heatmaps_np = create_mixed_gaussian_map(Y_coords.numpy(), grid_size=32, sigma=sigma)
    Y_heatmaps = torch.tensor(Y_heatmaps_np, dtype=torch.float32).unsqueeze(1)

    # split 80/10/10
    N = len(X_s_raw)
    indices = torch.randperm(N)

    n_train = int(0.8 * N)
    n_val = int(0.1 * N)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train+n_val]
    test_idx  = indices[n_train+n_val:]

    # split all data
    X_s_train, X_s_val, X_s_test = X_s_raw[train_idx], X_s_raw[val_idx], X_s_raw[test_idx]
    X_t_train, X_t_val, X_t_test = X_t_raw[train_idx], X_t_raw[val_idx], X_t_raw[test_idx]

    Y_c_train, Y_c_val, Y_c_test = Y_coords[train_idx], Y_coords[val_idx], Y_coords[test_idx]
    Y_h_train, Y_h_val, Y_h_test = Y_heatmaps[train_idx], Y_heatmaps[val_idx], Y_heatmaps[test_idx]
    n_src_train, n_src_val, n_src_test = n_sources[train_idx], n_sources[val_idx], n_sources[test_idx]

    # normalize data
    #norm_s = torch.max(torch.abs(X_s_train))
    norm_s = torch.quantile(torch.abs(X_s_train), 0.99)
    X_s_train = X_s_train / norm_s
    X_s_val   = X_s_val   / norm_s
    X_s_test  = X_s_test  / norm_s

    # normalize teacher data
    norm_t = torch.max(torch.abs(X_t_train))
    X_t_train = X_t_train / norm_t
    X_t_val   = X_t_val   / norm_t
    X_t_test  = X_t_test  / norm_t

    # adding noise to the test set to test robustness
    fixed_noise = torch.randn(X_s_test.size()) * 0.005
    X_s_test = X_s_test + fixed_noise

    # creating dataloaders
    train_data = TensorDataset(X_s_train, X_t_train, Y_c_train, Y_h_train, n_src_train)
    val_data   = TensorDataset(X_s_val,   X_t_val,   Y_c_val,   Y_h_val,   n_src_val)
    test_data  = TensorDataset(X_s_test,  X_t_test,  Y_c_test,  Y_h_test,  n_src_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader
