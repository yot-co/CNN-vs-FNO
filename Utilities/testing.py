import matplotlib.pyplot as plt
import numpy as np
from Models.CNN_Teacher import *
from Models.CNN_Student import *
from Models.FNO import *
from Utilities.data_utilities import calculate_accuracy, create_data_and_data_loaders, create_mixed_gaussian_map, generate_unified_data
import torch
from torch.utils.data import DataLoader, TensorDataset

# ====== General Utilities Function for Tests ======

#  function to calculate model size and number of parameters
def get_model_size_and_n_params(model, model_name):
  num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
  size_all_mb = (param_size + buffer_size) / 1024 ** 2

  print(f"{model_name} size: {size_all_mb:.2f} MB")
  print(f"number of trainable params: {num_trainable_params}")

  return size_all_mb, num_trainable_params

# ====== End of General Utilities Functions ======


# ====== First Test Functions - Error as a function of Epochs ======

# plotting function to compare results for the accuracy x epochs test
def plot_results(cnn_history, fno_history):
    epochs = len(cnn_history)
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(10, 6))

    # CNN part
    plt.plot(epochs_range, cnn_history, label='CNN Student', marker='o', linestyle='--', color='blue')

    # FNO part
    plt.plot(epochs_range, fno_history, label='FNO', marker='s', linestyle='-', color='orange')

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Position Error (Meters)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ====== End of First Test Functions ======

# ====== Second Test Functions - Error as a function of dataset size ======

# a complete function to run the second experiment
def run_efficiency_experiments(train_pool, val_dataset, test_tensors, sample_sizes, epochs=20, same_size_models=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # unpack train pool
    X_pool, X_t_pool, Y_c_pool, Y_h_pool, n_src_pool = train_pool
    X_test_clean, Y_coords_test, Y_heatmaps_test = test_tensors

    # prepare the validation loader
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # prepare the test loader, add noise
    X_test_noisy = X_test_clean + torch.randn_like(X_test_clean) * 0.005
    test_dataset = TensorDataset(X_test_noisy, torch.zeros_like(X_test_noisy), Y_coords_test, Y_heatmaps_test, torch.zeros(len(Y_coords_test)))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\n{'='*40}\nTraining Teacher\n{'='*40}")
    teacher = TeacherWaveToMapNet().to(device)

    # create full pool loader
    full_train_dataset = TensorDataset(*train_pool)
    full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)

    # train teacher
    train_teacher(teacher, full_train_loader, val_loader, epochs=epochs)
    teacher.eval() # freeze it

    results = {'sizes': [], 'cnn_acc': [], 'fno_acc': []}

    # train student and FNO on different subset sizes
    for size in sample_sizes:
        print(f"\nExperiment: Subset Size = {size}")

        # making sure there is no data leakage
        if size > len(X_pool):
            print(f"Warning: Requested size {size} is larger than training pool {len(X_pool)}")
            size = len(X_pool)

        subset_dataset = TensorDataset(
            X_pool[:size], X_t_pool[:size], Y_c_pool[:size], Y_h_pool[:size], n_src_pool[:size]
        )
        subset_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)

        # initialize models
        if same_size_models:
            cnn_student = WaveToMapNet_small().to(device)
            fno_student = LearnableBasisNet_Simple_small().to(device)
        else:
            cnn_student = WaveToMapNet().to(device)
            fno_student = LearnableBasisNet_Simple().to(device)

        # train the models
        print(f"Training CNN Student (Distilled from Oracle)...")
        cnn_hist = train_student(cnn_student, teacher, subset_loader, val_loader, test_loader, epochs=epochs)

        print(f"Training FNO...")
        fno_hist = train_fno(fno_student, subset_loader, val_loader, test_loader, epochs=epochs)

        # save the results for plotting and comparison
        results['sizes'].append(size)
        results['cnn_acc'].append(np.min(cnn_hist))
        results['fno_acc'].append(np.min(fno_hist))

    return results

# a function to setup the second experiment
def setup_global_experiment(total_samples=10000, val_size=1000, epochs=30, two_source_input_prob=0.5):
    test_size = val_size
    # generate the needed data
    X_s_raw, X_t_raw, Y_coords, n_src_raw = generate_unified_data(num_samples=total_samples, two_source_input_prob=two_source_input_prob)

    # convert the data to heatmaps
    Y_heatmaps_np = create_mixed_gaussian_map(Y_coords.numpy())
    Y_heatmaps = torch.tensor(Y_heatmaps_np, dtype=torch.float32).unsqueeze(1)

    # normalize the data
    norm_s = torch.max(torch.abs(X_s_raw))
    norm_t = torch.max(torch.abs(X_t_raw))

    X_s_norm = X_s_raw / norm_s
    X_t_norm = X_t_raw / norm_t

    n_reserved = val_size + test_size
    n_train_pool = total_samples - n_reserved

    # create a training set
    train_pool_tensors = (
        X_s_norm[:n_train_pool],
        X_t_norm[:n_train_pool],
        Y_coords[:n_train_pool],
        Y_heatmaps[:n_train_pool],
        n_src_raw[:n_train_pool]
    )

    # create a validation set
    val_dataset = TensorDataset(
        X_s_norm[n_train_pool : n_train_pool+val_size],
        X_t_norm[n_train_pool : n_train_pool+val_size],
        Y_coords[n_train_pool : n_train_pool+val_size],
        Y_heatmaps[n_train_pool : n_train_pool+val_size],
        n_src_raw[n_train_pool : n_train_pool+val_size]
    )

    # create a test set
    test_tensors = (
        X_s_norm[-test_size:],
        Y_coords[-test_size:],
        Y_heatmaps[-test_size:]
    )

    return train_pool_tensors, val_dataset, test_tensors

# plotting function to compare results for the sample second test
def plot_sample_efficiency(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results['sizes'], results['cnn_acc'], marker='o', label='CNN Student')
    plt.plot(results['sizes'], results['fno_acc'], marker='s', label='FNO')

    plt.title("Sample Efficiency Comparison")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Mean Position Error (Meters)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ====== End of Second Test Functions ======


# ====== Third Test Functions - Malfunctioning Sensor Robustness ======
# function to create a malfunctioning sensor data loader
def create_malfunction_loader(original_loader, target_sensor_idx=0, high_noise_level=0.1):
    malfunction_data = []

    for inputs, t_in, coords, hmaps, n_src in original_loader:

        # create a copy of the inputs to have a clean and corrupt one
        corrupt_inputs = inputs.clone()

        batch_size = inputs.shape[0]

        # random mask for the malfunctioning sensor
        is_dead_mask = torch.rand(batch_size) > 0.5

        for i in range(batch_size):
            if is_dead_mask[i]:
                # dead sensor case
                corrupt_inputs[i, target_sensor_idx, :] = 0.0
            else:
                # high noise case
                noise = torch.randn_like(corrupt_inputs[i, target_sensor_idx, :]) * high_noise_level
                corrupt_inputs[i, target_sensor_idx, :] += noise

        malfunction_data.append((corrupt_inputs, t_in, coords, hmaps, n_src))

    return malfunction_data

# plotting function to compare third test results
def plot_robustness_comparison(c_clean, c_broken, f_clean, f_broken):
  labels = ['Standard Test Set', 'Malfunction Test Set']
  cnn_scores = [c_clean, c_broken]
  fno_scores = [f_clean, f_broken]

  x = np.arange(len(labels))
  width = 0.35

  plt.figure(figsize=(8, 6))
  plt.bar(x - width/2, cnn_scores, width, label='CNN Student', color='royalblue')
  plt.bar(x + width/2, fno_scores, width, label='FNO', color='darkorange')

  plt.ylabel('Mean Position Error (Meters)')
  plt.title('Robustness to Sensor Malfunction (Dead/Noisy) - 1 Wave Source')
  plt.xticks(x, labels)
  plt.legend()
  plt.grid(axis='y', alpha=0.3)
  plt.show()

def run_sensor_malfunction_experiment(total_samples=5000, epochs=20, sensor_idx=0, grid_size=32, noise_level=0.001, dropout_prob=0.3, min_vel=280.0, max_vel=320.0, two_source_input_prob=0.0, wave_freq=5.0, time_steps=200, sigma=3.0, same_size_models=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nRunning Sensor Malfunction Robustness Test\n{'='*60}")
    # creating loaders
    train_loader, val_loader, test_loader = create_data_and_data_loaders(total_samples, grid_size, noise_level, dropout_prob, min_vel, max_vel, two_source_input_prob, wave_freq, time_steps, sigma)


    # train models on clean data
    print("Training models on clean data")
    teacher = TeacherWaveToMapNet().to(device)
    if same_size_models:
        cnn = WaveToMapNet_small().to(device)
        fno = LearnableBasisNet_Simple_small().to(device)
    else:
        cnn = WaveToMapNet().to(device)
        fno = LearnableBasisNet_Simple().to(device)

    train_teacher(teacher, train_loader, val_loader, epochs=epochs)
    print("Teacher Trained")

    train_student(cnn, teacher, train_loader, val_loader, test_loader, epochs=epochs)
    print("CNN Student Trained")

    train_fno(fno, train_loader, val_loader, test_loader, epochs=epochs)
    print(" FNO Trained")

    # create corrupted data
    print(f"\n Generating Malfunction Test Set (Sensor {sensor_idx} -> Dead/Noisy)...")
    malfunction_loader = create_malfunction_loader(test_loader, target_sensor_idx=sensor_idx, high_noise_level=0.5)

    print("\n Evaluating...")

    # check accuracy on the clean dataset
    acc_cnn_clean = calculate_accuracy(cnn, test_loader, device)
    acc_fno_clean = calculate_accuracy(fno, test_loader, device)

    # check accuracy on the corrupted dataset
    acc_cnn_broken = calculate_accuracy(cnn, malfunction_loader, device)
    acc_fno_broken = calculate_accuracy(fno, malfunction_loader, device)

    # plot
    print(f"\nResults Summary:")
    print(f"CNN | Clean Error: {acc_cnn_clean:.2f}m | Broken Sensor Error: {acc_cnn_broken:.2f}m")
    print(f"FNO | Clean Error: {acc_fno_clean:.2f}m | Broken Sensor Error: {acc_fno_broken:.2f}m")

    return acc_cnn_clean, acc_cnn_broken, acc_fno_clean, acc_fno_broken

# ====== End of Third Test Functions ======

# ====== Fourth Test Functions - Noisy Data Robustness ======
# a complete function to run the fourth experiment
def run_noise_robustness_test(cnn_model, fno_model, num_samples=5000, train_noise_level=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # different noise levels to test
    noise_levels = [0.0, 0.005, 0.01, 0.05, 0.1]

    results = {
        'noise_levels': noise_levels,
        'cnn_errors': [],
        'fno_errors': []
    }

    print(f"Generating Base Physics ({num_samples} samples)...")
    # generate nearly clean data
    X_s_clean, _, Y_coords, _ = generate_unified_data(
        num_samples=num_samples,
        noise_level=0.001,
        dropout_prob=0,
        two_source_input_prob=0,
        time_steps=200
    )

    # normalize the clean data
    norm_val = torch.quantile(torch.abs(X_s_clean), 0.99)
    X_s_clean = X_s_clean / norm_val

    # generate heatmaps for comparison
    Y_heatmaps_np = create_mixed_gaussian_map(Y_coords.numpy())
    Y_heatmaps = torch.tensor(Y_heatmaps_np, dtype=torch.float32).unsqueeze(1)

    print(f"{'='*50}")
    print(f"Starting Noise Robustness Test")
    print(f"{'='*50}")

    for noise_std in noise_levels:
        print(f"Testing Noise Level: {noise_std}...", end="")

        # set the seed
        g_cpu = torch.Generator()
        g_cpu.manual_seed(42)

        # add current noise level to the normalized clean data
        noise = torch.randn(X_s_clean.size(), generator=g_cpu) * noise_std
        X_test_noisy = X_s_clean + noise

        # create current test dataset
        test_dataset = TensorDataset(
            X_test_noisy,
            torch.zeros_like(X_test_noisy), 
            Y_coords,
            Y_heatmaps,
            torch.zeros(len(Y_coords)) 
        )

        # create current test loader
        current_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # evaluate CNN and save the results
        cnn_err = calculate_accuracy(cnn_model, current_test_loader, device)
        results['cnn_errors'].append(cnn_err)

        # evaluate FNO and save the results
        fno_err = calculate_accuracy(fno_model, current_test_loader, device)
        results['fno_errors'].append(fno_err)

        print(f" Done. (CNN: {cnn_err:.2f}m | FNO: {fno_err:.2f}m)")

    return results

# plotting function to compare fourth test results
def plot_noise_robustness(results, train_noise_level=0.001):
    levels = results['noise_levels']
    cnn_err = results['cnn_errors']
    fno_err = results['fno_errors']

    plt.figure(figsize=(10, 6))

    plt.plot(levels, cnn_err, 'o--', label='CNN Student', color='blue', linewidth=2)
    plt.plot(levels, fno_err, 's-', label='FNO', color='orange', linewidth=2)

    plt.title("Model Robustness: Accuracy vs. Noise Level", fontsize=14)
    plt.xlabel("Noise Standard Deviation ($\\sigma$)", fontsize=12)
    plt.ylabel("Mean Position Error (Meters)", fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.axvline(x=train_noise_level, color='green', linestyle=':', label=f'Training Noise Level ({train_noise_level})')
    plt.legend()

    plt.show()


# ====== End of Fourth Test Functions ======