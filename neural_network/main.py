import torch
import random
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler

from dataset import RegressionDataset
from model import MLP_Lipo, MLP_Abalone, MLP_Inhibition

LOGD74_DIR = 'data/logd74'
ABALONE_DIR = 'data/abalone_age'
INHIBITION_DIR = 'data/inhibition_data'

RANDOM_SAMPLING = 0
UNCERTAINTY_SAMPLING = 1
COVDROP = 2

def get_batch_correlation(train_dataset, next_indices)->np.float_:
    batch = np.zeros((len(next_indices), len(train_dataset[0][0])))
    for i, next_idx in enumerate(next_indices):
        features, _ = train_dataset[next_idx]
        batch[i] = features

    pearson_coefs = np.corrcoef(batch)
    n,_ = batch.shape

    iu1 = np.triu_indices(n=n,k=1)

    avg_pairwise_cor = np.mean(pearson_coefs[iu1])

    return avg_pairwise_cor

def scale_features(features):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    features = [feature for feature in features]
    
    return features

def get_datasets(features_path, targets_path, prop_train=0.8):
    np.random.seed(0)
    
    # turning these into python lists so adding to them within a 
    # Dataset object is a fast operation
    features = list(np.load(features_path))
    for i, datum in enumerate(features):
        features[i] = list(datum)

    features = scale_features(features)

    targets = list(np.load(targets_path))

    num_samples = len(targets)
    cutoff_index = int(num_samples * prop_train)
    
    perm_indices = np.random.permutation(num_samples)
    train_indices = perm_indices[:cutoff_index]
    test_indices = perm_indices[cutoff_index:]

    train_features = [features[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]

    train_targets = [targets[i] for i in train_indices]
    test_targets = [targets[i] for i in test_indices]

    train_dataset = RegressionDataset(train_features, train_targets)
    test_dataset = RegressionDataset(test_features, test_targets)

    return train_dataset, test_dataset

def train(model, dataset, optimizer, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        num_samples = 0
        for batch in dataloader:
            data, labels = batch
            optimizer.zero_grad()
            output = model(data)
            
            loss = torch.sum((output - labels) ** 2)
            loss.backward()
            optimizer.step()

            num_samples += len(data)
            total_epoch_loss += loss.item()

        avg_epoch_loss = total_epoch_loss / num_samples

def test(model, test_dataset, batch_size):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    total_loss = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch
            output = model(data)
            loss = torch.sum((output - labels) ** 2)

            total_loss += loss
            num_samples += len(data)

    mse_loss = total_loss / num_samples
    return mse_loss

def random_sample(not_chosen_indices, batch_size):
    return random.sample(list(not_chosen_indices), batch_size)

def uncertainty_sample(not_chosen_indices, model, train_dataset, batch_size, n=25):
    not_chosen_indices_list = list(not_chosen_indices)
    num_indices = len(not_chosen_indices_list)
    num_batches = (num_indices + batch_size - 1) // batch_size

    preds = torch.zeros((num_indices, n), dtype=torch.float)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_indices)
        batch_indices = not_chosen_indices_list[start_idx:end_idx]

        fingerprints = [train_dataset[idx][0] for idx in batch_indices]
        fingerprints = torch.stack(fingerprints)

        for j in range(n):
            logd74s = model(fingerprints)
            preds[start_idx:end_idx, j] = logd74s.squeeze()

    preds_std = preds.std(dim=1)
    max_indices = torch.argsort(preds_std, descending=True)[:batch_size]
    next_indices = [not_chosen_indices_list[max_idx] for max_idx in max_indices]

    return next_indices

def get_covariance_approx(not_chosen_indices, model, train_dataset, num_passes=100):
    not_chosen_data = [train_dataset[i][0] for i in not_chosen_indices]
    not_chosen_data = torch.stack(not_chosen_data, dim=0)
    
    results = torch.zeros((num_passes, len(not_chosen_data)), dtype=torch.float)

    # do multiple forward passes with all not_chosen_indices
    for p in range(num_passes):
        results[p] = model(not_chosen_data)

    means = torch.mean(results, dim=0)
    deviations = results - means
    covariance_matrix = torch.matmul(deviations.T, deviations) / (results.size(0) - 1)

    return covariance_matrix

def covdrop_sample(not_chosen_indices, model, train_dataset, batch_size, M=20):
    model.train()

    # approximate covariance matrix
    not_chosen_indices = list(not_chosen_indices)
    cov = get_covariance_approx(not_chosen_indices, model, train_dataset)
    var = torch.diag(cov)

    # select M batches using prob distribution over variances
    all_batch_indices = torch.zeros((M, batch_size), dtype=torch.int)
    for batch_id in range(M):
        batch_indices = torch.multinomial(var, num_samples=batch_size, replacement=False)
        all_batch_indices[batch_id] = batch_indices

    # calculate determinant for each of these batches
    prev_batch_det = float('-inf')
    best_batch_det = float('-inf')
    for batch_idx, batch_indices in enumerate(all_batch_indices):
        cur_batch_det = torch.det(cov[batch_indices[:, None], batch_indices])
        if cur_batch_det > best_batch_det:
            best_batch_det = cur_batch_det
            best_batch_idx = batch_idx

    i = 0
    iterations = 0
    max_iterations = 3
    while (best_batch_det > prev_batch_det) and (iterations < max_iterations):
        prev_batch_det = best_batch_det
        # for each batch
        for batch_idx in range(len(all_batch_indices)):
            # calculate batch det
            best_det = torch.det(
                cov[
                    all_batch_indices[batch_idx][:, None], 
                    all_batch_indices[batch_idx]
                ]
            )
            best_j = all_batch_indices[batch_idx, i]

            # for each sample
            for j in range(len(not_chosen_indices)):
                if j in all_batch_indices[batch_idx]:
                    continue

                # swap sample into position i
                all_batch_indices[batch_idx, i] = j

                # recalculate det
                det = torch.det(
                    cov[
                        all_batch_indices[batch_idx][:, None], 
                        all_batch_indices[batch_idx]
                    ]
                )
    
                # update det and batch if it beats previous batch best
                if det >= best_det:
                    best_det = det
                    best_j = j

            # update batch with best_j
            all_batch_indices[batch_idx, i] = best_j
            # update overall best with batch best
            if best_det > best_batch_det:
                best_batch_det = best_det
                best_batch_idx = batch_idx

        i = (i + 1) % batch_size
        iterations += 1

    best_batch_indices = all_batch_indices[best_batch_idx].tolist()
    best_batch_indices_nc = [not_chosen_indices[i] for i in best_batch_indices]

    return best_batch_indices_nc

def get_next_idx(
    not_chosen_indices: set, 
    model, 
    train_dataset,
    batch_size, 
    sampling_type
):
    if sampling_type == RANDOM_SAMPLING:
        next_indices = random_sample(not_chosen_indices, batch_size)
    elif sampling_type == UNCERTAINTY_SAMPLING:
        next_indices = uncertainty_sample(
            not_chosen_indices, 
            model, 
            train_dataset,
            batch_size
        )
    elif sampling_type == COVDROP:
        next_indices = covdrop_sample(
            not_chosen_indices, 
            model, 
            train_dataset,
            batch_size
        )

    return next_indices

def active_learn(
    start_prop,
    end_prop,
    train_dataset,
    test_dataset,
    model_class,
    optimizer_class,
    lr,
    train_batch_size,
    epochs,
    label_batch_size,
    sampling_type
):
    # randomly choose start_prop of data
    n = len(train_dataset)
    start_len = int(n * start_prop)

    perm = np.random.permutation(n)
    chosen_indices, not_chosen_indices = set(perm[:start_len]), set(perm[start_len:])

    features = [train_dataset[i][0].tolist() for i in chosen_indices]
    targets = [train_dataset[i][1].item() for i in chosen_indices]
    chosen_dataset = RegressionDataset(features, targets)

    test_losses = []
    correlations = []
    while (len(chosen_dataset)/n) < end_prop:
        # train the model on the dataset with that point
        model = model_class()
        optimizer = optimizer_class(model.parameters(), lr=lr)
        train(model, chosen_dataset, optimizer, epochs, train_batch_size)
    
        # calculate and record the test loss
        test_loss = test(model, test_dataset, train_batch_size)
        if (len(test_losses) + 1) % 10 == 0:
            print(f"Len of chosen dataset is: {len(chosen_dataset)}")
            print(f"Test loss in iteration {len(test_losses) + 1} is: {test_loss}")
        test_losses.append(test_loss.item())

        next_indices = get_next_idx(
            not_chosen_indices, 
            model, 
            train_dataset,
            label_batch_size,
            sampling_type
        )
        for next_idx in next_indices:
            chosen_indices.add(next_idx)
            if next_idx in not_chosen_indices:
                not_chosen_indices.remove(next_idx)
            else:
                print(f"Warning: {next_idx} not in {not_chosen_indices}")
        
            feature, target = train_dataset[next_idx]
            chosen_dataset.add(feature.tolist(), target.item())

        correlation = get_batch_correlation(train_dataset, next_indices)
        correlations.append(correlation)

    return test_losses, correlations

def main():
    start_prop = 0.2
    end_prop = 0.5
    sampling_types = [
        (RANDOM_SAMPLING, "random"), 
        (UNCERTAINTY_SAMPLING, "uncertainty"),
        (COVDROP, "covdrop")
    ]
    num_simulations = 5

    dataset_configs = [
        {
            "dataset_folder_name": "logd74",
            "features_path": f"{LOGD74_DIR}/fingerprints.npy",
            "target_path": f"{LOGD74_DIR}/logd74s.npy",
            "epochs": 100,
            "batch_size": 32,
            "label_batch_sizes": [16, 32],
            "model_class": MLP_Lipo,
            "optimizer_class": Adam,
            "lr": 0.0001
        },
        {
            "dataset_folder_name": "abalone_age",
            "features_path": f"{ABALONE_DIR}/X.npy",
            "target_path": f"{ABALONE_DIR}/y.npy",
            "epochs": 100,
            "batch_size": 64,
            "label_batch_sizes": [16, 32],
            "model_class": MLP_Abalone,
            "optimizer_class": Adam,
            "lr": 0.01
        },
        {
            "dataset_folder_name": "inhibition_data",
            "features_path": f"{INHIBITION_DIR}/X.npy",
            "target_path": f"{INHIBITION_DIR}/y.npy",
            "epochs": 1000,
            "batch_size": 32,
            "label_batch_sizes": [4, 8],
            "model_class": MLP_Inhibition,
            "optimizer_class": Adam,
            "lr": 0.005
        }
    ]

    for dataset_config in dataset_configs:
        dfn = dataset_config["dataset_folder_name"]
        for sampling_type, sn in sampling_types:
            label_batch_sizes = dataset_config["label_batch_sizes"]
            
            for label_batch_size in label_batch_sizes:
                print(f"Starting sims for {dfn}, {sn}, lbs: {label_batch_size}")
                test_losses_all = []
                correlations_all = []
                for simulation in range(num_simulations):
                        print(f"Start simulation: {simulation}")
                        train_dataset, test_dataset = get_datasets(
                            dataset_config["features_path"],
                            dataset_config["target_path"]
                        )

                        test_losses, correlations = active_learn(
                            start_prop,
                            end_prop,
                            train_dataset,
                            test_dataset,
                            dataset_config["model_class"],
                            dataset_config["optimizer_class"],
                            dataset_config["lr"],
                            dataset_config["batch_size"],
                            dataset_config["epochs"],
                            label_batch_size,
                            sampling_type
                        )
                        print(test_losses)
                        test_losses_all.append(test_losses)
                        correlations_all.append(correlations)

                results_losses = np.array(test_losses_all, dtype=float)
                results_corrs = np.array(correlations_all, dtype=float)
                
                results_losses_path = f"{dfn}_nn_{sn}_{label_batch_size}.npy"
                results_corrs_path = f"{dfn}_nn_{sn}_{label_batch_size}_cor.npy"
                
                np.save(results_losses_path, results_losses)
                np.save(results_corrs_path, results_corrs)

if __name__=="__main__":
    main()