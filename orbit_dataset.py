from ripser import Rips
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class Orbit5kDataset(Dataset):

    def __init__(self, X, y, X_orig, transform=None):
        self.X = X
        self.y = y
        self.X_orig = X_orig

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.X_orig[idx]

def generate_orbit(point_0, r, n=300):

    X = np.zeros([n, 2])

    xcur, ycur = point_0[0], point_0[1]

    for idx in range(n):
        xcur = (xcur + r * ycur * (1. - ycur)) % 1
        ycur = (ycur + r * xcur * (1. - xcur)) % 1
        X[idx, :] = [xcur, ycur]

    return X

def generate_orbits(m, rs=[2.5, 3.5, 4.0, 4.1, 4.3], n=300, random_state=None):

    # m orbits, each of n points of dimension 2
    orbits = np.zeros((m * len(rs), n, 2))

    # for each r
    for j, r in enumerate(rs):

        # initial points
        points_0 = random_state.uniform(size=(m,2))

        for i, point_0 in enumerate(points_0):
            orbits[j*m + i] = generate_orbit(points_0[i], rs[j], n)

    return orbits

def conv_pd(diagrams):
    pd = np.zeros((0, 3))

    for k, diagram_k in enumerate(diagrams):
        if k != 0:
            diagram_k= diagram_k[~np.isinf(diagram_k).any(axis=1)] # filter infs
            diagram_k = np.concatenate((diagram_k, k * np.ones((diagram_k.shape[0], 1))), axis=1)
            if len(diagram_k) > 0:
                pd = np.concatenate((pd, diagram_k))

    return pd

def collate_fn(data):

    tmp_pd, _, orbit = data[0]

    n_orbit = orbit.shape[0]

    n_batch = len(data)
    n_features_pd = tmp_pd.shape[1]
    n_points_pd = max(len(pd) for pd, _, _ in data)
    inputs_pd = np.zeros((n_batch, n_points_pd, n_features_pd), dtype=float)

    orbits = np.zeros((n_batch, n_orbit, 2), dtype=float)

    mask = np.zeros((n_batch, n_points_pd))
    labels = np.zeros(len(data))

    for i, (pd, label, orbit) in enumerate(data):
        inputs_pd[i][:len(pd)] = pd
        labels[i] = label
        mask[i][:len(pd)] = 1
        orbits[i][:n_orbit] = orbit
    return torch.Tensor(inputs_pd), torch.Tensor(mask).long(), torch.Tensor(labels).long(), torch.Tensor(orbits)

def get_datasets(dataset_size, batch_size, n_train, n_test, n_points=300):
    vr = Rips()
    random_state = np.random.RandomState(42)
    X_orbit5k = generate_orbits(dataset_size, n=n_points, random_state=random_state)

    X_big = []

    y = np.zeros(dataset_size)

    for i in range(1, 5):
        y = np.concatenate((y, np.ones(dataset_size) * i))

    for x in tqdm(X_orbit5k):
        diagram = conv_pd(vr.fit_transform(x))
        diagram[:, 1] -= diagram[:, 0] # to predict delta instead of second coord > first coord
        X_big.append(diagram)

    n_max = 0

    for x in X_big:
        n_max = max(n_max, len(x))

    dataset = Orbit5kDataset(X_big, y, X_orbit5k)

    dataset_train, dataset_test = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(54))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_test =  DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataset_train, dataset_test, dataloader_train, dataloader_test, n_max