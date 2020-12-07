import torch
import numpy as np

import tqdm
import math

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from custom_datasets.reconstruction_dataset import FaceReconstructionDataset

from utils import init_logger


EPOCHS = 40
LEARNING_RATE = 10.0 ** -4
BATCH_SIZE = 64


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


# our class must extend nn.Module
class binaryClassification(nn.Module):
    def __init__(self, input_dim=64*64*1):
        super(binaryClassification, self).__init__()

        self.layer_1 = nn.Linear(
            input_dim,
            64)
        # self.layer_2 = nn.Linear(1024, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(64)
        # self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        # x = self.relu(self.layer_2(x))
        # x = self.batchnorm2(x)
        x = self.layer_out(x)
        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def get_model_scores_and_labels_for_data_set(model, dataset_loader, device):
    y_true = []
    y_scores_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in dataset_loader:
            input = build_input_from_batch(X_batch, device)
            y_pred = model(input)
            y_pred_probs = torch.sigmoid(y_pred)
            y_scores_list.append(y_pred_probs.cpu().numpy())
            true_labels_tensor = build_labels_from_batch(X_batch)
            true_labels_np_array = true_labels_tensor.numpy()
            y_true.append(true_labels_np_array)
    return np.concatenate(y_scores_list), np.concatenate(y_true)


def build_input_from_batch(batch, device):
    diff = batch['reconstruction diff']
    diff = torch.sum(diff, -1) # sum through color channels
    diff = diff.to(device)
    N, H, W = diff.shape
    return diff.reshape(N, 1 * H * W)


def build_labels_from_batch(batch):
    y_batch = batch['label']
    y_batch.cuda()
    return y_batch


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 10.0 ** -5:
        lrate = 10.5 ** -5
    return lrate


def train_models(logger, recalc=True):
    logger.info("START NEW EVAL")
    logger.info("***" * 20)
    logger.info(f"recalc={recalc}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("loading train set, might take time due to heavy file read / gen...")
    train_dataset = FaceReconstructionDataset(
        '/mnt/data/deepfakes/context_encoder_dataset/train/',
        reconstruct_again=True)
    logger.info(f"Train set has: {len(train_dataset)} samples")
    logger.info(f"Train set has: {train_dataset.num_of_pristine_samples} "
          f"pristine samples")
    logger.info(
        f"Train set has: {train_dataset.num_of_fake_samples} fake samples")

    print("loading test set, might take time cause computing vgg features...")
    test_dataset = FaceReconstructionDataset(
        '/mnt/data/deepfakes/context_encoder_dataset/test/',
        reconstruct_again=True)
    logger.info(f"Test set has: {len(test_dataset)} samples")
    logger.info(f"Test set has: {test_dataset.num_of_pristine_samples} "
          f"pristine samples")
    logger.info(f"Test set has: {test_dataset.num_of_fake_samples} fake samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=0)
    model = binaryClassification()
    print(f"number of params in model = {get_n_params(model)}")
    logger.info(f"number of params in model = {get_n_params(model)}")
    model.to(device)
    print("The classification model.. ")
    print(model)
    logger.info(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=step_decay)

    print(f"Start training it...")
    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for itr, x_batch in enumerate(train_loader):
            y_batch = build_labels_from_batch(x_batch)
            input = build_input_from_batch(x_batch, device)

            optimizer.zero_grad()

            y_pred = model(input)

            y_gt = y_batch.unsqueeze(1)
            y_gt = y_gt.type_as(y_pred)
            y_gt = y_gt.cuda()

            loss = criterion(y_pred, y_gt)
            acc = binary_acc(y_pred, y_gt)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} '
              f'| Acc: {epoch_acc / len(train_loader):.3f}')
        logger.info(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} '
              f'| Acc: {epoch_acc / len(train_loader):.3f}')

        if e % 5 == 0 or True:
            if e % 10 == 0:
                print(f"calculating auc for train set...")
                y_scores_list, y_true = get_model_scores_and_labels_for_data_set(
                    model, train_loader, device)
                auc = roc_auc_score(y_true, y_scores_list)
                print(f"Epoch {e + 0:003} | train AuC is {auc:.5f}")
                logger.info(f"Epoch {e + 0:003} | train AuC is {auc:.5f}")

            print(f"calculating auc for test set...")
            y_scores_list, y_true = get_model_scores_and_labels_for_data_set(
                model, test_dataloader, device)
            auc = roc_auc_score(y_true, y_scores_list)
            print(f"Epoch {e + 0:003} | test AuC is {auc:.5f}")
            logger.info(f"Epoch {e + 0:003} | test AuC is {auc:.5f}")
        # scheduler.step()
    logger.info("===" * 20)


def main():
    logger = init_logger('model_train')
    train_models(logger, recalc=False)


if __name__ == "__main__":
    main()
