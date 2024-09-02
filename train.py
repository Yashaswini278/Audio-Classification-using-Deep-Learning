import argparse
import numpy as np
from dataset import UrbanSoundDataset
from model import CNN 
from plotting import * 
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader
import torchaudio
from nnAudio import features 
import librosa
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm 
import os

def train_model(model, train_dl, val_dl, epochs, device, print_loss=False):
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=epochs,
                                                anneal_strategy='linear')
    history=[]

    for epoch in tqdm(range(epochs)):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss +=  loss.item()
            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        train_acc = correct_prediction/total_prediction

        # Validation phase
        correct_pred = 0
        total_pred  = 0
        model.eval()
        # Disable gradient updates
        with torch.no_grad():
          for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            # Get predictions
            outputs = model(inputs)
            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_pred += (prediction == labels).sum().item()
            total_pred += prediction.shape[0]
          
          val_acc = correct_pred/total_pred

        result = {}
        result['avg_train_loss'] = avg_loss
        result['train_acc'] = train_acc
        result['val_acc'] = val_acc

        history.append(result)
        
        if (print_loss): 
          print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Train Acc: {train_acc:.2f}, Val_acc: {val_acc:.2f}')
        
        if(val_acc > 0.9):
          break
    
    print('Finished Training')

    return history

def test_model(model, test_dl, device):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in test_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
  return acc

def main():
    parser = argparse.ArgumentParser(description='Urban Sound Classification')
    parser.add_argument('--annotations', type=str, required=True, help='Path to UrbanSound8K.csv')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--trans_type', type=str, choices=['mel_spec', 'gamma'], required=True, 
                        help='Transformation type: "mel_spec" for Mel spectrogram or "gamma" for Gammatonegram')
    args = parser.parse_args()

    SAMPLE_RATE = 44100
    NUM_SAMPLES = 44100*4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mels=64)

    gammatonegram = features.gammatone.Gammatonegram(
        sr = SAMPLE_RATE, 
        n_fft = 1024,
        hop_length = 512, 
        n_bins = 64)

    if args.trans_type == "mel_spec":
        transformation = mel_spectrogram
    else:
        transformation = gammatonegram

    usd = UrbanSoundDataset(args.annotations, args.data, args.trans_type, transformation, SAMPLE_RATE, NUM_SAMPLES, device)

    # Random split of 80:20 between training and testing
    num_items = len(usd)
    num_train = round(num_items * 0.8)
    num_test = num_items - num_train
    train, test_ds = random_split(usd, [num_train, num_test])
    # Random split of 80:20 between training and validation
    num_trainds = round(num_train*0.8)
    num_valds = num_train - num_trainds
    train_ds, val_ds = random_split(train, [num_trainds, num_valds])

    # Create training, validation and test data loaders
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = CNN().to(device)

    history = train_model(model, train_dl, val_dl, args.epochs, device)
    test_acc = test_model(model, test_dl, device)

    model_type = f"CNN_{args.trans_type}"
    plot_train_test_loss(history, test_acc, model_type)
    plot_confusion_matrix(model, test_dl, model_type)

if __name__ == "__main__":
    main()