import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch 
import numpy as np

def plot_train_test_loss(history, test_acc, model_type): 
    plt.figure(figsize=(24,12))
    plt.subplot(2, 1, 1)
    plt.plot([history[ind]['avg_train_loss'] for ind in range(len(history))])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.subplot(2, 1, 2)
    plt.plot([history[ind]['train_acc'] for ind in range(len(history))], label='train acc')
    plt.plot([history[ind]['val_acc'] for ind in range(len(history))], label = 'val_acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.suptitle(f"Final Train Accuracy = {history[len(history)-1]['train_acc']:.2f}, Val Accuracy = {history[len(history)-1]['val_acc']:.2f}, Test Accuracy = {test_acc:.2f}", fontsize=20)
    plt.legend()
    plt.savefig(f"results/train_test_loss_{model_type}.png")

y_pred = []
y_true = []
  

def plot_confusion_matrix(model, test_dl, model_type):
    y_true = []
    y_pred = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
          y_pred.extend(prediction.cpu())
          y_true.extend(labels.cpu())
        
    
    # constant for classes
    classes = ('air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'enginge_idling',
            'gun_shot', 'jackhammer', 'siren', 'street_music')
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"results/confusion_matrix_{model_type}.png")