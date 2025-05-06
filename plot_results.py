import matplotlib.pyplot as plt
import json

def plot_results(log_file):
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    epochs = range(1, len(logs['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, logs['train_loss'], label='Train Loss')
    plt.plot(epochs, logs['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, logs['train_acc'], label='Train Accuracy')
    plt.plot(epochs, logs['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_results("ml/training_logs.json")
