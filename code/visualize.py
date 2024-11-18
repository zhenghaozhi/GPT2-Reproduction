import matplotlib.pyplot as plt
import os

def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    train_steps = [i * train_interval for i in range(len(train_loss_list))]
    val_steps = [i * val_interval for i in range(len(val_loss_list))]

    plt.figure(figsize=(10, 6))
    
    plt.plot(train_steps, train_loss_list, label='Training Loss', color='blue', marker='o')
    plt.plot(val_steps, val_loss_list, label='Validation Loss', color='red', marker='x')
    
    plt.title(f'Training and Validation Loss for {dataset}')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()
