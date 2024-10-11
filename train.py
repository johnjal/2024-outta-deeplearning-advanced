from read_dataset import *
from preprop import *
from model import segmentation_model
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_loss(losses, accuracy, batch_size):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train Loss')
    plt.plot(accuracy, label='Train Accuracy')
    plt.xlabel(f'Epoch (x{batch_size})')
    plt.ylabel('Loss and Accuracy')
    plt.title('Training Loss/Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    
def accuracy(predictions, truth):
    predicted_labels = torch.argmax(predictions, axis=1)
    correct = (predicted_labels == truth).float()
    accuracy = correct.mean().item()
    return accuracy



def train(img_list, gt_list, model, epoch, learning_rate, optimizer, criterion, data_len, batch_size=1, fp=0.8, sp=0.5): ## batch_size=1 일때 best. outlier 때문인듯..?
    running_loss = 0.0
    optimizer.zero_grad()
    model.train()
    device = next(model.parameters()).device
    losses = []
    accuracy_list = []


    for i in range(epoch):
        for batch_cnt in range(data_len // batch_size):
            # [1] 빈칸을 작성하시오.
            # 학습 과정
            images = []
            labels = []
            for iter in range(batch_size):
                index = iter + batch_cnt * batch_size
                img, label = get_data(img_list[index], gt_list[index])
                images.append(img)
                labels.append(torch.from_numpy(label.copy()).long())

            images = torch.cat(images.copy(), dim=0).to(device)
            labels = torch.cat(labels.copy(), dim=0).to(device)

            optimizer.zero_grad()

            predictions = model(images)

            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())
            accuracy_list.append(accuracy(predictions, labels))
            

            if batch_cnt == (2000 // batch_size) and i == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= fp
            if batch_cnt == (5000 // batch_size) and i == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= sp
                

            if (batch_cnt % (100 // batch_size) == 0) and (batch_cnt != 0):
                plot_loss(losses, accuracy_list, batch_size)
                print(f'Iteration: {batch_cnt*batch_size + data_len*i}, Loss: {losses[-1]}, Accuracy: {accuracy_list[-1]}')
        torch.save(model.state_dict(), f'model_state_dict{i}.pth')