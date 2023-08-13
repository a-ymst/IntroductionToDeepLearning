import torch                                        # PyTorch
import torchvision                                  # PyTorch library for image processing
import numpy as np                                  # for scientific computing (e.g. culclations with array)
import matplotlib.pyplot as plt                     # for visualization

# h should be a list like [epoch, loss_train, acc_train, loss_test, acc_test]
def show_learning_curve(h):
    # Draw the Loss curve 
    plt.plot(h[:,0], h[:,1], c="blue", label="train")   # Loss on the training set
    plt.plot(h[:,0], h[:,3], c="orange", label="test")  # Loss on the test set

    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    # Draw the Accuracu curve 
    plt.plot(h[:,0], h[:,2], c="blue", label="train")        # Accuracy on the training set
    plt.plot(h[:,0], h[:,4], c="orange", label="test")      # Accuracy on the test set

    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_confusion_matrix(model, data_loader, class_names, device):

    pred_list = []
    true_list = []

    # iterate over test data
    for x, t in data_loader:
        x = x.to(device)                # send the data to the GPU/CPU
        t = t.to(device)                # send the data to the GPU/CPU

        y = model(x)                    # prediction(forward calculation)
        pred = torch.max(y, 1)[1]       # predected label
        
        pred = pred.data.cpu().numpy()  # send to the CPU and convert to numpy array
        true = t.data.cpu().numpy()     # send to the CPU and convert to numpy array

        pred_list.extend(pred)             # Save Predicted label
        true_list.extend(true)             # Save True label


    cm = confusion_matrix(y_pred=pred_list, y_true=true_list)
    cmp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    cmp.plot(cmap=plt.cm.Blues)
