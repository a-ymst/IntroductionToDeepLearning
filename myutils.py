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


def show_misrecognizd_images(model, data_loader, img_shape, class_names, device):
    count = 0

    # Display 50 images with "correct label : prediction"
    plt.figure(figsize=(15, 20))
    for x, t in data_loader:
        # obtain predicted labels
        x = x.to(device)            # send the data to the GPU/CPU
        t = t.to(device)            # send the data to the GPU/CPU

        y = model(x)
        pred = torch.max(y, 1)[1]

        for i in np.arange(x.shape[0]):
            
            if (pred[i] != t[i]):
                ax = plt.subplot(10, 10, count+1)
                img = x[i].cpu().numpy().copy()       # Tensor to NumPy
                img = img.reshape(img_shape)
                if(len(img_shape) == 3):
                    img = np.transpose(img, (1, 2, 0)) # Change axis order (channel, row, column) -> (row, column, channel)
                img = (img + 1)/2   # Revert the range of values ​​from [-1,1] to [0,1]

                # show result
                plt.imshow(img, cmap='gray')
                ax.set_title(f'{class_names[t[i]]}:{class_names[pred[i]]}')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                count += 1
                if count == 100:
                    break
        else:
            continue
        break
        
    plt.show()


def show_image(img, imgtype='np', cmap='gray', figsize=(10,6), vmin=None, vmax=None):
    if(imgtype == "tensor"):
        img = img.numpy()

    if(imgtype == "np" or imgtype == "tensor"):
        if(len(img.shape) == 3):
            img = np.transpose(img, (1, 2, 0)) # Change axis order (channel, row, column) -> (row, column, channel)

    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.show()

def show_kernels(model, kernel_key, figsize=(20,1), subplotsize=16, img_max_num=48):
    kernels = np.array(model.state_dict()[kernel_key].cpu())
    kernel_num = kernels.shape[0]
    channel_num = kernels.shape[1]

    img_index = 0
    for k in range(kernel_num):
        for c in range(channel_num):
            if(img_index % subplotsize == 0):
                plt.show()
                plt.figure(figsize=figsize)

            plt.subplot(1, subplotsize, img_index % subplotsize + 1)
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            plt.title(f"k,c:{k},{c}")
            plt.imshow(kernels[k][c], cmap="gray")
            img_index += 1
            if(img_index >= img_max_num):
                break
        else:
            continue
        break


import torchvision.transforms as transforms     # Transforms

def feature_to_img(feature, nrow=4, img_width=1000):
    feature = feature.unsqueeze(1)  # (N, H, W) -> (N, C, H, W)
    img = torchvision.utils.make_grid(feature.cpu(), nrow=nrow, normalize=True, pad_value=1)     # make images and arrange in a grid
    img = transforms.functional.to_pil_image(img)    # tensor to  PIL Image
    img_height = int(img_width * img.height / img.width)  # resize
    img = img.resize((img_width, img_height))
    return img


def show_features(features, nrow=10):
    for name, x in features.items():
        img = feature_to_img(feature=x, nrow=nrow)
        print(name, x.shape)
        display(img)


def save_checkpoint(model, optimizer, history, filepath):
    checkpoint = {
        'model_state_dict': model.to('cpu').state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
        }
    print("saving checkpoint '{}'".format(filepath))
    torch.save(checkpoint, filepath)


# Note: Input model & optimizer should be pre-defined.  This function only updates their states.
import os
def load_checkpoint(model, optimizer, filepath):

    if os.path.isfile(filepath):
        print("loading checkpoint '{}'".format(filepath))
    else:
        print("File not found: '{}'".format(filepath))
        return

    start_epoch = 0
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    history = checkpoint['history']

    print(checkpoint)

    return model, optimizer, history
