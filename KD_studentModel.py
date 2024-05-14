import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision.models as models
import pickle

# check if cude is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# folders path containing images from dataset
train_dir = '/scratch/hsk8171/deepLearning/archive/Training'
test_dir = '/scratch/hsk8171/deepLearning/archive/Testing'

# Transformation applied on training and test dataset
train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])


# from the specified folder load the dataset images using pytorch ImageFolder API
test_dataset = ImageFolder(test_dir,transform=test_transforms)
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False)


train_dataset = ImageFolder(train_dir,transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
class_names = train_dataset.classes

# print the number of categories present in datatset
print(class_names)



# Load the pretrained model from torchvision.models and freeze the CNN parameters.
vgg = models.vgg16(pretrained=True)

# Freeze the parameters of the pretrained model
for param in vgg.parameters():
    param.requires_grad = False

# Modify the VGG model architecture to remove the last linear layers
vgg.features = nn.Sequential(*list(vgg.features.children())[:30])  # Keep only the first 30 layers
vgg.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Replace the original avgpool layer
features = vgg.features(torch.randn(1, 3, 128, 128))  # Get the output feature map size
features = vgg.avgpool(features)  # Apply the modified avgpool layer
num_features = features.size(1) * features.size(2) * features.size(3)  # Calculate the correct num_features
vgg.classifier = nn.Linear(num_features, 4)  # Replace the classifier with a single linear layer for 4 classes

# Move the model to GPU if available
model = vgg.to(device)


#Load the pre-trained teacher model
model.load_state_dict(torch.load('best_vgg.pth'))
print("Model loaded from checkpoint.")
summary(model, input_size=(3, 128, 128))  # Print teacher model summary



# Define Student model
class StudentBrainTumorModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*32*hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# initialise the student Model and print student Model summary and data
student_model = StudentBrainTumorModel(3, 3, len(train_dataset.classes)).to(device)
summary(student_model, input_size=(3, 128, 128))



# Define accuracy function 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
test_loss, test_acc = 0, 0
model.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        test_pred = model(X)

        test_loss += loss_fn(test_pred, y)
        test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

    # Calculate the test loss average per batch
    test_loss /= len(test_dataloader)

    # Calculate the test accuracy average per batch
    test_acc /= len(test_dataloader)

print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%\n")


# structures for storing the train loss, train accuracy, test loss and test accuracies
train_losses = [0]
train_accuracies = [0]
test_losses = [0]
test_accuracies = [0]

# define KD loss for Teacher- Student Model distillation.
def distillation_loss(student_output, teacher_output, labels, temperature=1.0, alpha=0.5):
    cross_entropy_loss = nn.CrossEntropyLoss()(student_output, labels)
    kd_loss = nn.KLDivLoss()(F.log_softmax(student_output / temperature, dim=1),
                             F.softmax(teacher_output / temperature, dim=1)) * (temperature ** 2)
    return alpha * cross_entropy_loss + (1 - alpha) * kd_loss

from tqdm.auto import tqdm

# Run the training loop , after every epoch, check the Test loss and Test accuracy
epochs = 50
best_accuracy = 0.0 # store the model is test accuracy is than past best accuracies

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_loss, train_acc = 0, 0
    student_model.train()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):

        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = student_model(X)
        model.eval()
        teacher_pred = model(X)
        # 2. Calculate loss and accuracy (per batch)
        loss = distillation_loss(y_pred, teacher_pred, y) #loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Divide total train loss and acc by lenght of train dataloader
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

    train_losses.append(float(train_loss))
    train_accuracies.append(float(train_acc))

    test_loss, test_acc = 0, 0
    student_model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = student_model(X)
            model.eval()
            teacher_pred = model(X)
            test_loss += distillation_loss(test_pred, teacher_pred, y) #loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test accuracy average per batch
        test_acc /= len(test_dataloader)

    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%\n")
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(student_model.state_dict(), 'best_student_model.pth')
    test_losses.append(float(test_loss))
    test_accuracies.append(float(test_acc))


# create the plots for train loss, train accuracy
# create the plots for test loss, test accuracy

plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(epochs+1), train_losses, label='Train')
plt.plot(range(epochs+1), test_losses, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(epochs+1), train_accuracies, label='Train')
plt.plot(range(epochs+1), test_accuracies, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

plt.savefig('student_model_loss_and_accuracy_curves.png')

# Save the loss and accuracy lists to a file
with open("loss_and_accuracy_data.pkl", "wb") as f:
    pickle.dump(
        {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
        },
        f,
    )

# define function for evaluating the trained model of test datatset.
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    student_model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)
            # Forward pass (model outputs raw logits)
            pred_logit = student_model(sample)

            # Get predicition probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob.cpu())

        # Stack the pred_probs to turn list into a tensor
        return torch.stack(pred_probs)

import random

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_dataset), k=12):
    test_samples.append(sample)
    test_labels.append(label)

# Make predictions
pred_probs = make_predictions(model=student_model,
                               data=test_samples)

# Convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)
# Plot predictions
plt.figure(figsize=(12,9))
nrows = 3
ncols = 4

# Plot predictions for some random images from the test dataset.
for i, sample in enumerate(test_samples):
    # Create subplot
    plt.subplot(nrows, ncols, i+1)
    # Plot the target image
    plt.imshow(sample.permute(1, 2, 0))

    # Find the prediction (in text form e.g "Sandal")
    pred_label = class_names[pred_classes[i]]

    # Get the truth label (in text form)
    truth_label = class_names[test_labels[i]]

    # Create a title for the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Check for equality between pred and truth and change color of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=8, c="g")
    else:
        plt.title(title_text, fontsize=8, c="r")

    plt.axis(False)
    plt.savefig(f'predicted_images_student_model{i}.png')


