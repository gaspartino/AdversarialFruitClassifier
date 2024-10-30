import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchattacks
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

#==================================  Defining the dataset  ==================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = '/images'

# Definir transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize(256),                
    transforms.CenterCrop(224),            
    transforms.ToTensor(),                 
    transforms.Normalize(                  
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class DatasetPersonalizado(Dataset):
    def __init__(self, base_dir=None, transform=None, images=None, labels=None):
        self.base_dir = base_dir
        self.transform = transform
        self.class_names = []
        self.images = images if images is not None else [] 
        self.labels = labels if labels is not None else [] 

        if base_dir:
            self.load_data_from_directory(base_dir)

    def load_data_from_directory(self, base_dir):
        self.class_names = os.listdir(base_dir)
      
        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(base_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.labels.append(idx)  # Usar índice como rótulo

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_unique_labels(self):
        return set(self.labels)

dataset = DatasetPersonalizado(base_dir, transform=transform)

labels = list(set(label for _, label in dataset))

# Dividir o conjunto de dados em treino e test inicialmente
train_dataset, test_dataset = train_test_split(dataset,  test_size=0.1, random_state=42)

# Criar DataLoader para iterar sobre os dados de forma eficiente
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = dataset.class_names

#==================================  Model training  ==================================

# Função para calcular métricas de avaliação
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return test_loss/len(test_loader), accuracy, precision, recall, f1

model = models.resnet34(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(labels))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_accuracy = 0
best_model = None


# Treinamento do modelo
num_epochs = 23
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Época (Treino) [{epoch+1}/{num_epochs}], Perda: {running_loss/len(train_loader):.4f}, Acurácia: {accuracy:.4f}')

print('Treinamento concluído!')

#==================================  Model evaluation  ==================================
model.eval()

print('\n\nTeste do modelo:')
test_loss, accuracy, precision, recall, f1 = evaluate_model(model, test_loader, criterion)
print(f'Perda (teste): {test_loss:.4f}, Acurácia: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

print('Teste concluído!')

#==================================  Attacking model  ==================================

epsilon = 0.007  # Perturbação máxima permitida para o FGSM
fgsm = torchattacks.FGSM(model, eps=epsilon)

correct = 0
total = 0

images_list = []
adversarial_images_list = []
true_labels_list = []
predicted_labels_list = []

# Gerar imagens adversariais
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    adversarial_batch = fgsm(images, labels)

    outputs = model(adversarial_batch)
    _, predicted = torch.max(outputs.data, 1)

    images_list.append(images.cpu())
    adversarial_images_list.append(adversarial_batch.cpu())
    true_labels_list.append(labels.cpu())
    predicted_labels_list.append(predicted.cpu())

images_list = torch.cat(images_list)
adversarial_images_list = torch.cat(adversarial_images_list)
true_labels_list = torch.cat(true_labels_list)
predicted_labels_list = torch.cat(predicted_labels_list)

# Calcular a acurácia
def calculate_accuracy(true_labels, predicted_labels):
    correct = (true_labels == predicted_labels).sum().item()
    accuracy = correct / true_labels.size(0) * 100
    return accuracy

adversarial_accuracy = calculate_accuracy(true_labels_list, predicted_labels_list)
print(f'Acurácia nos exemplos adversariais: {adversarial_accuracy:.2f}%')


# Função para desnormalizar a imagem
def denormalize(tensor):
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)  # Garantir que os valores estão no intervalo [0, 1]
    return tensor

def show_attack_results(imgs_to_show, model, images, adversarial_images, labels, predicted_labels, class_names):
    for i in range(imgs_to_show):
        original_image = denormalize(images[i])
        adversarial_image = denormalize(adversarial_images[i])
        true_label = labels[i].item()
        predicted_label = predicted_labels[i].item()

        true_label_name = class_names[labels[i].item()]
        predicted_label_name = class_names[predicted_labels[i].item()]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(original_image)
        axs[0].set_title(f'Original\nTrue Label: {true_label_name}')
        axs[0].axis('off')

        axs[1].imshow(adversarial_image)
        axs[1].set_title(f'Adversarial\nPredicted: {predicted_label_name}')
        axs[1].axis('off')

        plt.show()

show_attack_results(6, model, images_list, adversarial_images_list, true_labels_list, predicted_labels_list, dataset.class_names)
