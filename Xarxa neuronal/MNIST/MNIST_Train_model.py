import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ---------- Definició del model ----------

# Definim la classe del model de xarxa neuronal convolucional (CNN).
# Utilitzarem aquest model per classificar els dígits del dataset MNIST (0-9).

class ClassificadorCNN(nn.Module):
    def __init__(self, n_classes=10): # n_classes = 10 perquè MNIST té 10 classes (dígits del 0 al 9)
        super().__init__()

        # Bloc convolucional 1: extreu característiques bàsiques (vores, textures)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 1 canal d'entrada (escala de grisos) -> 32 filtres
            nn.MaxPool2d(kernel_size=2), # Reducció espacial: 28x28 -> 14x14
            nn.ReLU()                  # Funció d'activació ReLU
        )

        # Bloc convolucional 2: extreu característiques més complexes
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 32 canals d'entrada -> 64 filtres
            nn.MaxPool2d(kernel_size=2), # Reducció espacial: 14x14 -> 7x7
            nn.ReLU()                  # Funció d'activació ReLU

        )

        # Capes totalment connectades (classificador final)
        self.fc = nn.Sequential(
            nn.Flatten(),               # Aplanem el tensor: 64 × 7 × 7 = 3136 valors
            nn.Linear(64 * 7 * 7, 128), # Capa oculta totalment connectada
            nn.ReLU(),                  # Funció d'activació ReLU
            nn.Linear(128, n_classes)   # Capa de sortida: 128 -> 10 classes
        )

    # Definim el pas endavant de la xarxa (primera predicció abans de calcular l'error)
    def forward(self, x):
        x = self.conv1(x) # Primer bloc convolucional
        x = self.conv2(x) # Segon bloc convolucional
        return self.fc(x) # Classificador final


# ---------- Funció de pèrdua, optimitzador i càrrega de dades ----------

if __name__ == '__main__':

    # Definim una funció que substitueixi a la funció print per guardar els missatges de log en un fitxer
    def log_print(message):
        with open('Xarxa neuronal/MNIST/entrenament_log.txt', 'a') as f:
            f.write(f'{message}\n')
        print(f'{message}')


    # ---------- Hiperparàmetres ----------

    epochs = 20          # Nombre d'èpoques
    lr = 1e-4            # Taxa d'aprenentatge
    weight_decay = 1e-5  # Decaïment de pesos
    batch_size = 64      # Mida del batch

    # ---------- Càrrega del dataset MNIST ----------

    # Definim la transformació de les imatges: les convertim a tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Descarreguem el dataset MNIST (si no existeix, el descarrega automàticament)
    dataset_train_complet = datasets.MNIST(root='Xarxa neuronal/MNIST/dades', train=True,  download=True, transform=transform)
    dataset_test           = datasets.MNIST(root='Xarxa neuronal/MNIST/dades', train=False, download=True, transform=transform)

    # Dividim el dataset d'entrenament en entrenament (80%) i validació (20%)
    n_train = int(len(dataset_train_complet) * 0.8)
    n_val   = len(dataset_train_complet) - n_train
    dataset_train, dataset_val = random_split(dataset_train_complet, [n_train, n_val])

    # Creem els DataLoaders per carregar les dades en batches durant l'entrenament
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(dataset_val,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False, num_workers=2)

    # ---------- Inicialització del model ----------

    # Seleccionem el dispositiu: GPU si està disponible, sinó CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ClassificadorCNN().to(device) # Creem una instància del model i la movem al dispositiu

    # Utilitzem CrossEntropyLoss perquè és un problema de classificació multiclasse (10 classes)
    criteri    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path = 'Xarxa neuronal/MNIST/Pesos_mnist.pth' # Ruta per desar el punt de control
    start_epoch = 1  # Època inicial

    # Si existeix un punt de control, carreguem l'estat del model i de l'optimitzador
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizador.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        log_print(f'Resumint l\'entrenament des de l\'època {start_epoch}\nTaxa d\'aprenentatge: {lr}\nDecaïment de pesos: {weight_decay}')
    except FileNotFoundError:
        log_print(f'Començant entrenament de la xarxa CNN.\nDataset: MNIST\nFiltres conv1: 32 | Filtres conv2: 64\nCapa oculta FC: 128 neurones\nClasses de sortida: 10\nTaxa d\'aprenentatge: {lr}\nDecaïment de pesos: {weight_decay}\nFunció de pèrdua: CrossEntropyLoss\nOptimitzador: Adam\nÈpoques: {epochs}')
        pass


    best_val = float('inf')
    patience, wait = 10, 0  # Aturem l'entrenament si no millora durant 10 èpoques consecutives

    # ---------- Entrenament ----------
    # Entrenem el model utilitzant les dades d'entrenament i actualitzem els pesos
    # amb l'optimitzador definit. Després de cada època, validem el model.

    for epoch in range(start_epoch, start_epoch + epochs): # Per cada època
        model.train()
        for xb, yb in train_loader: # Per cada batch de dades d'entrenament
            xb, yb = xb.to(device), yb.to(device) # Movem les dades al dispositiu
            optimizador.zero_grad()           # Resetejem els gradients de l'optimitzador
            preds = model(xb)                 # Fem la predicció del model per les dades d'entrada
            loss  = criteri(preds, yb)        # Calculem la pèrdua entre les prediccions i les sortides reals
            loss.backward()                   # Retropropagació per calcular els gradients
            optimizador.step()                # Actualitzem els pesos del model

        # ---------- Validació ----------
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds     = model(xb)
                val_loss += criteri(preds, yb).item() * xb.size(0)
                val_correct += (preds.argmax(dim=1) == yb).sum().item()

        val_loss    /= len(val_loader.dataset) # type: ignore
        val_accuracy = round(val_correct / len(val_loader.dataset) * 100, 2) # type: ignore

        log_print(f'Època {epoch:02d} – train loss: {loss.item():.4f} – val loss: {val_loss:.4f} – val acc: {val_accuracy:.2f}%') # type: ignore

        # Early stopping: aturem si la pèrdua de validació no millora
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log_print('Early stopping activat.')
                break


     # ---------- Test ---------- 
     # (Aqui avaluem el model utilitzant les dades de prova carregades anteriorment, i calculem la pèrdua i l'exactitud del model. 
     # Aquesta part no actualitza els pesos del model, només calcula la pèrdua i l'exactitud perquè poguem veure com avança l'entrenament de la nostra IA.)

    model.eval() # Posem el model en mode avaluació
    correct = 0
    test_loss    = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds      = model(xb)
            loss       = criteri(preds, yb)
            test_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(dim=1) == yb).sum().item()

    test_loss    /= len(test_loader.dataset) # type: ignore
    accuracy = round(correct / len(test_loader.dataset) * 100, 2) # type: ignore
    log_print(f'Test Loss: {test_loss:.4f}, Percentatge d\'encert: {accuracy:.2f}%')


    # ---------- Guardem els pesos obtinguts un cop finalitzem l'entrenament ----------

    torch.save({
        'epoch': epoch,                      # Guardem l'època actual # type: ignore
        'model_state': model.state_dict(),
        'optimizer_state': optimizador.state_dict(),
        'loss': loss.item()                  # type: ignore
    }, checkpoint_path)