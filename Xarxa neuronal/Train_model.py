import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Dades   # Importem el mòdul Dades on tenim el DataLoader definit


# ---------- Definició del model ----------

# Definim la classe del model de xarxa neuronal. Aqui definirem el model que utilitzarem per fer el classificador. 
# Més endavant, utilitzarem aquesta classe per crear dos instàncies del model, una per l'entrenament i una altra per la validació.

class Classificador(nn.Module):
    def __init__(self, n_entrada = 2, n_sortida = 1, n_oculta = 32): # Configurem les neurones d'entrada, sortida i ocultes.
        super().__init__() 
        self.net = nn.Sequential( # Definim la xarxa neuronal com una seqüència de capes.
            nn.Linear(n_entrada, n_oculta),   # Entrada → Capa oculta
            nn.ReLU(),                  # Funció d'activació ReLU
            nn.Linear(n_oculta, n_oculta),   # Capa oculta → Capa oculta
            nn.ReLU(),                  # Funció d'activació ReLU
            nn.Linear(n_oculta, n_oculta),   # Capa oculta → Capa oculta
            nn.ReLU(),                  # Funció d'activació ReLU
            nn.Linear(n_oculta, n_sortida),   # Capa oculta → logit
            nn.Sigmoid()                # Converteix el logit a una probabilitat entre 0 i 1
        )

    # Ara es defineix el pas endavant de la xarxa (Aquest pas correspón a la primera predicció que fa la xarxa abans de calcular l'error)

    def forward(self, x):
        return self.net(x) 

# ---------- Funció de pèrdua i optimitzador i càrrega de dades ----------

model = Classificador()   # Creem una instància del model
# model.load_state_dict(torch.load('Pesos_classificador.pth')) # Carreguem els pesos inicials del model des d'un fitxer. Això és útil si volem continuar l'entrenament d'un model ja entrenat anteriorment. Si és la primera vegada que entrenem el model, podem comentar aquesta línia.
epochs = 200     # Nombre d'èpoques
lr = 1e-4 # Taxa d'aprenentatge
weight_decay = 1e-5 # Decaïment de pesos
criteri = nn.BCELoss()                     # Funció de pèrdua: Binary Cross Entropy Loss
optimizador = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Optimitzador: Adam       
train_loader, val_loader, test_loader = Dades.carregador_dades("Xarxa neuronal/dades.csv") # Carreguem les dades utilitzant la funció del mòdul Dades    
checkpoint_path=('Pesos_classificador.pth')  # Ruta per desar el punt de control
start_epoch=1  # Època inicial

# Si existeix un punt de control, carreguem l'estat del model i de l'optimitzador
try:
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model_state'])
    optimizador.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    print(f'Resuming from epoch {start_epoch}')
except FileNotFoundError:
    pass


best_val = float('inf')
patience, wait = 20, 0

# ---------- Entrenament ---------- (Aqui entrenem el model utilitzant les dades d'entrenament carregades anteriorment, i actualitzem els pesos del model utilitzant l'optimitzador definit.)


for epoch in range(start_epoch, start_epoch + epochs): # Per cada època
    model.train()
    for xb, yb in train_loader: # Per cada batch de dades d'entrenament (es a dir, per cada grup de dades que s'utilitza per actualitzar els pesos del model (entrades - sortida))
        optimizador.zero_grad() # Resetejem els gradients de l'optimitzador
        preds = model(xb).squeeze() # Fem la predicció del model per les dades d'entrada del batch
        loss = criteri(preds, yb.squeeze()) # Calculem la pèrdua entre les prediccions i les sortides reals del batch
        loss.backward() # Fem el retropropagació per calcular els gradients
        optimizador.step() # Actualitzem els pesos del model utilitzant l'optimitzador
     
     # optional validation
    if val_loader is not None:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).squeeze()
                val_loss += criteri(preds, yb.squeeze()).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch:02d} – train loss: {loss.item():.4f} – val loss: {val_loss:.4f}')

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping')
                break
    else:
        print(f'Epoch {epoch:02d} – loss: {loss.item():.4f}')
    

 # ---------- Test ---------- (Aqui avaluem el model utilitzant les dades de prova carregades anteriorment, i calculem la pèrdua i l'exactitud del model. Aquesta part no actualitza els pesos del model, només calcula la pèrdua i l'exactitud perquè poguem veure com avança l'entrenament de la nostra IA.)

model.eval()
correct = 0
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).squeeze()
        loss = criteri(preds, yb.squeeze())
        test_loss += loss.item() * xb.size(0)
        predicted = (preds >= 0.5).float()
        correct += (predicted == yb.squeeze()).sum().item()

    test_loss /= len(test_loader.dataset) # type: ignore
    accuracy = correct / len(test_loader.dataset) # type: ignore
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

# ---------- Guardem els pesos obtinguts un cop finalitzem l'entrenament ----------

torch.save({
        'epoch': epoch,                     # Guardem l'època actual
        'model_state': model.state_dict(),
        'optimizer_state': optimizador.state_dict(),
        'loss': loss.item()
    }, checkpoint_path)