import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
from MNIST_Train_model import ClassificadorCNN


# ---------- Càrrega del dataset de validació (per calibrar els màxims) ----------

transform = transforms.Compose([transforms.ToTensor()])

dataset_train_complet = datasets.MNIST(root='Xarxa neuronal/MNIST/dades', train=True,  download=False, transform=transform)
n_train = int(len(dataset_train_complet) * 0.8)
n_val   = len(dataset_train_complet) - n_train
_, dataset_val = random_split(dataset_train_complet, [n_train, n_val])
val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=0)  # num_workers=0 per compatibilitat amb Windows


# ---------- Càrrega del model CNN entrenat ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ClassificadorCNN()
checkpoint = torch.load('Xarxa neuronal/MNIST/Pesos_mnist.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# ---------- Nou log ----------

def log_print(message):
    with open('Xarxa neuronal/MNIST/MNIST_SNN_log.txt', 'a', encoding='utf-8') as f:
        f.write(f'{message}\n')
    print(f'{message}')

# ---------- Càlcul de les activacions màximes per capa ----------

def computa_activacio_maxima(model, dataloader, percentil=95.5):
    """
    Recorre el dataset i calcula el percentil 90 de les activacions de cada capa ReLU.
    Aquest percentil es fa servir com a factor de normalització per convertir els pesos del CNN a taxes de dispar en la SNN.
    """
    model.eval()

    # Acumulem TOTES les activacions de cada capa per poder calcular el percentil
    vals1, vals2, vals3, vals4 = [], [], [], []

    max_batches = 15  # Prou mostres per un percentil representatiu sense desbordament de memòria

    with torch.no_grad():
        for i, (xb, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            xb = xb.to(device)

            # ---- Fase 1: conv1 -> pool -> ReLU ----
            a1 = model.conv1(xb)
            vals1.append(a1.flatten())

            # ---- Fase 2: conv2 -> pool -> ReLU ----
            a2 = model.conv2(a1)
            vals2.append(a2.flatten())

            # ---- Fase 3a: flatten -> linear1 -> ReLU ----
            flat = model.fc[0](a2)
            a3   = F.relu(model.fc[1](flat))
            vals3.append(a3.flatten())

            # ---- Fase 3b: linear2 (sortida, sense ReLU) ----
            a4 = model.fc[3](a3)
            vals4.append(a4.flatten())

    # Calculem el percentil NOMÉS sobre els valors positius de cada capa. (Evitem zeros i negatius, es a dir, classes incorrectes i activacions ReLU saturades, que desplacen el percentil cap a zero i donen factors de normalització incorrectes. )

    def percentil_positiu(llista, p):
        tots = torch.cat(llista)
        positius = tots[tots > 0]
        if positius.numel() == 0:
            raise ValueError("Cap activació positiva trobada — capa mai activada.")
        return torch.quantile(positius, p / 100.0).item()

    a1_p99 = percentil_positiu(vals1, percentil)
    a2_p99 = percentil_positiu(vals2, percentil)
    a3_p99 = percentil_positiu(vals3, percentil)
    a4_p99 = percentil_positiu(vals4, percentil)

    return a1_p99, a2_p99, a3_p99, a4_p99


a1_max, a2_max, a3_max, a4_max = computa_activacio_maxima(model, val_loader)
log_print(f"Percentil 90 activacions — conv1: {a1_max:.4f} | conv2: {a2_max:.4f} | fc1: {a3_max:.4f} | fc2: {a4_max:.4f}")


# ---------- Model SNN ----------

class SNN_CNN():
    def __init__(self, cnn_model, a1_max, a2_max, a3_max, a4_max, passos_temps=200):
        self.t = passos_temps

        # Extracció i normalització dels pesos de cada capa

        # Conv1 (conv2d: pesos [32,1,3,3] i biaix [32])

        # Conv1
        self.w_conv1 = cnn_model.conv1[0].weight.data / a1_max
        self.b_conv1 = cnn_model.conv1[0].bias.data   / (a1_max * passos_temps)

        # Conv2
        self.w_conv2 = cnn_model.conv2[0].weight.data / a2_max
        self.b_conv2 = cnn_model.conv2[0].bias.data   / (a2_max * passos_temps)

        # Linear1
        self.w_fc1 = cnn_model.fc[1].weight.data / a3_max
        self.b_fc1 = cnn_model.fc[1].bias.data   / (a3_max * passos_temps)

        # Linear2 — capa de sortida
        self.w_fc2 = cnn_model.fc[3].weight.data / a4_max
        self.b_fc2 = cnn_model.fc[3].bias.data   / (a4_max * passos_temps)

        self.pool = torch.nn.MaxPool2d(kernel_size=2)

    def __call__(self, x_input):
        return self.forward(x_input)

    def forward(self, x_input):
        """
        Pas endavant de la SNN amb rate coding.
        x_input: tensor [1, 1, 28, 28] amb valors en [0, 1].
        Retorna: tensor [10] amb les taxes de dispar acumulades per cada classe.
        """
        threshold = 1.0  # Llindar de dispar
        decay     = 1.0  # Sense leak (Integrate-and-Fire pur): equivalent matemàtic al CNN.
                         # Amb decay < 1 (LIF), neurones poc actives mai acumulen prou potencial.

        # Inicialitzem els potencials de membrana de cada capa a zero
        v1 = torch.zeros(1, 32, 14, 14, device=device)  # Sortida conv1 + pool: [1,32,14,14]
        v2 = torch.zeros(1, 64,  7,  7, device=device)  # Sortida conv2 + pool: [1,64,7,7]
        v3 = torch.zeros(128,             device=device) # Sortida linear1:      [128]
        v4 = torch.zeros(10,              device=device) # Sortida linear2:      [10]

        dispars_totals = torch.zeros(10, device=device)  # Acumulem dispars de sortida per classe

        for _ in range(self.t):

            # ---- Rate coding: generem spikes d'entrada proporcionals a la intensitat del píxel ----
            input_spikes = (torch.rand_like(x_input) < x_input).float()  # [1,1,28,28] spikes binaris on la probabilitat de spike és proporcional al valor del píxel

            # ---- CAPA 1: conv1 + pool -> potencial -> dispar ----
            corrent1 = self.pool(F.conv2d(input_spikes, self.w_conv1, self.b_conv1, padding=1))
            v1 = v1 * decay + corrent1
            spikes1 = (v1 >= threshold).float()
            v1 = v1 - spikes1 * threshold              # Reset subtractiu

            # ---- CAPA 2: conv2 + pool -> potencial -> dispar ----
            corrent2 = self.pool(F.conv2d(spikes1, self.w_conv2, self.b_conv2, padding=1))
            v2 = v2 * decay + corrent2
            spikes2 = (v2 >= threshold).float()
            v2 = v2 - spikes2 * threshold

            # ---- CAPA 3: flatten -> linear1 -> potencial -> dispar ----
            flat = spikes2.flatten()                   # [3136]
            corrent3 = self.w_fc1 @ flat + self.b_fc1  # [128]
            v3 = v3 * decay + corrent3
            spikes3 = (v3 >= threshold).float()
            v3 = v3 - spikes3 * threshold

            # ---- CAPA SORTIDA: linear2 -> potencial -> dispar ----
            corrent4 = self.w_fc2 @ spikes3 + self.b_fc2  # [10]
            v4 = v4 * decay + corrent4
            spikes4 = (v4 >= threshold).float()
            v4 = v4 - spikes4 * threshold

            dispars_totals += spikes4                  # Acumulem dispars de cada classe

        return dispars_totals / self.t                 # Taxa de dispar [0, 1] per cada classe


# ---------- Transformació de la imatge (igual que a MNIST_Test_model.py) ----------

transform_imatge = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


# ---------- Funció de predicció conjunta CNN + SNN ----------

def prediccio(model_cnn, model_snn, ruta_imatge):
    """
    Donada la ruta d'una imatge, retorna la predicció del model CNN i del model SNN.
    Si la imatge és majoritàriament blanca (fons blanc), s'inverteixen els colors
    per adaptar-la al format MNIST (fons negre, dígit blanc).
    """
    imatge = Image.open(ruta_imatge)
    if imatge.size != (28, 28):
        print(f"Advertència: imatge redimensionada de {imatge.size[0]}×{imatge.size[1]} a 28×28.")
        imatge = transforms.Resize((28, 28))(imatge)

    tensor = transform_imatge(imatge).unsqueeze(0).to(device)  # [1, 1, 28, 28] # type: ignore

    # Inversió automàtica si el fons és blanc
    if tensor.mean().item() > 0.5:
        tensor = 1.0 - tensor

    # ---- Predicció CNN ----
    with torch.no_grad():
        logits_cnn        = model_cnn(tensor).squeeze()          # [10]
        probs_cnn         = F.softmax(logits_cnn, dim=0)
        digit_cnn         = probs_cnn.argmax().item()
        confianca_cnn     = probs_cnn[digit_cnn].item()          # type: ignore

    # ---- Predicció SNN ----
    with torch.no_grad():
        taxes_snn         = model_snn(tensor)                    # [10] taxes de dispar
        digit_snn         = taxes_snn.argmax().item()
        confianca_snn     = taxes_snn[digit_snn].item()          # type: ignore
        probs_snn         = taxes_snn / (taxes_snn.sum() + 1e-9) # Normalitzem per visualitzar com a %

    return digit_cnn, confianca_cnn, probs_cnn, digit_snn, confianca_snn, probs_snn


# ---------- Avaluació sobre el dataset de test ----------
 
def avalua_models(model_cnn, model_snn, dataloader):
    """
    Avalua el model CNN, el model SNN i la quantitat de vegades que la SNN no retorna res
    sobre un DataLoader complet i imprimeix el percentatge d'encert de
    cadascun, igual que la funció accuracy_snn/accuracy_mlp de l'SNN 
    original però per a classificació de 10 classes (argmax).
    """
    cnn_correct = snn_correct = total = snn_0 = 0
 
    with torch.no_grad():
        for i, (xb, yb) in enumerate(dataloader):
            xb, yb = xb.to(device), yb.to(device)
 
            # ---- CNN: argmax dels logits ----
            logits = model_cnn(xb)                                    # [batch, 10]
            cnn_correct += (logits.argmax(dim=1) == yb).sum().item()
 
            # ---- SNN: avaluem imatge per imatge (forward espera [1,1,28,28]) ----
            for j in range(xb.size(0)):
                taxes = model_snn(xb[j].unsqueeze(0))                 # [10]
                if taxes.argmax().item() == yb[j].item():
                    snn_correct += 1
                if taxes[taxes.argmax().item()] == 0:
                    snn_0 += 1
 
            total += xb.size(0)
 
            # Progrés cada 10 batches
            if (i + 1) % 10 == 0:
                print(f"  Avaluant... {total} imatges processades", end='\r')
 
    cnn_acc = cnn_correct / total * 100
    snn_acc = snn_correct / total * 100
 
    log_print(f"\n{'─'*45}")
    log_print(f"  CNN  ->  Encerts: {cnn_correct}/{total}   Accuracy: {cnn_acc:.2f}%")
    log_print(f"  SNN  ->  Encerts: {snn_correct}/{total}   Accuracy: {snn_acc:.2f}%  ({model_snn.t} passos)")
    log_print(f"  SNN  ->  Casos amb 0 dispars a la classe predita: {snn_0} ({snn_0/total*100:.2f}%)")
    log_print(f"{'─'*45}")
 
    return cnn_acc, snn_acc

# ---------- Loop interactiu ----------

if __name__ == '__main__':

    dataset_test = datasets.MNIST(root='Xarxa neuronal/MNIST/dades', train=False, download=False, transform=transform)
    test_loader  = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=0)

    modelSNN = SNN_CNN(model, a1_max, a2_max, a3_max, a4_max)

    while True:
        try:
            ruta = input("\nNom del fitxer d'imatge (o '/' per sortir): ").strip()

            if ruta.lower() == '/':
                print("Sortint...")
                break

            rutaCompleta = f'Xarxa neuronal/MNIST/{ruta}'
            digit_cnn, confianca_cnn, probs_cnn, digit_snn, confianca_snn, probs_snn = prediccio(model, modelSNN, rutaCompleta)

            # ---- Resultats CNN ----
            print(f"\n{'─'*45}")
            print(f"  CNN  ->  Dígit predit: {digit_cnn}   Confiança: {confianca_cnn*100:.2f}%")
            print(f"{'─'*45}")
            for i, p in enumerate(probs_cnn):
                barra = '█' * int(p.item() * 30)
                print(f"  {i}: {barra:<30} {p.item()*100:.2f}%")

            # ---- Resultats SNN ----
            print(f"\n{'─'*45}")
            print(f"  SNN  ->  Dígit predit: {digit_snn}   Confiança: {confianca_snn*100:.2f}%  ({modelSNN.t} passos)")
            print(f"{'─'*45}")
            for i, p in enumerate(probs_snn):
                barra = '█' * int(p.item() * 30)
                print(f"  {i}: {barra:<30} {p.item()*100:.2f}%")

        except FileNotFoundError:
            print(f"No s'ha trobat el fitxer '{ruta}'. Comprova la ruta i torna-ho a intentar.") # type: ignore
        except ValueError as e:
            print(f"Error: {e}")