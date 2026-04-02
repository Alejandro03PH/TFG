import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import base64
import io
from MNIST_Train_model import ClassificadorCNN

app = Flask(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Càrrega del model CNN ----------

model_cnn = ClassificadorCNN()
checkpoint = torch.load('Xarxa neuronal/MNIST/Pesos_mnist.pth', map_location=device)
model_cnn.load_state_dict(checkpoint['model_state'])
model_cnn.to(device)
model_cnn.eval()
print("Model CNN carregat.")

# ---------- Calibració de les activacions màximes per l'SNN ----------

transform = transforms.Compose([transforms.ToTensor()])
dataset_train = datasets.MNIST(root='Xarxa neuronal/MNIST/dades', train=True, download=False, transform=transform)
n_train = int(len(dataset_train) * 0.8)
n_val   = len(dataset_train) - n_train
_, dataset_val = random_split(dataset_train, [n_train, n_val])
val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=0)

def computa_activacio_maxima(model, dataloader, percentil=95.5):
    model.eval()
    vals1, vals2, vals3, vals4 = [], [], [], []
    max_batches = 15

    with torch.no_grad():
        for i, (xb, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            xb = xb.to(device)
            a1   = model.conv1(xb)
            vals1.append(a1.flatten())
            a2   = model.conv2(a1)
            vals2.append(a2.flatten())
            flat = model.fc[0](a2)
            a3   = F.relu(model.fc[1](flat))
            vals3.append(a3.flatten())
            a4   = model.fc[3](a3)
            vals4.append(a4.flatten())

    def p_positiu(llista, p):
        tots     = torch.cat(llista)
        positius = tots[tots > 0]
        if positius.numel() == 0:
            raise ValueError("Cap activació positiva.")
        return torch.quantile(positius, p / 100.0).item()

    return (p_positiu(vals1, percentil), p_positiu(vals2, percentil),
            p_positiu(vals3, percentil), p_positiu(vals4, percentil))

a1_max, a2_max, a3_max, a4_max = computa_activacio_maxima(model_cnn, val_loader)
print(f"Màxims calibrats — conv1: {a1_max:.4f} | conv2: {a2_max:.4f} | fc1: {a3_max:.4f} | fc2: {a4_max:.4f}")

# ---------- Model SNN ----------

class SNN_CNN():
    def __init__(self, cnn_model, a1_max, a2_max, a3_max, a4_max, passos_temps=200):
        self.t      = passos_temps
        self.pool   = torch.nn.MaxPool2d(kernel_size=2)

        self.w_conv1 = cnn_model.conv1[0].weight.data / a1_max
        self.b_conv1 = cnn_model.conv1[0].bias.data   / (a1_max * passos_temps)
        self.w_conv2 = cnn_model.conv2[0].weight.data / a2_max
        self.b_conv2 = cnn_model.conv2[0].bias.data   / (a2_max * passos_temps)
        self.w_fc1   = cnn_model.fc[1].weight.data    / a3_max
        self.b_fc1   = cnn_model.fc[1].bias.data      / (a3_max * passos_temps)
        self.w_fc2   = cnn_model.fc[3].weight.data    / a4_max
        self.b_fc2   = cnn_model.fc[3].bias.data      / (a4_max * passos_temps)

    def forward(self, x_input):
        threshold = 1.0
        decay     = 1.0

        v1 = torch.zeros(1, 32, 14, 14, device=device)
        v2 = torch.zeros(1, 64,  7,  7, device=device)
        v3 = torch.zeros(128,          device=device)
        v4 = torch.zeros(10,           device=device)
        dispars = torch.zeros(10,      device=device)

        for _ in range(self.t):
            input_spikes = (torch.rand_like(x_input) < x_input).float()

            corrent1 = self.pool(F.conv2d(input_spikes, self.w_conv1, self.b_conv1, padding=1))
            v1 = v1 * decay + corrent1
            s1 = (v1 >= threshold).float()
            v1 = v1 - s1 * threshold

            corrent2 = self.pool(F.conv2d(s1, self.w_conv2, self.b_conv2, padding=1))
            v2 = v2 * decay + corrent2
            s2 = (v2 >= threshold).float()
            v2 = v2 - s2 * threshold

            flat     = s2.flatten()
            corrent3 = self.w_fc1 @ flat + self.b_fc1
            v3       = v3 * decay + corrent3
            s3       = (v3 >= threshold).float()
            v3       = v3 - s3 * threshold

            corrent4 = self.w_fc2 @ s3 + self.b_fc2
            v4       = v4 * decay + corrent4
            s4       = (v4 >= threshold).float()
            v4       = v4 - s4 * threshold

            dispars += s4

        return dispars / self.t

model_snn = SNN_CNN(model_cnn, a1_max, a2_max, a3_max, a4_max, passos_temps=200)
print("Model SNN creat. Servidor llest.")

# ---------- Transformació de la imatge rebuda ----------

transform_imatge = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# ---------- Endpoint de predicció ----------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_b64 = data['image'].split(',')[1]          # Treiem el prefix "data:image/png;base64,"
    img_bytes = base64.b64decode(img_b64)
    imatge  = Image.open(io.BytesIO(img_bytes))

    tensor = transform_imatge(imatge).unsqueeze(0).to(device)  # [1, 1, 28, 28] #type: ignore

    # Inversió automàtica si el fons és blanc
    if tensor.mean().item() > 0.5:
        tensor = 1.0 - tensor

    # ---- CNN ----
    with torch.no_grad():
        logits    = model_cnn(tensor).squeeze()
        probs_cnn = F.softmax(logits, dim=0).tolist()

    # ---- SNN ----
    with torch.no_grad():
        taxes     = model_snn.forward(tensor)
        total_tax = taxes.sum().item()
        probs_snn = (taxes / (total_tax + 1e-9)).tolist() if total_tax > 0 else [0.0] * 10
        taxes_raw = taxes.tolist()

    return jsonify({
        'cnn': probs_cnn,
        'snn': probs_snn,
        'snn_raw': taxes_raw,   # taxes de dispar reals (no normalitzades) per informació addicional
    })

# ---------- Servim el frontend ----------

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000)