import torch
from Train_model import Classificador
import Dades 

# --------- Carreguem el model entrenat ---------

checkpoint_path=('Xarxa neuronal/Classificador/Pesos_classificador.pth')  # Ruta per desar el punt de control
model = Classificador()   # Creem una instància del model
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt['model_state'])
model.eval()  # Posem el model en mode avaluació



# --------- Definim la funció de predicció per a un punt donat ---------

def prediccio(model, punt):
    tensor = torch.tensor(normalize_xy(punt[0], punt[1]), dtype=torch.float32).unsqueeze(0)  # [1, 2]
    with torch.no_grad():
        prob = model(tensor).squeeze().item()
        label = int(prob >= 0.5)
    return prob, label



# --------- Loop interactiu per a prediccions ---------

if __name__ == '__main__':

    train_loader, val_loader, test_loader = Dades.carregador_dades("Xarxa neuronal/Classificador/Dades.csv") # Carreguem les dades utilitzant la funció del mòdul Dades

    def compute_output_max(model, dataloader):
        max_val = 0.0
        with torch.no_grad():
            for xb, _ in dataloader:
                out = model(xb)          # shape [batch, 1]
                batch_max = out.max().item()
                if batch_max > max_val:
                    max_val = batch_max
        return max_val
    
    def normalize_xy(x, y):
        x = (x + 40.0) / 80.0
        y = (y + 20.0) / 40.0
        return x, y

    while True:
        try:
            # Pregunta a l'usuari per les coordenades del punt
            x = input("\nPunt x: ").strip()
            y = input("Punt y: ").strip()
            xf = float(x)
            yf = float(y)
            score, label = prediccio(model, (xf, yf))
            a_max = compute_output_max(model, val_loader)
            print(f"Valor màxim de sortida a les dades de validació: {a_max:.4f}")
            prob = score / a_max

            print(f"\nPunt: ({xf}, {yf})")
            print(f"Probabilitat d'estar dintre: {prob:.4f}")

        except ValueError:
            print("El valor introduït no és un valor numèric.")