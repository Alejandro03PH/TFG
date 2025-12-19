import torch
import torch.nn as nn

class Classificador(nn.Module):
    def __init__(self, n_entrada = 2, n_sortida = 1, n_oculta = 8): # Configurem les neurones d'entrada, sortida i ocultes.
        super().__init__() 
        self.net = nn.Sequential( # Definim la xarxa neuronal com una seqüència de capes.
            nn.Linear(n_entrada, n_oculta),   # Entrada → Capa oculta
            nn.ReLU(),                  # Funció d'activació ReLU
            nn.Linear(n_oculta, n_oculta),   # Capa oculta → Capa oculta
            nn.ReLU(),                  # Funció d'activació ReLU
            nn.Linear(n_oculta, n_sortida),   # Capa oculta → logit
            nn.Sigmoid()                # Converteix el logit a una probabilitat entre 0 i 1
        )

    # Ara es defineix el pas endavant de la xarxa (Aquest pas correspón a la primera predicció que fa la xarxa abans de calcular l'error)

    def forward(self, x):
        return self.net(x) 
    


checkpoint_path=('Xarxa neuronal/Pesos_classificador.pth')  # Ruta per desar el punt de control
model = Classificador()   # Creem una instància del model
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt['model_state'])
model.eval()  # Posem el model en mode avaluació



# --------- Definim la funció de predicció per a un punt donat ---------

def prediccio(model, punt):
    tensor = torch.tensor(punt, dtype=torch.float32).unsqueeze(0)  # [1, 2]
    with torch.no_grad():
        prob = model(tensor).squeeze().item()
        label = int(prob >= 0.5)
    return prob, label



# --------- Loop interactiu per a prediccions ---------

if __name__ == '__main__':
    net = model

    while True:
        try:
            # Pregunta a l'usuari per les coordenades del punt
            x = input("\nPunt x: ").strip()
            y = input("Punt y: ").strip()
            xf = float(x)
            yf = float(y)
            prob, label = prediccio(net, (xf, yf))

            print(f"\nPunt: ({xf}, {yf})")
            print(f"Probabilitat d'estar dintre: {prob:.4f}")

        except ValueError:
            print("El valor introduït no és un valor numèric.")