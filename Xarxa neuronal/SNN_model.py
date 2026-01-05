import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from Train_model import Classificador
import Dades

train_loader, val_loader, test_loader = Dades.carregador_dades("Xarxa neuronal/dades.csv") # Carreguem les dades utilitzant la funció del mòdul Dades

def computa_activacio_maxima(model, dataloader):
    """
    Calcula la màxima activació de les capes ocultes i de sortida del model.

    Retorna:
        a1_max: Activació màxima de la capa oculta
        a2_max: Activació màxima de la capa de sortida
    """
    model.eval()

    a1_max = 0.0
    a2_max = 0.0

    with torch.no_grad():
        for xb, _ in dataloader:
            # ---- Capa oculta ----
            z1 = model.net[0](xb)      # Linear
            a1 = torch.relu(z1)        # ReLU
            a1_batch_max = a1.max().item()
            if a1_batch_max > a1_max:
                a1_max = a1_batch_max

            # ---- Capa de sortida ----
            z2 = model.net[2](a1)      # Linear
            a2 = torch.relu(z2)        # ReLU
            a2_batch_max = a2.max().item()
            if a2_batch_max > a2_max:
                a2_max = a2_batch_max

    # Seguretat: Comprovem que no siguin zero
    if a1_max == 0.0:
        raise ValueError("a1_max es zero — capa oculta mai activada.")
    if a2_max == 0.0:
        raise ValueError("a2_max es zero — capa de sortida mai activada.")

    return a1_max, a2_max

model = Classificador()

# Carreguem els pesos del model entrenat
checkpoint = torch.load("Xarxa neuronal/Pesos_classificador.pth")
model.load_state_dict(checkpoint['model_state'])
model.eval()

a1_max, a2_max = computa_activacio_maxima(model, val_loader)

# 1. SNN MODEL

class SNN():
    def __init__(self, mlp_model, a1_max, a2_max, passos_temps=50):
        self.t = passos_temps
        
        # Extracció de pesos i biaixos de l'MLP per a l'SNN:
        
        # Capa 1
        self.w1 = mlp_model.net[0].weight.data / a1_max  # Matriu 6x2
        self.b1 = mlp_model.net[0].bias.data / a1_max    # Vector 6
        
        # Capa Sortida
        self.w2 = mlp_model.net[2].weight.data / a2_max  # Matriu 1x6
        self.b2 = mlp_model.net[2].bias.data / a2_max    # Vector 1

    def __call__(self, x_input):
        return self.forward(x_input)

    def forward(self, x_input):
              
        # --- CAPA 1 (Entrada -> Oculta 1) ---

        threshold= 1.0     # Llindar sortida
        decay = 0.9         # Leak (mantenim memòria)

        v1 = torch.zeros(self.w1.size(0))  # Inicialitzem potencials de membrana a zero
        v2 = torch.zeros(self.w2.size(0))  # Inicialitzem potencials de membrana a zero
    
        dispars_totals = 0
        
        for pas in range(self.t):
            
            # Input rate coding
            input_spikes = (torch.rand_like(x_input) < x_input).float()

            # CAPA OCULTA
            corrent1 = torch.matmul(input_spikes, self.w1.T) + self.b1
            v1 = v1 * decay + corrent1
            
            spikes1 = (v1 >= threshold).float()
            v1 = v1 - (spikes1 * threshold)  # Reset amb el SEU threshold
            
            # CAPA SORTIDA 
            
            corrent2 = torch.matmul(spikes1, self.w2.T) + self.b2
            v2 = v2 * decay + corrent2
            
            spike_out = (v2 >= threshold).float()
            v2 = v2 - (spike_out * threshold)
            
            if spike_out > 0:
                dispars_totals += 1
                
        return dispars_totals / self.t


#--------------------------------------------------------------------------------------------------------------------


# 2. FUNCIONS AUXILIARS

def prediccio_mlp(model, punt):
    tensor = torch.tensor(normalize_xy(punt[0], punt[1]), dtype=torch.float32).unsqueeze(0)  # [1, 2]
    with torch.no_grad():
        prob = model(tensor).squeeze().item()
    return prob

def prediccio_snn(model, punt):
    tensor = torch.tensor(normalize_xy(punt[0], punt[1]), dtype=torch.float32)
    with torch.no_grad():
        prob = model(tensor)   # already a float in [0,1]
    return prob

def normalize_xy(x, y):
    x_norm = (x + 40.0) / 80.0   # maps [-40,40] → [0,1]
    y_norm = (y + 20.0) / 40.0   # maps [-20,20] → [0,1]
    return (x_norm, y_norm)

def compute_output_max(model, dataloader):
        max_val = 0.0
        with torch.no_grad():
            for xb, _ in dataloader:
                out = model(xb)          # shape [batch, 1]
                batch_max = out.max().item()
                if batch_max > max_val:
                    max_val = batch_max
        return max_val

"""def inspeccionar_pesos(model):
    print("\n" + "="*60)
    print("RADIOGRAFIA COMPLETA (TOTS ELS PESOS I BIAIXOS)")
    print("="*60)
    
    # Recorrem les capes lineals: 0 (Entrada), 2 (Oculta), 4 (Sortida)
    # He afegit el 4 perquè també vegis la capa final
    for idx in [0, 2, 4]:
        
        # Comprovació de seguretat per si la xarxa és més petita
        if idx >= len(model.net): continue
            
        capa = model.net[idx]
        
        # Només volem veure capes Lineals (amb pesos)
        if isinstance(capa, torch.nn.Linear):
            w = capa.weight.data
            b = capa.bias.data
            
            print(f"\n--- CAPA {idx} ({capa.in_features} entrades -> {capa.out_features} sortides) ---")
            
            # 1. BIAIXOS
            print(f" > BIAIXOS (Vector de {len(b)}):")
            print(b.tolist())
            
            # 2. PESOS
            print(f"\n > PESOS (Matriu {w.shape[0]}x{w.shape[1]}):")
            print("   (Cada fila és una neurona de destí. Cada columna ve de l'entrada anterior)")
            print(w) # Imprimeix el Tensor directament, que es llegeix prou bé
            
            # Si prefereixes format llista per copiar-ho, descomenta la línia de sota:
            # print(w.tolist())

    print("="*60 + "\n")"""

"""def generar_mapes():
    # Carreguem pesos
    mlp = Classificador()
    try:
        ruta = 'Xarxa neuronal/Pesos_classificador.pth'
        ckpt = torch.load(ruta)
        state = ckpt['model_state'] if (isinstance(ckpt, dict) and 'model_state' in ckpt) else ckpt
        mlp.load_state_dict(state)
        mlp.eval()
        print(f"Pesos carregats de: {ruta}")
    except:
        print("AVÍS: No s'han trobat els pesos. Usant pesos aleatoris (el mapa no tindrà sentit real).")

    # Instanciem l'SNN amb els pesos de l'MLP
    snn = SNN(mlp, a1_max, a2_max, passos_temps=50) # 50 passos és suficient per visualitzar

    # Definim el Grid
    x_val = np.arange(-40, 41, 1) # De -40 a 40
    y_val = np.arange(-20, 21, 1) # De -20 a 20
    
    grid_mlp = np.zeros((len(y_val), len(x_val)))
    grid_snn = np.zeros((len(y_val), len(x_val)))

    print(f"\nGenerant prediccions per a {len(x_val)*len(y_val)} punts...")
    start_time = time.time()

    # Bucle principal
    with torch.no_grad():
        for i, y in enumerate(y_val):
            for j, x in enumerate(x_val):
                punt = torch.tensor([float(x), float(y)])
                
                # Predicció MLP
                pred_mlp = mlp(punt.unsqueeze(0)).item()/1.3394  # Normalitzem pel màxim obtingut en validació
                grid_mlp[i, j] = pred_mlp
                
                # Predicció SNN
                pred_snn = snn.forward(punt)
                grid_snn[i, j] = pred_snn
            
            # Barra de progrés simple
            if i % 5 == 0:
                print(f"Processant fila Y={y}...")

    print(f"Fet en {time.time() - start_time:.2f} segons.")


    # 3. Plot dels resultats
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mapa MLP
    im1 = axes[0].imshow(grid_mlp, origin='lower', extent=[-40, 40, -20, 20], 
                         cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Predicció MLP (Original)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Mapa SNN
    im2 = axes[1].imshow(grid_snn, origin='lower', extent=[-40, 40, -20, 20], 
                         cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Predicció SNN (Spiking)')
    axes[1].set_xlabel('X')
    
    # Text amb els paràmetres usats (per recordatori visual)
    text_params = "Ajusta 'amp' i 'thresh'\nper igualar els colors!"
    axes[1].text(0.95, 0.05, text_params, transform=axes[1].transAxes, 
                 ha='right', color='white', fontsize=9, fontweight='bold')

    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()"""

if __name__ == '__main__':
    modelSNN = SNN(model, a1_max, a2_max, passos_temps=200)
    """while True:
        try:
            # Pregunta a l'usuari per les coordenades del punt
            x = input("\nPunt x: ").strip()
            y = input("Punt y: ").strip()
            xf = float(x)
            yf = float(y)
            score = prediccio_mlp(model, (xf, yf))
            a_max = compute_output_max(model, val_loader)
            print(f"Valor màxim de sortida a les dades de validació: {a_max:.4f}")
            prob = score / a_max

            print(f"\nPunt: ({xf}, {yf})")
            print(f"Probabilitat MLP: {prob:.4f}")

            scoreSNN = prediccio_snn(modelSNN, (xf, yf))
            print(f"Valor màxim de sortida a les dades de validació: {a_max:.4f}")

            print(f"\nPunt: ({xf}, {yf})")
            print(f"Probabilitat SNN: {scoreSNN:.4f}")            

        except ValueError:
            print("El valor introduït no és un valor numèric.")"""
    x_vals = np.arange(-40, 41, 1)
    y_vals = np.arange(-20, 21, 1)

    mlp_x, mlp_y = [], []
    snn_x, snn_y = [], []

    threshold = 0.5

    print("Evaluating grid...")

    Puntsiguals = 0

    for y in y_vals:
        for x in x_vals:
            # ANN
            p_mlp = prediccio_mlp(model, (x, y))
            if p_mlp >= threshold:
                mlp_x.append(x)
                mlp_y.append(y)
            # SNN
            p_snn = prediccio_snn(modelSNN, (x, y))
            if p_snn >= threshold+0.11   :  # Ajustem el llindar per a SNN per compensar diferències
                snn_x.append(x)
                snn_y.append(y)
            if ((p_mlp >= threshold) and (p_snn >= threshold+0.11)) or ((p_mlp < threshold) and (p_snn < threshold+0.11)):
                Puntsiguals += 1
    print(f"Punts iguals (ANN i SNN dins): {Puntsiguals} de {len(x_vals)*len(y_vals)}, {Puntsiguals/(len(x_vals)*len(y_vals))*100:.2f}%")

# ------------------------
# Plot
# ------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# ANN plot
    axes[0].scatter(mlp_x, mlp_y, c='green', s=8)
    axes[0].set_title("ANN")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_xlim(-40, 40)
    axes[0].set_ylim(-20, 20)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

# SNN plot
    axes[1].scatter(snn_x, snn_y, c='green', s=8)
    axes[1].set_title("SNN")
    axes[1].set_xlabel("X")
    axes[1].set_xlim(-40, 40)
    axes[1].set_ylim(-20, 20)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()