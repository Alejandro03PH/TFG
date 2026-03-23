import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from MNIST_Train_model import ClassificadorCNN
 
# --------- Carreguem el model entrenat ---------
 
checkpoint_path = 'Xarxa neuronal/MNIST/Pesos_mnist.pth'  # Ruta del punt de control
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
model = ClassificadorCNN()  # Creem una instància del model
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()  # Posem el model en mode avaluació
 
 
# --------- Definim la transformació de la imatge ---------
 
# La mateixa transformació que durant l'entrenament: escala de grisos + tensor [0, 1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Assegurem 1 canal (escala de grisos)
    transforms.ToTensor()                          # Converteix píxels [0, 255] -> [0.0, 1.0]
])
 
 
# --------- Definim la funció de predicció per a una imatge donada ---------
 
def prediccio(model, ruta_imatge):
 
    # Carreguem la imatge i verifiquem que sigui 28x28
    imatge = Image.open(ruta_imatge)
    if imatge.size != (28, 28):
        print(f"Advertència: La imatge '{ruta_imatge}' no té les dimensions 28x28. Es redimensionarà automàticament, però la qualitat es pot veure afectada, i per tant la precisió de la predicció també.")
        imatge = transforms.Resize((28, 28))(imatge)
 
    tensor = transform(imatge).unsqueeze(0).to(device)  # [1, 1, 28, 28] # type: ignore
 
    # MNIST espera fons negre (0) i dígit blanc (1).
    # Si la mitjana del tensor és baixa (prop de 0), la imatge ja és majoritàriament negra -> correcte.
    # Si la mitjana és alta (prop de 1), la imatge és majoritàriament blanca -> invertim els colors.
    if tensor.mean().item() > 0.5:
        tensor = 1.0 - tensor  # Inversió: cada píxel p -> 1 - p
 
    with torch.no_grad():
        logits = model(tensor).squeeze()         # [10] — un valor per cada dígit
        probabilitats = F.softmax(logits, dim=0) # Convertim logits a probabilitats [0, 1]
        digit_predit = probabilitats.argmax().item()
        confianca = probabilitats[digit_predit].item() # type: ignore
 
    return digit_predit, confianca, probabilitats
 
 
# --------- Loop interactiu per a prediccions ---------
 
if __name__ == '__main__':
 
    while True:
        try:
            ruta = input("\nNom del fitxer d'imatge (o '/' per sortir): ").strip()
 
            if ruta.lower() == '/':
                print("Sortint...")
                break
            
            rutaCompleta = f'Xarxa neuronal/MNIST/{ruta}'  # Construïm la ruta completa
            digit, confianca, probabilitats = prediccio(model, rutaCompleta)
 
            print(f"\nDígit predit: {digit}")
            print(f"Confiança:    {confianca * 100:.2f}%")
            print("\nProbabilitats per cada dígit:")
            for i, p in enumerate(probabilitats):
                barra = '█' * int(p.item() * 30)  # Barra visual proporcional
                print(f"  {i}: {barra:<30} {p.item() * 100:.2f}%")
 
        except FileNotFoundError:
            print(f"No s'ha trobat el fitxer '{ruta}'. Comprova la ruta i torna-ho a intentar.") # type: ignore
        except ValueError as e:
            print(f"Error: {e}")