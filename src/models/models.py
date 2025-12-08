from src.models.deit_model import DeiTModel
from src.models.vit_model import VitModel
from src.models.attention_cnn_model import AttentionCNNModel
# --- MODIFICA 1: Importiamo la tua nuova classe ---
from src.models.lit_spectral_transformer import LitSpectralTransformer

def get_model(model_type: str, pretrained: bool, num_classes: int, in_channels: int = 3):
    """
    Factory function per istanziare i modelli in base alla configurazione.
    """
    
    # Logica per il modello originale del tuo collega
    if model_type == "attention_combined_cnn":
        return AttentionCNNModel(num_bands=in_channels, num_classes=num_classes)
    
    # --- MODIFICA 2: Aggiungiamo il tuo modello ---
    elif model_type == "spectral_transformer":
        # Nota: 'in_channels' qui riceve il valore 'bands' (es. 30) dal config
        return LitSpectralTransformer(bands=in_channels, num_classes=num_classes)
    
    # Gestione errori se il nome non esiste
    else:
        raise ValueError(f"Model type '{model_type}' not recognized in models.py")