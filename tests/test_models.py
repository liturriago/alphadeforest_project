import pytest
import torch
from alphadeforest.models.cae import ConvolutionalAutoencoder
from alphadeforest.models.memory import MemoryNetwork
from alphadeforest.models.alpha_deforest import AlphaDeforest

@pytest.fixture
def model_params():
    """Configuración estándar para los tests."""
    return {
        "batch_size": 2,
        "time_steps": 5,
        "emb_dim": 64,
        "latent_dim": 64,
        "h": 128,
        "w": 128,
        "cae_h": 16, # 128 / 2 / 2 / 2
        "cae_w": 16,
        "hidden_mem": 128
    }

def test_cae_shapes(model_params):
    """Verifica que el Autoencoder mantenga dimensiones y comprima bien."""
    model = ConvolutionalAutoencoder(
        embedding_dim=model_params["emb_dim"], 
        latent_dim=model_params["latent_dim"]
    )
    
    # Simular un batch de imágenes (B, D, H, W)
    x = torch.randn(
        model_params["batch_size"], 
        model_params["emb_dim"], 
        model_params["h"], 
        model_params["w"]
    )
    
    x_rec, z_f = model(x)
    
    # 1. La reconstrucción debe tener el mismo tamaño que la entrada
    assert x_rec.shape == x.shape, f"Error en reconstrucción: {x_rec.shape}"
    
    # 2. El espacio latente debe estar comprimido (B, latent_dim, cae_h, cae_w)
    assert z_f.shape == (
        model_params["batch_size"], 
        model_params["latent_dim"], 
        model_params["cae_h"], 
        model_params["cae_w"]
    )

def test_memory_network_shapes(model_params):
    """Verifica que la LSTM + Attention devuelva un vector del tamaño correcto."""
    input_dim = model_params["latent_dim"] * model_params["cae_h"] * model_params["cae_w"]
    model = MemoryNetwork(input_dim=input_dim, hidden_dim=model_params["hidden_mem"])
    
    # Simular secuencia de embeddings aplanados (B, T, Z)
    z_seq = torch.randn(
        model_params["batch_size"], 
        model_params["time_steps"], 
        input_dim
    )
    
    z_pred = model(z_seq)
    
    # Debe devolver una predicción para el siguiente paso (B, Z)
    assert z_pred.shape == (model_params["batch_size"], input_dim)

def test_alpha_deforest_full_flow(model_params):
    """Test de integración: verifica el diccionario de salida del modelo completo."""
    model = AlphaDeforest(
        embedding_dim=model_params["emb_dim"],
        latent_dim=model_params["latent_dim"],
        cae_h=model_params["cae_h"],
        cae_w=model_params["cae_w"],
        hidden_dim_mem=model_params["hidden_mem"]
    )
    
    # Simular batch de secuencias temporales (B, T, D, H, W)
    x_seq = torch.randn(
        model_params["batch_size"], 
        model_params["time_steps"], 
        model_params["emb_dim"], 
        model_params["h"], 
        model_params["w"]
    )
    
    outputs = model(x_seq)
    
    # Verificar que todas las llaves necesarias existan
    expected_keys = ["reconstructions", "z_f", "z_pred", "recon_error"]
    for key in expected_keys:
        assert key in outputs, f"Falta la llave {key} en el output"

    # Verificar dimensiones clave
    # z_pred debe tener (T-1) pasos porque predice a partir del segundo
    assert outputs["z_pred"].shape[1] == model_params["time_steps"] - 1
    assert outputs["recon_error"].shape == (model_params["batch_size"], model_params["time_steps"])