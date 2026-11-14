# carga de modelo y librerías
import pandas as pd # esta abajo duplicado

'''
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd

# Cargar BETO (Spanish BERT)
print("Cargando BETO desde HuggingFace (puede tardar la primera vez)...")
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model.eval()  # Modo evaluación (no entrenamiento)

print("BETO cargado correctamente")
print(f"   Modelo: dccuchile/bert-base-spanish-wwm-cased")
print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Dimensión del embedding: {model.config.hidden_size}")
print(f"   Capas de atención: {model.config.num_hidden_layers}")
print(f"   Cabezas de atención: {model.config.num_attention_heads}")
'''

ruta_archivo = '../data/processed/processed_es_fa.csv'
texto = pd.read_csv(ruta_archivo)


def obtener_embedding_token(texto, palabra_objetivo, tokenizer, model):
    """
    Obtiene el embedding contextual de una palabra específica dentro de un texto.
    Usa la última capa oculta de BETO.
    """
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state: [batch, seq_len, hidden_dim]
    hidden_states = outputs.last_hidden_state[0]  # Primera (y única) oración

    # Decodificar tokens para encontrar la palabra objetivo
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Buscar el token que corresponde a la palabra objetivo
    for i, token in enumerate(tokens):
        if palabra_objetivo.lower() in token.lower():
            return hidden_states[i].numpy(), tokens

    # Si no se encuentra exacto, buscar por subword
    for i, token in enumerate(tokens):
        if token.replace("##", "") in palabra_objetivo.lower():
            return hidden_states[i].numpy(), tokens

    return None, tokens


# --- Demostración: "banco" en diferentes contextos ---
oraciones_banco = [
    ("Fui al banco a depositar dinero en mi cuenta.", "banco"),
    ("Me senté en el banco del parque a leer un libro.", "banco"),
    ("El banco de peces nadaba cerca de la costa.", "banco"),
]

print("=== EMBEDDINGS CONTEXTUALES: 'banco' en 3 contextos ===\n")

embeddings_banco = []
for oracion, palabra in oraciones_banco:
    emb, tokens = obtener_embedding_token(oracion, palabra, tokenizer, model)
    if emb is not None:
        embeddings_banco.append(emb)
        print(f"Contexto: \"{oracion}\"")
        print(f"Tokens:   {tokens}")
        print(f"Vector (primeros 5): [{', '.join(f'{v:.4f}' for v in emb[:5])}...]\n")

# Calcular similitudes entre los diferentes "banco"
if len(embeddings_banco) >= 3:
    from sklearn.metrics.pairwise import cosine_similarity

    sim_01 = cosine_similarity([embeddings_banco[0]], [embeddings_banco[1]])[0][0]
    sim_02 = cosine_similarity([embeddings_banco[0]], [embeddings_banco[2]])[0][0]
    sim_12 = cosine_similarity([embeddings_banco[1]], [embeddings_banco[2]])[0][0]

    print("  Similitudes entre los 3 usos de 'banco':")
    print(f"    banco(dinero)  <-> banco(parque):  {sim_01:.4f}")
    print(f"    banco(dinero)  <-> banco(peces):   {sim_02:.4f}")
    print(f"    banco(parque)  <-> banco(peces):   {sim_12:.4f}")
    print(f"\n  ¡Los vectores son DIFERENTES para la misma palabra!")
    print(f"     En Word2Vec, 'banco' tendría siempre el mismo vector.")
