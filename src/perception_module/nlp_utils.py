import numpy as np
import webcolors


EMBEDDING_MODEL = "text-embedding-3-small"

def semantic_similarity(word2vec_model, word1: str, word2: str) -> float:
    """Calculate semantic similarity between two words using Word2Vec."""
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()

    if word1 == word2:
        return 1.0

    if word2vec_model is None:
        return 0.0

    try:
        def get_phrase_vector(phrase):
            words = phrase.replace('_', ' ').split()
            vectors = []
            for word in words:
                if word in word2vec_model:
                    vectors.append(word2vec_model[word])
            if vectors:
                return np.mean(vectors, axis=0)
            return None

        vec1 = get_phrase_vector(word1)
        vec2 = get_phrase_vector(word2)

        if vec1 is None or vec2 is None:
            return 0.0

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return max(0.0, float(similarity))

    except Exception as e:
        return 0.0



def color_name_to_rgb(color_name: str) -> tuple:
    """
    Convert color name to normalized RGB values [0-1].

    Args:
        color_name: Color name (e.g. 'red', 'blue', 'dark green')

    Returns:
        tuple: (r, g, b) normalized [0-1], or None if color is not recognized
    """
    if not color_name or color_name.strip() == "":
        return None

    color_name = color_name.lower().strip()

    try:
        # Try with webcolors (supports standard CSS names)
        rgb = webcolors.name_to_rgb(color_name)
        return (rgb.red / 255.0, rgb.green / 255.0, rgb.blue / 255.0)
    except:
        pass

    # Fallback: basic color dictionary
    BASIC_COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        'beige': (245, 245, 220),
        'tan': (210, 180, 140),
        'silver': (192, 192, 192),
        'gold': (255, 215, 0)
    }

    if color_name in BASIC_COLORS:
        rgb = BASIC_COLORS[color_name]
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

    # Se non riconosciuto, ritorna None
    return None


def color_similarity_rgb(color1: str, color2: str, word2vec_model=None) -> float:
    """
    Calcola la similarità tra due colori usando la distanza RGB euclidea.
    Se i colori non sono riconosciuti, fa fallback a word2vec.

    Args:
        color1, color2: Nomi dei colori da confrontare
        word2vec_model: Modello word2vec per fallback (opzionale)

    Returns:
        float: Similarità [0, 1] dove 1 = colori identici, 0 = colori molto diversi
    """
    # Se sono esattamente uguali
    if color1.lower().strip() == color2.lower().strip():
        return 1.0

    # Converti i nomi in RGB
    rgb1 = color_name_to_rgb(color1)
    rgb2 = color_name_to_rgb(color2)

    # Se uno dei due non è riconosciuto, usa fallback word2vec
    if rgb1 is None or rgb2 is None:
        if word2vec_model is not None:
            # Fallback a similarità semantica word2vec
            return semantic_similarity(word2vec_model, color1, color2)
        else:
            return 0.0

    # Calcola distanza euclidea nello spazio RGB normalizzato
    # Distanza massima possibile = sqrt(3) (da bianco a nero)
    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    max_distance = np.sqrt(3.0)  # sqrt(1^2 + 1^2 + 1^2)

    # Converti distanza in similarità: 0 distanza = 1 similarità
    similarity = 1.0 - (distance / max_distance)

    return max(0.0, min(1.0, similarity))

def get_embedding(client, text):
    """
    Restituisce embedding vettoriale tramite OpenAI.
    Usato per matching semantico tra descrizioni oggetti.
    """
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return np.array(resp.data[0].embedding)
    except Exception as e:
        print(f"Error during embedding creation: {e}")
        print("\n\n\n")  # Spaces for debugging
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)



def lost_similarity(word2vec_model, label1, label2, color1, color2, material1, material2, desc1, desc2):
    """
    Calculate overall similarity between two objects using:
    lost_similarity = alpha*label_sim + beta*color_sim + gamma*material_sim + delta*desc_sim

    MODIFIED v7: Optimized weights to give more importance to description
    - alpha (label):       0.25 (25%)
    - beta (color):        0.20 (20%) - USES RGB DISTANCE instead of word2vec
    - gamma (material):    0.15 (15%) - reduced because "paper" matched too much
    - delta (description): 0.40 (40%) - increased to distinguish similar objects

    Args:
        word2vec_model: Word2vec model for semantic similarity
        label1, label2: Labels of the two objects
        color1, color2: Colors of the two objects (color name strings)
        material1, material2: Materials of the two objects
        desc1, desc2: Descriptions of the two objects (embeddings)

    Returns:
        float: Overall similarity [0, 1]
    """
    alpha = 0.05   # label weight
    beta = 0.30    # color weight
    gamma = 0.15   # material weight (reduced from 0.25)
    delta = 0.50  # description weight (increased from 0.25)

    # Calculate individual similarities
    label_sim = semantic_similarity(word2vec_model, label1, label2)

    # MODIFIED: Use RGB distance for colors instead of word2vec (with fallback)
    color_sim = color_similarity_rgb(color1, color2, word2vec_model)

    material_sim = semantic_similarity(word2vec_model, material1, material2)

    # For description, use cosine similarity between embeddings
    if desc1 is not None and desc2 is not None:
        desc_sim = cosine_similarity(desc1, desc2)
    else:
        desc_sim = 0.0

    # Calculate weighted similarity
    total_sim = alpha * label_sim + beta * color_sim + gamma * material_sim + delta * desc_sim

    return total_sim
