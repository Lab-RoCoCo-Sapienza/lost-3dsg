#!/usr/bin/env python3
"""
Object Manager - Semantic and Spatial Tracking degli Oggetti Rilevati

COSA FA? booh

"""
import rclpy, json, os, time, logging, threading, sys, webcolors
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
from openai import OpenAI
from tiago_project.msg import ObjectDescriptionArray, Bbox3dArray
from visualization_msgs.msg import Marker, MarkerArray
from object_info import Object
from debug_utils import TrackingLogger
from world_model import wm
import gensim.downloader as api
from utils import *
from nlp_utils import *
from datetime import datetime
from cv_utils import *

# =============  EXPLORATION PARAMETERS =============
EXPLORATION_IOU_THRESHOLD = 0.18 # IoU threshold to match objects during exploration phase
EXPLORATION_MODE = True  # Flag to indicate if we are in exploration mode

SIM_THRESHOLD = 0.75
TRACKING_IOU_THRESHOLD = 0.3  # IoU threshold to consider objects as "moved" in tracking mode
VOLUME_EXPANSION_RATIO = 0.05 # 5% expansion relative to object dimensions


# Load OpenAI API key
# Setup file logger and project paths
file_path = os.path.abspath(__file__)
# Find project root: if in install dir, go to workspace root, else go up to find CMakeLists.txt
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))

with open(os.path.join(PROJECT_ROOT, "src", "perception_module", "api.txt"), "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)


world2vec = api.load('word2vec-google-news-300')


# Setup file logger and project paths
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))
log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"object_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Setup module logger
module_logger = logging.getLogger('object_manager_module')
module_logger.setLevel(logging.DEBUG)
module_logger.addHandler(file_handler)

# Dedicated tracking file for important operations
tracking_log_file = os.path.join(log_dir, f"tracking_operations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Initialize TrackingLogger
tracking_logger = TrackingLogger(tracking_log_file)


def create_object_key(label, material, color, description):
    """
    Crea una chiave univoca per un oggetto basata su label, material, color e description.

    Args:
        label: Label dell'oggetto
        material: Materiale dell'oggetto
        color: Colore dell'oggetto
        description: Descrizione dell'oggetto

    Returns:
        str: Chiave univoca (JSON string)
    """
    key_dict = {
        "label": label if label else "",
        "material": material if material else "",
        "color": color if color else "",
        "description": description if description else ""
    }
    return json.dumps(key_dict, sort_keys=True)


def find_best_matching_key(target_label, target_material, target_color, target_description,
                           target_embedding, keys_dict, word2vec_model, threshold=0.0):
    """
    Trova la chiave migliore in keys_dict che corrisponde all'oggetto target usando lost_similarity.

    Args:
        target_label, target_material, target_color, target_description: Attributi dell'oggetto target
        target_embedding: Embedding della descrizione dell'oggetto target
        keys_dict: Dizionario con chiavi composite da cercare
        word2vec_model: Modello word2vec
        threshold: Soglia minima di similarità

    Returns:
        tuple: (best_key, best_similarity) o (None, 0.0) se nessun match
    """
    best_key = None
    best_similarity = threshold

    for key in keys_dict.keys():
        key_data = json.loads(key)

        # Ottieni embedding per la descrizione della chiave
        key_embedding = get_embedding(client, key_data["description"]) if key_data["description"] else None

        # Calcola lost_similarity
        similarity = lost_similarity(
            word2vec_model,
            target_label, key_data["label"],
            target_color, key_data["color"],
            target_material, key_data["material"],
            target_embedding, key_embedding
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_key = key

    return best_key, best_similarity


def compute_pov_volume(bboxes_list, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """
    Calcola il POV (Point of View) volume che contiene tutte le detection.

    Questo volume rappresenta "cosa sta guardando la camera" in questo frame.
    Gli oggetti persistenti che sono DENTRO questo volume ma NON sono stati
    visti devono essere eliminati.

    Args:
        bboxes_list: lista di bbox dict (x_min, x_max, y_min, y_max, z_min, z_max)
        expansion_ratio: percentuale di espansione (default 0.2 = 20%)

    Returns:
        dict: POV volume espanso, o None se la lista è vuota
    """
    if not bboxes_list:
        return None

    # Soglia volume massimo per applicare riduzione bbox (in metri cubi)
    MAX_VOLUME_THRESHOLD = 0.5  # Se bbox > 0.5 m³, riduci del 30%
    BBOX_REDUCTION_RATIO = 0.30  # 30% di riduzione

    def shrink_bbox(bbox, ratio):
        """Riduce una bbox del ratio% mantenendo il centro"""
        x_center = (bbox["x_min"] + bbox["x_max"]) / 2.0
        y_center = (bbox["y_min"] + bbox["y_max"]) / 2.0
        z_center = (bbox["z_min"] + bbox["z_max"]) / 2.0

        x_size = (bbox["x_max"] - bbox["x_min"]) * (1.0 - ratio)
        y_size = (bbox["y_max"] - bbox["y_min"]) * (1.0 - ratio)
        z_size = (bbox["z_max"] - bbox["z_min"]) * (1.0 - ratio)

        return {
            "x_min": x_center - x_size / 2.0,
            "x_max": x_center + x_size / 2.0,
            "y_min": y_center - y_size / 2.0,
            "y_max": y_center + y_size / 2.0,
            "z_min": z_center - z_size / 2.0,
            "z_max": z_center + z_size / 2.0
        }

    def bbox_volume(bbox):
        """Calcola il volume di una bbox"""
        return ((bbox["x_max"] - bbox["x_min"]) *
                (bbox["y_max"] - bbox["y_min"]) *
                (bbox["z_max"] - bbox["z_min"]))

    # Processa le bbox: se troppo grandi, riducile
    processed_bboxes = []
    for bbox in bboxes_list:
        vol = bbox_volume(bbox)
        if vol > MAX_VOLUME_THRESHOLD:
            # Bbox troppo grande, usa versione ridotta per calcolo massimi
            processed_bbox = shrink_bbox(bbox, BBOX_REDUCTION_RATIO)
            print(f"   [POV] Bbox volume {vol:.3f} m³ > {MAX_VOLUME_THRESHOLD} m³ → ridotta del {BBOX_REDUCTION_RATIO*100:.0f}%")
        else:
            processed_bbox = bbox
        processed_bboxes.append(processed_bbox)

    # Trova i limiti che contengono TUTTE le bbox (processate)
    x_min = min(bbox["x_min"] for bbox in processed_bboxes)
    x_max = max(bbox["x_max"] for bbox in processed_bboxes)
    y_min = min(bbox["y_min"] for bbox in processed_bboxes)
    y_max = max(bbox["y_max"] for bbox in processed_bboxes)
    z_min = min(bbox["z_min"] for bbox in processed_bboxes)
    z_max = max(bbox["z_max"] for bbox in processed_bboxes)

    # Calcola dimensioni del POV
    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    # Espandi proporzionalmente
    x_expansion = x_size * expansion_ratio
    y_expansion = y_size * expansion_ratio
    z_expansion = z_size * expansion_ratio

    # Espansione minima per evitare volumi troppo piccoli (es. singolo oggetto)
    MIN_EXPANSION = 0.1 # 10 cm minimo
    x_expansion = max(x_expansion, MIN_EXPANSION)
    y_expansion = max(y_expansion, MIN_EXPANSION)
    z_expansion = max(z_expansion, MIN_EXPANSION)

    # NUOVO: Ottimizza il volume per coprire TUTTA la depth disponibile senza andare oltre
    # z_max rappresenta il punto più lontano visto dalla camera
    # Vogliamo espandere il volume Z al massimo possibile (tutta la depth vista)
    # ma NON oltre z_max (altrimenti il volume eccede la percezione disponibile)

    # Il volume dovrebbe arrivare FINO a z_max (espandere al massimo)
    # ma se z_max + z_expansion supera z_max, significa che stiamo andando oltre
    # In questo caso, impostiamo z_expansion = 0 per non espandere oltre la depth vista

    # Calcola quanto possiamo espandere senza superare la depth massima vista
    # In pratica, vogliamo che z_max_finale = z_max (nessuna espansione oltre)
    # Quindi z_expansion sul lato destro (z_max) dovrebbe essere 0
    # Ma possiamo comunque espandere verso z_min

    pov_z_min = z_min - z_expansion
    pov_z_max = z_max  # NON espandiamo oltre z_max (la depth massima vista)

    print(f"   [POV] Volume Z ottimizzato: [{pov_z_min:.3f}m, {pov_z_max:.3f}m] (depth max vista: {z_max:.3f}m)")

    return {
        "x_min": x_min - x_expansion,
        "x_max": x_max + x_expansion,
        "y_min": y_min - y_expansion,
        "y_max": y_max + y_expansion,
        "z_min": pov_z_min,
        "z_max": pov_z_max
    }


def expand_bbox_for_search(bbox, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """
    MODIFICATO: Espande un bounding box in modo proporzionale alle sue dimensioni.
    
    Args:
        bbox: dict con chiavi x_min, x_max, y_min, y_max, z_min, z_max
        expansion_ratio: percentuale di espansione rispetto alle dimensioni (default 0.2 = 20%)

    Returns:
        dict: bounding box espanso
    """
    # Calcola dimensioni attuali
    x_size = bbox["x_max"] - bbox["x_min"]
    y_size = bbox["y_max"] - bbox["y_min"]
    z_size = bbox["z_max"] - bbox["z_min"]
    
    # Calcola espansione proporzionale per ogni asse
    x_expansion = x_size * expansion_ratio
    y_expansion = y_size * expansion_ratio
    z_expansion = z_size * expansion_ratio
    
    return {
        "x_min": bbox["x_min"] - x_expansion,
        "x_max": bbox["x_max"] + x_expansion,
        "y_min": bbox["y_min"] - y_expansion,
        "y_max": bbox["y_max"] + y_expansion,
        "z_min": bbox["z_min"] - z_expansion,
        "z_max": bbox["z_max"] + z_expansion
    }


def bbox_intersects_volume(bbox, volume):
    """
    NEW_MERGE (from ROS1): Controlla se un bounding box interseca un volume di ricerca.

    Args:
        bbox: bounding box dell'oggetto persistente
        volume: volume di ricerca espanso

    Returns:
        bool: True se c'è intersezione
    """
    if bbox is None:
        return False

    # Controlla se NON c'è sovrapposizione (then negate)
    no_overlap = (
        bbox["x_max"] < volume["x_min"] or bbox["x_min"] > volume["x_max"] or
        bbox["y_max"] < volume["y_min"] or bbox["y_min"] > volume["y_max"] or
        bbox["z_max"] < volume["z_min"] or bbox["z_min"] > volume["z_max"]
    )

    return not no_overlap


def bbox_centroid_in_volume(bbox, volume):
    """
    NUOVO: Controlla se il CENTROIDE di un bounding box è dentro un volume.

    Più conservativo di bbox_intersects_volume: oggetti grandi (es. divano)
    con solo un piccolo angolo nel POV non verranno eliminati se il loro
    centro è fuori dal volume.

    Args:
        bbox: bounding box dell'oggetto (dict con x_min, x_max, y_min, y_max, z_min, z_max)
        volume: volume di ricerca (dict con x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        bool: True se il centroide della bbox è completamente dentro il volume
    """
    if bbox is None:
        return False

    # Calcola il centroide della bbox
    centroid_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
    centroid_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    centroid_z = (bbox["z_min"] + bbox["z_max"]) / 2.0

    # Controlla se il centroide è dentro il volume
    is_inside = (
        volume["x_min"] <= centroid_x <= volume["x_max"] and
        volume["y_min"] <= centroid_y <= volume["y_max"] and
        volume["z_min"] <= centroid_z <= volume["z_max"]
    )

    return is_inside


def save_persistent_perceptions(node):
    """
    NEW_MERGE: Salva persistent_perceptions su file JSON.
    Mantiene persistenza tra esecuzioni del nodo.
    """
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "persistent_perception.json")

    data = []
    for obj in wm.persistent_perceptions:
        data.append({
            "label": obj.label,
            "description": obj.description,
            "color": obj.color,
            "material": obj.material,
            "shape": obj.shape,
            "bbox": obj.bbox
        })

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    # ROS2_MIGRATION
    msg = f"Salvati {len(data)} oggetti in persistent_perception.json"
    node.log_both('info', msg)
    labels = [obj["label"] for obj in data]
    node.log_both('info', f"Label oggetti salvati: {labels}")

    # Log dettagliato su file
    for obj_data in data:
        node.log_both('debug', f"  - {obj_data['label']}: {obj_data['description']}")


def save_uncertain_objects(node):
    """
    Salva uncertain_objects su file di testo.
    Traccia oggetti potenzialmente spostati o duplicati.
    """
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "uncertain_objects.txt")

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"OGGETTI INCERTI - Aggiornato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        if not node.uncertain_objects:
            f.write("Nessun oggetto incerto al momento.\n")
        else:
            f.write(f"Totale oggetti incerti: {len(node.uncertain_objects)}\n\n")

            for i, obj in enumerate(node.uncertain_objects, 1):
                f.write(f"{i}. {obj.label}\n")
                f.write(f"   Descrizione: {obj.description}\n")
                f.write(f"   Colore: {obj.color}\n")
                f.write(f"   Materiale: {obj.material}\n")

                if obj.bbox:
                    x_center = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
                    y_center = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
                    z_center = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0

                    x_size = obj.bbox["x_max"] - obj.bbox["x_min"]
                    y_size = obj.bbox["y_max"] - obj.bbox["y_min"]
                    z_size = obj.bbox["z_max"] - obj.bbox["z_min"]

                    f.write(f"   Posizione centro: X={x_center:.3f}, Y={y_center:.3f}, Z={z_center:.3f}\n")
                    f.write(f"   Dimensioni: {x_size:.3f}m x {y_size:.3f}m x {z_size:.3f}m\n")
                else:
                    f.write(f"   Bbox: NON DISPONIBILE\n")

                f.write("\n" + "-" * 80 + "\n\n")

    node.log_both('info', f"Salvati {len(node.uncertain_objects)} oggetti incerti in uncertain_objects.txt")



class ObjectManagerNode(Node):
    def __init__(self):
        super().__init__('object_description_listener_node')

        self.file_logger = module_logger
        self.file_logger.info("=== ObjectManagerNode Initialized ===")
        self.get_logger().info(f"Log salvato in: {log_file}")

        # Variabili di istanza
        self.exploration_mode = True  # Controllato da thread separato

        self.latest_bboxes = {}

        # Uncertain objects list
        self.uncertain_objects = []  

        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_standard = QoSProfile(depth=10)

        self.persistent_bbox_pub = self.create_publisher(MarkerArray, '/persistent_bbox', qos_latch)
        self.considered_volume_pub = self.create_publisher(MarkerArray, '/considered_volume', qos_standard)
        self.uncertain_bboxes_pub = self.create_publisher(MarkerArray, '/uncertain_object', qos_standard)

        self.create_subscription(Bbox3dArray, "/bbox_3d", self.bbox_callback, qos_standard)
        self.create_subscription(ObjectDescriptionArray, "/object_descriptions", self.description_callback, qos_standard)

        self._bbox_timer = self.create_timer(2.0, self.periodic_bbox_publisher)  

        exploration_thread = threading.Thread(target=self.wait_for_exploration_end, daemon=True)
        exploration_thread.start()


    def log_both(self, level, message):
        # Log on ROS
        if level == 'info':
            self.get_logger().info(message)
        elif level == 'warn':
            self.get_logger().warn(message)
        elif level == 'error':
            self.get_logger().error(message)
        elif level == 'debug':
            self.get_logger().debug(message)

        # Log on file 
        if level == 'info':
            self.file_logger.info(message)
        elif level == 'warn':
            self.file_logger.warning(message)
        elif level == 'error':
            self.file_logger.error(message)
        elif level == 'debug':
            self.file_logger.debug(message)

        # Print on console for info and warn
        if level in ['info', 'warn']:
            prefix = "[INFO] " if level == 'info' else "[WARN] "
            print(f"{prefix}{message}")


    def wait_for_exploration_end(self):

        self.log_both('warn', "=" * 60)
        self.log_both('warn', "[EXPLORATION] Il robot is in EXPLORATION.")
        input()  # Wait for user to press Enter

        self.log_both('warn', "=" * 60)
        self.log_both('warn', "[EXPLORATION] Exploration phase ENDED!")
        self.log_both('warn', "[TRACKING] Tracking mode ACTIVATED")
        self.log_both('warn', "=" * 60)
        print("\n\n\n")

        self.exploration_mode = False

        publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)

    def description_callback(self, msg):
        if not hasattr(self, 'tracking_step_counter'):
            self.tracking_step_counter = 0
        if not self.exploration_mode:
            self.tracking_step_counter += 1

        in_exploration = self.exploration_mode
        mode_str = "EXPLORATION" if in_exploration else f"TRACKING STEP {self.tracking_step_counter}"
        print(f"Mode: {mode_str}")

        # IMPORTANT: Do NOT return if msg.descriptions is empty!
        # An empty message means "I looked but saw nothing"
        # This is FUNDAMENTAL for removing objects in the POV
        

        current_perception_objects = []
        objects_modified = False
        matched_bboxes = []

        for description in msg.descriptions:
            label = description.label
            color = description.color
            material = description.material
            description_text = description.description

            print(f"1. Calculating embedding for description...")
            description_embedding = get_embedding(client, description_text)


            # Find the old key with only label
            old_key = create_object_key(label, "", "", "")

            if old_key not in self.latest_bboxes:
                print("No matching bbox found for this label.")
                continue

            # Found matching bbox
            bbox = self.latest_bboxes[old_key]["bbox"]
            matched_label = self.latest_bboxes[old_key]["label"]

            new_key = create_object_key(label, material, color, description_text)

            # Remove old key and add new key
            del self.latest_bboxes[old_key]
            self.latest_bboxes[new_key] = {
                "bbox": bbox,
                "label": label,
                "color": color,
                "material": material,
                "description": description_text
            }

            print(f"   Old key (solo label): {old_key}...")
            print(f"   New key (completa):   {new_key}...")
        
            matched_bboxes.append(bbox)

            already_seen = False

            # ======== EXPLORATION LOGIC ========
            if in_exploration:
                print(f"3. [EXPLORATION] Comparing with {len(wm.persistent_perceptions)} persistent objects using lost_similarity...")
                for obj in wm.persistent_perceptions:
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(client, obj.description)

                    if obj.embedding is None:
                        continue

                    # Use lost_similarity
                    similarity = lost_similarity(
                        world2vec,
                        label, obj.label,
                        color, obj.color,
                        material, obj.material,
                        description_embedding, obj.embedding
                    )
                    print("Comparing with persistent object:", obj.label)
                    if similarity > SIM_THRESHOLD:
                        print(f"Semantic match! Calculating spatial IoU...")
                        if obj.bbox is not None:
                            iou = compute_iou_3d(bbox, obj.bbox)
                            print(f"IoU: {iou:.3f} (threshold: {EXPLORATION_IOU_THRESHOLD})")

                            if iou < EXPLORATION_IOU_THRESHOLD:
                                print(f"IoU too low - objects considered different")
                                continue
                            else:
                                # Equal object found in the same position
                                already_seen = True
                                current_perception_objects.append(obj)
                                obj.bbox = bbox
                                print(f"FULL MATCH! '{label}' = '{obj.label}' (lost_sim={similarity:.3f}, IoU={iou:.3f})")
                                break

            # ======== Tracking ========
            else:
                print(f"3. [TRACKING STEP {self.tracking_step_counter}] Creating search volume proportional to object size...")
                # MODIFIED: Expanded volume proportionally (used only for debug info)
                search_volume = expand_bbox_for_search(bbox, VOLUME_EXPANSION_RATIO)

                print(f"4. [FIX] Semantic matching among ALL persistent objects ({len(wm.persistent_perceptions)} objects)...")
                best_match = None
                best_score = 0

                for obj in wm.persistent_perceptions:
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(client, obj.description)

                    if obj.embedding is None:
                        continue

                    # Use lost_similarity
                    similarity = lost_similarity(
                        world2vec,
                        label, obj.label,
                        color, obj.color,
                        material, obj.material,
                        description_embedding, obj.embedding
                    )
                    print(f"   Comparing with persistent object: '{obj.label}' (lost_sim={similarity:.3f})")
                    # FIX: Indicate whether the object is inside the search volume
                    is_in_volume = bbox_intersects_volume(obj.bbox, search_volume) if obj.bbox else False
                    print(f"      - Object bbox intersects search volume: {is_in_volume}")

                    if similarity > SIM_THRESHOLD and similarity > best_score:
                        best_score = similarity
                        best_match = obj

                # Se trovato match, aggiorna
                if best_match:
                    print(f"Best match found: '{best_match.label}' (score={best_score:.2f})")
                    already_seen = True

                    # Compute IoU
                    old_bbox = best_match.bbox
                    iou = compute_iou_3d(bbox, old_bbox)
                    print(f"IoU with existing position: {iou:.3f} (threshold: {TRACKING_IOU_THRESHOLD})")

                    # CASE 2: High IoU (>= threshold) → Same object, update bbox
                    if iou >= TRACKING_IOU_THRESHOLD:
                        print(f"CASE 2: Object recognized in the same position (IoU={iou:.3f})")
                        print(f"Updating bbox of '{best_match.label}'")

                        # Small movement logging
                        if iou < 0.9:  # Only if there has been a minimum movement
                            old_x = (old_bbox['x_min'] + old_bbox['x_max']) / 2.0
                            old_y = (old_bbox['y_min'] + old_bbox['y_max']) / 2.0
                            old_z = (old_bbox['z_min'] + old_bbox['z_max']) / 2.0
                            new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
                            new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
                            new_z = (bbox['z_min'] + bbox['z_max']) / 2.0


                        best_match.bbox = bbox
                        current_perception_objects.append(best_match)
                        objects_modified = True
                        
                    # CASE 3: Low IoU (< threshold) → Same object but moved FAR
                    else:
                        
                        distance = 0.0
                        if best_match.bbox:
                            old_x = (best_match.bbox['x_min'] + best_match.bbox['x_max']) / 2.0
                            old_y = (best_match.bbox['y_min'] + best_match.bbox['y_max']) / 2.0
                            old_z = (best_match.bbox['z_min'] + best_match.bbox['z_max']) / 2.0
                            new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
                            new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
                            new_z = (bbox['z_min'] + bbox['z_max']) / 2.0
                            distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)

                            print(f"\nOLD POSITION:")
                            print(f"      Center: X={old_x:.3f}, Y={old_y:.3f}, Z={old_z:.3f}")
                            print(f"\nNEW POSITION:")
                            print(f"     Center: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
                            print(f"\nMovement distance: {distance:.3f} meters")



                        # OBJECT MOVED FAR: Remove old position from persistent_perceptions
                        if best_match in wm.persistent_perceptions:
                            wm.persistent_perceptions.remove(best_match)
                            print(f"   🔄 MODIFICATION: '{best_match.label}' removed from old position (will be reinserted in the new one)")

                        # MODIFICATION: Add to uncertain_objects ONLY IF movement > 80 cm (0.8 meters)
                        if distance > 0.8:
                            if best_match not in self.uncertain_objects:
                                self.uncertain_objects.append(best_match)
                                print(f"   📝 '{best_match.label}' added to UNCERTAIN_OBJECTS (movement={distance:.3f}m > 0.8m)")
                                print(f"   → It will be saved in the uncertain_objects.txt file and published on /uncertain_object\n")
                        else:
                            print(f"{best_match.label} NOT added to UNCERTAIN_OBJECTS (movement={distance:.3f}m ≤ 0.8m)\n")

                        # Create new object with new position
                        updated_obj = Object(label, None, bbox, description_text, color, material)
                        updated_obj.embedding = description_embedding
                        wm.persistent_perceptions.append(updated_obj)
                        current_perception_objects.append(updated_obj)
                        print(f"MODIFICATION: '{label}' reinserted with new position in persistent_perceptions")
                        
                        objects_modified = True
                else:
                    print(f"No match found in the search volume")

            # If not already seen → add as a new object (always GREEN)
            if not already_seen:
                print(f"\nNEW OBJECT DETECTED!")
                # Create the new object
                new_obj = Object(label, None, bbox, description_text, color, material)
                new_obj.embedding = description_embedding

                # Always add to persistent_perceptions (GREEN bbox)
                wm.persistent_perceptions.append(new_obj)
                current_perception_objects.append(new_obj)
                objects_modified = True

                # Calculate bbox volume for the new object
                x_size = bbox["x_max"] - bbox["x_min"]
                y_size = bbox["y_max"] - bbox["y_min"]
                z_size = bbox["z_max"] - bbox["z_min"]
                volume = x_size * y_size * z_size

                mode_tag = "[ESPLORAZIONE]" if in_exploration else f"[TRACKING STEP {self.tracking_step_counter}]"
                print(f"{mode_tag} New object '{label}' added to persistent_perceptions (bbox volume: {volume:.3f} m³)")
                save_persistent_perceptions(self)

        # ======== MANAGEMENT OF OBJECTS NOT SEEN IN THE CURRENT FRAME ========
        # DELETION

        if not in_exploration:

            description_recieved = len(msg.descriptions) > 0

            objects_to_remove = []
            uncertain_to_remove = []

            if not description_recieved:
                if self.latest_bboxes:
                    detected_bboxes = [entry["bbox"] for entry in self.latest_bboxes.values()]
                    pov_volume = compute_pov_volume(detected_bboxes, VOLUME_EXPANSION_RATIO)

                    if pov_volume:
                        publish_pov_volume(self, pov_volume, self.considered_volume_pub)

                        for obj in wm.persistent_perceptions:
                            if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                                objects_to_remove.append(obj)
                                print(f"DELETION: '{obj.label}' is IN POV but NOT SEEN (zero detection) → Will be REMOVED from persistent_perceptions")

                            else:
                                print(f"'{obj.label}' not in POV (centroid) → ignored")

                        if self.uncertain_objects:
                            for uncertain_obj in self.uncertain_objects:
                                if uncertain_obj.bbox and bbox_centroid_in_volume(uncertain_obj.bbox, pov_volume):
                                    uncertain_to_remove.append(uncertain_obj)
                                    print(f"DELETION: '{uncertain_obj.label}' (ORANGE) is IN POV but NOT SEEN → Will be REMOVED from uncertain_objects")
                                else:
                                    print(f"'{uncertain_obj.label}' (ORANGE) not in POV (centroid) → ignored")

                        print(f"\nResult: {len(objects_to_remove)} persistent objects to remove, {len(uncertain_to_remove)} uncertain objects to remove")
                    else:
                        print(f"Unable to calculate POV volume from depth (invalid bboxes)")
                else:
                    print(f"No bbox available from depth camera - unable to calculate POV")
                    print(f"This is normal if detection_node hasn't published bbox yet")

            elif description_recieved:
                detected_bboxes = matched_bboxes
                pov_volume = compute_pov_volume(detected_bboxes, VOLUME_EXPANSION_RATIO)

                if pov_volume:
                    publish_pov_volume(self, pov_volume, self.considered_volume_pub)


                    for obj in wm.persistent_perceptions:
                        if obj not in current_perception_objects:
                            if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                                objects_to_remove.append(obj)
                            else:
                                continue
                        else:
                            continue
                else:
                    print(f"Unable to calculate POV volume from depth (invalid bboxes)")

            if objects_to_remove:
                print(f"\nDELETING OBJECTS FROM PERSISTENT_PERCEPTIONS - TRACKING STEP {self.tracking_step_counter}")

                for obj in objects_to_remove:
                    wm.persistent_perceptions.remove(obj)

                objects_modified = True
                save_persistent_perceptions(self)
            else:
                print(f"✓ Nessun oggetto da rimuovere in questo step")

            # Verifica uncertain_objects per rimozione definitiva
            print(f"\n{'─'*60}")
            print(f"[TRACKING STEP {self.tracking_step_counter}] Verifica oggetti incerti con POV VOLUME...")
            print(f"{'─'*60}")
            if description_recieved and pov_volume:
                for uncertain_obj in self.uncertain_objects:
                    # MODIFIED: Check if the CENTROID is inside the POV (less aggressive)
                    if uncertain_obj.bbox and bbox_centroid_in_volume(uncertain_obj.bbox, pov_volume):
                        # If POV contains the old position → I VERIFIED that zone → REMOVE
                        # Doesn't matter if I see something or not - I looked there, no need to keep it
                        uncertain_to_remove.append(uncertain_obj)
                        print(f"   ✅ DELETION: '{uncertain_obj.label}' (ORANGE) in POV - zone VERIFIED → Will be REMOVED from uncertain_objects")
                    else:
                        print(f"   ⏭️ '{uncertain_obj.label}' (ORANGE) is OUTSIDE the POV (centroid) → ignored")

            if uncertain_to_remove:
                print(f"\nDELETION OF UNCERTAIN OBJECTS FROM UNCERTAIN_OBJECTS (VERIFIED ZONE) - TRACKING STEP {self.tracking_step_counter}")
                for uncertain_obj in uncertain_to_remove:
                    self.uncertain_objects.remove(uncertain_obj)

                objects_modified = True
            else:
                if self.uncertain_objects:
                    print(f"No uncertain object to remove in this step")
                else:
                    print(f"No uncertain object present")

        if objects_modified and not in_exploration:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            save_uncertain_objects(self)



    def bbox_callback(self, msg):
        """
        Callback for 3D bounding boxes.
        Temporarily stores bbox by label, waiting for semantic matching.
        """

        self.latest_bboxes = {}
        for box in msg.boxes:
            # Normalize values (ensure min < max)
            x_min, x_max = min(box.x_min, box.x_max), max(box.x_min, box.x_max)
            y_min, y_max = min(box.y_min, box.y_max), max(box.y_min, box.y_max)
            z_min, z_max = min(box.z_min, box.z_max), max(box.z_min, box.z_max)

            # Calculate dimensions and volume
            x_size = x_max - x_min
            y_size = y_max - y_min
            z_size = z_max - z_min
            volume = x_size * y_size * z_size

            bbox_data = {
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
                "z_min": z_min, "z_max": z_max
            }
            temp_key = create_object_key(box.label, "", "", "")

            self.latest_bboxes[temp_key] = {
                "bbox": bbox_data,
                "label": box.label,
                "color": "",
                "material": "",
                "description": ""
            }


    def periodic_bbox_publisher(self):
        """
        NEW_MERGE (from ROS1): Periodically publishes persistent and uncertain bounding boxes.
        Keeps markers visible on RViz during tracking.
        """
        # Publish only if NOT in exploration mode
        if not self.exploration_mode:
            if len(wm.persistent_perceptions) > 0:
                self.log_both('debug', f"[PERIODIC] Periodic publishing of bbox ({len(wm.persistent_perceptions)} objects)")
                publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)

            # NEW: Periodically publishes and saves uncertain objects
            if len(self.uncertain_objects) > 0:
                self.log_both('debug', f"[PERIODIC] Periodic publishing of uncertain objects ({len(self.uncertain_objects)} objects)")
                publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
                save_uncertain_objects(self)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"OBJECT MANAGER closed{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        try:
            node.close_json_writers()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        tracking_logger.close()  


if __name__ == "__main__":
    main()