from datetime import datetime

class TrackingLogger:
    """Classe per logging con recap leggibili in un singolo file"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.log_file = open(filepath, 'w', buffering=1, encoding='utf-8')

        # Scrivi header informativo
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write("TRACKING LOG - RECAP DETTAGLIATO DELLE OPERAZIONI\n")
        self.log_file.write(f"Sessione avviata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n\n")

    def write_readable(self, message):
        """Scrive un messaggio leggibile nel log"""
        self.log_file.write(f"{message}\n")
        self.log_file.flush()

    def log_exploration_end(self, objects_list):
        """Scrive il recap completo alla fine dell'esplorazione"""
        self.log_file.write("\n" + "=" * 80 + "\n")
        self.log_file.write("FINE FASE ESPLORAZIONE\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Oggetti totali rilevati: {len(objects_list)}\n\n")

        if objects_list:
            self.log_file.write("OGGETTI NELLA SCENA:\n")
            for i, obj in enumerate(objects_list, 1):
                desc = obj.description if obj.description else "nessuna descrizione"
                color = obj.color if obj.color else "colore sconosciuto"
                material = obj.material if obj.material else "materiale sconosciuto"
                shape = obj.shape if obj.shape else "forma sconosciuta"
                self.log_file.write(f"  {i}. {obj.label} ({color}, {material}, {shape}) - {desc}\n")
        else:
            self.log_file.write("Nessun oggetto rilevato.\n")

        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()

    def log_tracking_step_start(self, step_number):
        """Inizia un nuovo tracking step"""
        self.log_file.write("\n" + "─" * 80 + "\n")
        self.log_file.write(f"TRACKING STEP {step_number}\n")
        self.log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("─" * 80 + "\n")
        self.log_file.flush()

    def log_deletion(self, obj_label, reason, bbox=None, step_number=None, obj=None, case_type=None):
        """Registra eliminazione con frase leggibile"""
        if obj:
            desc = obj.description if obj.description else "nessuna descrizione"
            color = obj.color if obj.color else "colore sconosciuto"
            material = obj.material if obj.material else "materiale sconosciuto"
            shape = obj.shape if obj.shape else "forma sconosciuta"
        else:
            desc = "nessuna descrizione disponibile"
            color = "colore sconosciuto"
            material = "materiale sconosciuto"
            shape = "forma sconosciuta"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • ELIMINATO{case_info}: '{obj_label}' ({color}, {material}, {shape}) - {desc}"
        self.write_readable(message)

    def log_position_change(self, obj_label, old_bbox, new_bbox, distance, step_number=None, obj=None, case_type=None):
        """Registra cambiamento posizione con frase leggibile"""
        if obj:
            desc = obj.description if obj.description else "nessuna descrizione"
            color = obj.color if obj.color else "colore sconosciuto"
            material = obj.material if obj.material else "materiale sconosciuto"
            shape = obj.shape if obj.shape else "forma sconosciuta"
        else:
            desc = "nessuna descrizione disponibile"
            color = "colore sconosciuto"
            material = "materiale sconosciuto"
            shape = "forma sconosciuta"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • AGGIORNATO{case_info}: '{obj_label}' ({color}, {material}, {shape}) - {desc} (spostato di {distance:.2f}m)"
        self.write_readable(message)

    def log_uncertain_added(self, obj_label, reason, distance, bbox=None, step_number=None, obj=None, case_type=None):
        """Registra aggiunta a uncertain con frase leggibile"""
        if obj:
            desc = obj.description if obj.description else "nessuna descrizione"
            color = obj.color if obj.color else "colore sconosciuto"
            material = obj.material if obj.material else "materiale sconosciuto"
            shape = obj.shape if obj.shape else "forma sconosciuta"
        else:
            desc = "nessuna descrizione disponibile"
            color = "colore sconosciuto"
            material = "materiale sconosciuto"
            shape = "forma sconosciuta"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • OGGETTO INCERTO{case_info}: '{obj_label}' ({color}, {material}, {shape}) - {desc} (spostamento {distance:.2f}m, da verificare)"
        self.write_readable(message)

    def log_new_object(self, obj, case_type=None):
        """Registra l'aggiunta di un nuovo oggetto"""
        desc = obj.description if obj.description else "nessuna descrizione disponibile"
        color = obj.color if obj.color else "colore sconosciuto"
        material = obj.material if obj.material else "materiale sconosciuto"
        shape = obj.shape if obj.shape else "forma sconosciuta"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • NUOVO OGGETTO{case_info}: '{obj.label}' ({color}, {material}, {shape}) - {desc}"
        self.write_readable(message)

    def close(self):
        """Chiude il file di log"""
        self.log_file.write("\n" + "=" * 80 + "\n")
        self.log_file.write(f"Sessione terminata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.close()

