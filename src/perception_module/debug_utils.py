from datetime import datetime

class TrackingLogger:
    """Class for logging with readable recaps in a single file"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.log_file = open(filepath, 'w', buffering=1, encoding='utf-8')

        # Write informative header
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write("TRACKING LOG - DETAILED OPERATIONS RECAP\n")
        self.log_file.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n\n")

    def write_readable(self, message):
        """Write a readable message to the log"""
        self.log_file.write(f"{message}\n")
        self.log_file.flush()

    def log_exploration_end(self, objects_list):
        """Write complete recap at the end of exploration"""
        self.log_file.write("\n" + "=" * 80 + "\n")
        self.log_file.write("END OF EXPLORATION PHASE\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Total objects detected: {len(objects_list)}\n\n")

        if objects_list:
            self.log_file.write("OBJECTS IN THE SCENE:\n")
            for i, obj in enumerate(objects_list, 1):
                desc = obj.description if obj.description else "no description"
                color = obj.color if obj.color else "unknown color"
                material = obj.material if obj.material else "unknown material"
                shape = obj.shape if obj.shape else "unknown shape"
                self.log_file.write(f"  {i}. {obj.label} ({color}, {material}, {shape}) - {desc}\n")
        else:
            self.log_file.write("No objects detected.\n")

        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()

    def log_tracking_step_start(self, step_number):
        """Start a new tracking step"""
        self.log_file.write("\n" + "─" * 80 + "\n")
        self.log_file.write(f"TRACKING STEP {step_number}\n")
        self.log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("─" * 80 + "\n")
        self.log_file.flush()

    def log_deletion(self, obj_label, reason, bbox=None, step_number=None, obj=None, case_type=None):
        """Log deletion with readable sentence"""
        if obj:
            desc = obj.description if obj.description else "no description"
            color = obj.color if obj.color else "unknown color"
            material = obj.material if obj.material else "unknown material"
            shape = obj.shape if obj.shape else "unknown shape"
        else:
            desc = "no description available"
            color = "unknown color"
            material = "unknown material"
            shape = "unknown shape"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • DELETED{case_info}: '{obj_label}' ({color}, {material}, {shape}) - {desc}"
        self.write_readable(message)

    def log_position_change(self, obj_label, old_bbox, new_bbox, distance, step_number=None, obj=None, case_type=None):
        """Log position change with readable sentence"""
        if obj:
            desc = obj.description if obj.description else "no description"
            color = obj.color if obj.color else "unknown color"
            material = obj.material if obj.material else "unknown material"
            shape = obj.shape if obj.shape else "unknown shape"
        else:
            desc = "no description available"
            color = "unknown color"
            material = "unknown material"
            shape = "unknown shape"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • UPDATED{case_info}: '{obj_label}' ({color}, {material}, {shape}) - {desc} (moved {distance:.2f}m)"
        self.write_readable(message)

    def log_uncertain_added(self, obj_label, reason, distance, bbox=None, step_number=None, obj=None, case_type=None):
        """Log addition to uncertain with readable sentence"""
        if obj:
            desc = obj.description if obj.description else "no description"
            color = obj.color if obj.color else "unknown color"
            material = obj.material if obj.material else "unknown material"
            shape = obj.shape if obj.shape else "unknown shape"
        else:
            desc = "no description available"
            color = "unknown color"
            material = "unknown material"
            shape = "unknown shape"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • UNCERTAIN OBJECT{case_info}: '{obj_label}' ({color}, {material}, {shape}) - {desc} (displacement {distance:.2f}m, to be verified)"
        self.write_readable(message)

    def log_new_object(self, obj, case_type=None):
        """Log addition of a new object"""
        desc = obj.description if obj.description else "no description available"
        color = obj.color if obj.color else "unknown color"
        material = obj.material if obj.material else "unknown material"
        shape = obj.shape if obj.shape else "unknown shape"

        case_info = f" [{case_type}]" if case_type else ""
        message = f"  • NEW OBJECT{case_info}: '{obj.label}' ({color}, {material}, {shape}) - {desc}"
        self.write_readable(message)

    def close(self):
        """Close the log file"""
        self.log_file.write("\n" + "=" * 80 + "\n")
        self.log_file.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.close()

