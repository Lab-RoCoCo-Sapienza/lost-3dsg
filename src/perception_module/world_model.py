class WorldModel:
    """
    WorldModel singleton class to manage perceptions.
    Maintains actual and persistent perceptions of objects.
    In actual perceptions, objects detected in the current frame are stored.
    In persistent perceptions, objects that have been consistently detected over time are stored.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._actual_perceptions = []
            cls._instance._persistent_perceptions = []
        return cls._instance
    
    @property
    def actual_perceptions(self):
        return self._actual_perceptions
    
    @property
    def persistent_perceptions(self):
        return self._persistent_perceptions
    
    def add_actual_perception(self, obj):
        self._actual_perceptions.append(obj)
    
    def add_persistent_perception(self, obj):
        self._persistent_perceptions.append(obj)
    
    def clear_actual_perceptions(self):
        self._actual_perceptions.clear()

wm = WorldModel()