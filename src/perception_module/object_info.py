class Object:
    """
    Class that represent each detected object with its properties.
    Attributes:
        label (str): The label of the object.
        centroid (tuple): The centroid coordinates of the object.
        bbox (tuple, optional): The bounding box of the object. Defaults to None.
        description (str, optional): A textual description of the object. Defaults to "".
        color (str, optional): The color of the object. Defaults to "".
        material (str, optional): The material of the object. Defaults to "".
        shape (str, optional): The shape of the object. Defaults to "".
    """

    def __init__(self, label, centroid, bbox=None, description="", color="",material="",shape=""):
        self.label = label
        self.centroid = centroid
        self.bbox = bbox
        self.description = description
        self.color= color
        self.material=material
        self.shape=shape