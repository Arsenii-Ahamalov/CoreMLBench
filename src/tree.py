class Tree:
    def __init__(self,depth):
        self.depth = depth
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.is_leaf = True
        self.value = None
        self.current_depth = 0
