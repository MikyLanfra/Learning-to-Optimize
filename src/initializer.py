import itertools, torch, operator
import numpy as np
from torch.distributions import Dirichlet
from functools import reduce
from skimage.transform import rotate

class Param_Initializer():

    def __init__(self, optimizee_class, optimizee_kwargs:dict[str, list]={}):
        self.cls = optimizee_class
        self.optims = []
        self.attr = optimizee_kwargs
    
    def get_attr(self, attrs:dict[str, list]):
        """
        Returns the attributes of the class.
        
        Returns:
            list: List of attributes of the class.
        """
        self.attr = attrs
    
    def initialize(self):
        """
        Initializes the optimizee class and returns a list of instances.
        
        Returns:
            list: List of instances of the optimizee class.
        """
        # Generate all combinations of parameter values using itertools.product
        self.optims = []
        param_combinations = list(itertools.product(*self.attr.values()))
        
        # Create instances of the optimizee class for each combination
        for combination in param_combinations:
            # Create a dictionary of the current parameter values with the parameter names
            param_dict = dict(zip(self.attr.keys(), combination))
            # Create an instance of the optimizee class using the parameters
            self.optims.append(self.cls(**param_dict))
        
        return self.optims

    def get_num_optims(self):
        """
        Returns the number of optimizer instances that would be created 
        based on the current parameter combinations.

        Returns:
            int: Number of possible optimizer instances.
        """
        if not self.attr:
            return 0
        return reduce(operator.mul, (len(v) for v in self.attr.values()), 1)      

class Data_Initializer():

    def __init__(self, optimizee_class,optimizee_kwargs:dict[str,list]={}, distribution=Dirichlet(torch.tensor([1.0, 1.0])),num_optims:int=1, subset_size:int=1):
        self.cls = optimizee_class
        self.optims = []
        self.num_optims = num_optims
        self.attr = optimizee_kwargs
        self.distribution = distribution
        self.subset_size = subset_size
        self.indices = []

    def get_data(self, X, y):
        """
        Sets the data for the optimizee class.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Output data.
        """
        self.attr["X"] = X
        self.attr["y"] = y

    def initialize(self):
        if "X" not in self.attr.keys():
            raise ValueError("X is not in the attributes of the optimizee class")
        else:
            self.X = self.attr["X"]
            self.y = self.attr["y"]
            max_val = self.attr["X"].shape[0]

        if len(self.indices)==0:
            for i in range(self.num_optims):
                indexes = self.distribution.sample((self.subset_size,))[:, 0]
                indexes = (indexes * max_val).int().numpy()
                indexes = indexes.tolist()
                self.indices.append(indexes)

        self.optims = []
        for i in range(self.num_optims):
            params = self.attr.copy()
            params["X"] = self.X[self.indices[i]]
            params["y"] = self.y[self.indices[i]]
            self.optims.append(self.cls(**params))
        return self.optims
    
    def get_num_optims(self):
        return self.num_optims
    
class LabelPoison_Initializer():

    def __init__(self, optimizee_class,optimizee_kwargs:dict[str,list]={},num_optims:int=1, poisoning_rate:float=0.1):
        self.cls = optimizee_class
        self.optims = []
        self.num_optims = num_optims
        self.attr = optimizee_kwargs
        self.poisoning_rate = poisoning_rate
        self.indices = []

    def get_data(self, X, y):
        """
        Sets the data for the optimizee class.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Output data.
        """
        self.attr["X"] = X
        self.attr["y"] = y

    def initialize(self):
        self.optims = []
        for i in range(self.num_optims):
            params = self.attr.copy()
            params["y"] = self.poison_label()
            self.optims.append(self.cls(**params))
        return self.optims
    
    def get_num_optims(self):
        return self.num_optims
    
    def poison_label(self):
        y = self.attr["y"]
        y_current = y.clone()
        count_poisoned = 0
        for i in range(len(y)):
            if torch.rand(1).item() < self.poisoning_rate:
                index = torch.randint(0, len(y_current[i]))
                y_current[i] = torch.zeros_like(y_current[i])
                y_current[i][index] = 1.0
                count_poisoned += 1
        print(f"Poisoned {count_poisoned} labels out of {len(y)}")
        return y_current

class Rotation_Initializer():
    def __init__(self, optimizee_class,optimizee_kwargs:dict[str,list]={}, rotation_angles:list=[0]):
        self.cls = optimizee_class
        self.optims = []
        self.attr = optimizee_kwargs
        self.get_num_optims = len(rotation_angles)
        self.rotation_angles = rotation_angles
        self.X_rotated_list = []

    def get_data(self, X, y):
        """
        Sets the data for the optimizee class.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Output data.
        """
        self.attr["X"] = X
        self.attr["y"] = y

    def initialize(self):
        if len(self.optims)==0:
            X = self.attr["X"]
            X_rotated = np.zeros_like(X)
            for i in range(len(X)):
                X_rotated[i]=rotate(X.reshape(8, 8), angle=self.rotation_angles[i]).flatten()
            self.X_rotated_list.append(torch.tensor(X_rotated, dtype=torch.float32))
            

        self.optims=[]
        for i in range(self.num_optims):
            params = self.attr.copy()
            params["X"] = self.X_rotated_list[i]
            self.optims.append(self.cls(**params))
        return self.optims
    
    def get_num_optims(self):
        return self.num_optims