import itertools
from functools import reduce
import operator

class Initializer():

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