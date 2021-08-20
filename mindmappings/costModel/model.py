class Model:
    """
        Cost model that interfaces between optimizer and oracle cost.

        Base Class.
    """

    def __init__(self, **kwargs):
        """ 
            Initialize the cost model.

            Input
            -------------
            arch : The architectural specifications of the accelerator (that cost model understands)
            algorithm : The choice of algorithm (based on how cost model understands)
            problem : Paramterization of the algorithm (e.g., DNN Layer Shape)

            Returns
            -------------
            None

        """
        self.algorithm = kwargs.get('algorithm', None)
        self.arch = kwargs.get('architecture', None)
        self.problem = kwargs.get('problem', None)
        self.__dict__.update(kwargs)

    def getMapping(self, **kwargs):
        """ 
            Gives a random valid mapping.

            Input
            -------------
            None

            Returns
            -------------
            mapping: A mapping object, which needs to be understood by the costFn function.

        """
        raise NotImplementedError()

    def costFn(self, mapping, **kwargs):
        """ 
            Return the cost of the mapping

            Input
            -------------
            mapping: The mapping for which the cost is to be returned

            Returns
            -------------
            cost: Returns the cost of the function. Additional argumetns can be used to
                    decide the metric of the cost.

        """
        raise NotImplementedError()

    def getMapCost(self, **kwargs):
        """ 
            Choose a random mapping and return its cost.

            Input
            -------------
            kwargs: Any argument that the subclass may want for cost.

            Returns
            -------------
            cost
            mapping

        """
        raise NotImplementedError()

    def getOracleCost(self, **kwargs):
        """ 
            Get the theoretical lower bound cost. This is useful to normalize the dataset across
            different problems.

            Input
            -------------
            kwargs: Any argument that the subclass may want for cost.

            Returns
            -------------
            cost

        """
        raise NotImplementedError()

    def getDomain(self):
        """ 
            Get the domain of each programmable attribute.

            Input
            -------------
            None

            Returns
            -------------
            domain

        """
        raise NotImplementedError()

    def getProjection(self, mapping, **kwargs):
        """ 
            Returns the projection of the mapping to valid map space.

            Input
            -------------
            mapping (can be valid or invalid)

            Returns
            -------------
            mapping (valid)

        """
        raise NotImplementedError()

    def getInputVector(self, mapping, **kwargs):
        """
        
            Returns an input vector for training: <mapspace_id + mapping as a vector >. This can be fed to an MLP.
            
            Input
            -------------
            mapping  

            Returns
            -------------
            a list of [*mapspace_id, *flattened_mapping]

            """
        raise NotImplementedError()

    def getOutputCost(self, outvector, **kwargs):
        """
        
            Returns the cost we are optimizing, given the prediction from a DNN.
            
            See Section 4 of the paper for the output cost representation.
            
            Input
            -------------
            outvector (a vector of meta-stats)

            Returns
            -------------
            cost (float)

            """
        raise NotImplementedError()

    # Optional
    def getMapSpaceSize(self):
        """ 
            Returns the Size of the Map Space

            Input
            -------------
            None

            Returns
            -------------
            Size (int)

        """
        raise NotImplementedError()