
class SetupAttacks:

    """
        This class holds all necessary parameters to create a whitebox attacks.
        
        constructor args:
        -----------------
        
        sess = Current session in which the checkpoint is restored
        
        input_tensors = input placeholder or signature of the model for which the attack is being attacked
        
        output_tensor = output placeholder of the model being attacked
        
        logits = logits are variables before they are passed ot the softmax function. Here they represent an entire model because it called functions from previous layers to get the final value
        
        Notes on logits: logits contain basically the information of the whole network as the logitsare the final layer (without softmax)
        so the TFClassifier uses the logits instead of output from the softmax, "output" in the model for creating adversarial examples        
        
        graph: graph created from the current model.
    """


    def __init__(self, sess, input_placeholder, output_placeholder, logits, loss, graph):
        self.sess = sess
        self.input_placeholder = input_placeholder
        self.output_placeholder = output_placeholder
        self.logits = logits
        self.loss = loss
        self.current_graph = graph

