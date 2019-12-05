import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            converged = True
            #loop over dataset and predict; if wrong prediction, update weight and set flag to True
            for x, y in dataset.iterate_once(1):
                y_predicted = self.get_prediction(x)
                if y_predicted != nn.as_scalar(y):
                    converged = False
                    self.w.update(x, nn.as_scalar(y))
            if converged:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_lsize = 250
        self.learning_rate = -.01
        self.batch_size = 200

        self.m_1 = nn.Parameter(1, self.hidden_lsize)
        self.b_1 = nn.Parameter(1, self.hidden_lsize)

        self.m_2 = nn.Parameter(self.hidden_lsize, 1)
        self.b_2 = nn.Parameter(1, 1)

        #self.m_3 = nn.Parameter(self.hidden_lsize, 1)
        #self.b_3 = nn.Parameter(1, 1)
 

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xm_1 = nn.Linear(x, self.m_1)
        layer_1 = nn.AddBias(xm_1, self.b_1)
        
        non_lin_1 = nn.ReLU(layer_1)

        xm_2 = nn.Linear(non_lin_1, self.m_2)
        layer_2 = nn.AddBias(xm_2, self.b_2)

        #non_lin_2 = nn.ReLU(layer_2)

        #xm_3 = nn.Linear(non_lin_2, self.m_3)
        #layer_3 = nn.AddBias(xm_3, self.b_3)

        #non_lin_3 = nn.ReLU(layer_3)
    
        return layer_2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_size):
            loss_node = self.get_loss(x, y)
            if nn.as_scalar(loss_node) <= .02:
                break
            else:
                grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2 = nn.gradients(loss_node, [self.m_1, self.b_1, self.m_2, self.b_2])

                self.m_1.update(grad_wrt_m1, self.learning_rate)
                self.b_1.update(grad_wrt_b1, self.learning_rate)
                self.m_2.update(grad_wrt_m2, self.learning_rate)
                self.b_2.update(grad_wrt_b2, self.learning_rate)
                #self.m_3.update(grad_wrt_m3, self.learning_rate)
                #self.b_3.update(grad_wrt_b3, self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_lsize = 250
        self.learning_rate = -.4
        self.batch_size = 200

        self.m_1 = nn.Parameter(784, self.hidden_lsize)
        self.b_1 = nn.Parameter(1, self.hidden_lsize)

        self.m_2 = nn.Parameter(self.hidden_lsize, 10)
        self.b_2 = nn.Parameter(1, 10)

        #self.m_3 = nn.Parameter(self.hidden_lsize, 1)
        #self.b_3 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        xm_1 = nn.Linear(x, self.m_1)
        layer_1 = nn.AddBias(xm_1, self.b_1)
        
        non_lin_1 = nn.ReLU(layer_1)

        xm_2 = nn.Linear(non_lin_1, self.m_2)
        layer_2 = nn.AddBias(xm_2, self.b_2)

        #non_lin_2 = nn.ReLU(layer_2)

        #xm_3 = nn.Linear(non_lin_2, self.m_3)
        #layer_3 = nn.AddBias(xm_3, self.b_3)

        #non_lin_3 = nn.ReLU(layer_3)
    
        return layer_2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_size):
            loss_node = self.get_loss(x, y)
            if dataset.get_validation_accuracy() >= .975:
                break
            else:
                grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2 = nn.gradients(loss_node, [self.m_1, self.b_1, self.m_2, self.b_2])

                self.m_1.update(grad_wrt_m1, self.learning_rate)
                self.b_1.update(grad_wrt_b1, self.learning_rate)
                self.m_2.update(grad_wrt_m2, self.learning_rate)
                self.b_2.update(grad_wrt_b2, self.learning_rate)
                #self.m_3.update(grad_wrt_m3, self.learning_rate)
                #self.b_3.update(grad_wrt_b3, self.learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_lsize = 250
        self.learning_rate = -.01
        self.batch_size = 200

    
        m_1f = nn.Parameter(self.num_chars, self.hidden_lsize)
        b_1f = nn.Parameter(1, self.hidden_lsize)
        m_2f = nn.Parameter(self.hidden_lsize, 10)
        b_2f = nn.Parameter(1, self.hidden_lsize)
        xm_1 = nn.Linear(x, m_1f)
        layer_1 = nn.AddBias(xm_1, b_1f)
        non_lin_1 = nn.ReLU(layer_1)
        xm_2 = nn.Linear(non_lin_1, m_2f)
        self.f_initial = nn.AddBias(xm_2, b_2f)
        

        self.w = nn.Parameter(self.batch_size, self.num_chars)
        self.w_hidden = nn.Parameter(self.hidden_lsize, 1)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        #first summarize xs into vector
        z_curr = nn.DotProduct(self.w, xs[0])
        h_curr = self.f_initial(xs[0])

        for i in range(1, len(xs)):
            z_curr = nn.Add(nn.Linear(xs[i], self.w), nn.Linear(h_curr, self.w_hidden))
            h_curr = z_curr
        return z_curr

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_size):
            loss_node = self.get_loss(x, y)
            if dataset.get_validation_accuracy() >= .81:
                break
            else:
                grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2 = nn.gradients(loss_node, [self.m_1, self.b_1, self.m_2, self.b_2])

                self.m_1.update(grad_wrt_m1, self.learning_rate)
                self.b_1.update(grad_wrt_b1, self.learning_rate)
                self.m_2.update(grad_wrt_m2, self.learning_rate)
                self.b_2.update(grad_wrt_b2, self.learning_rate)

