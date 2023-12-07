import numpy as np
import os
from Author import Author

class MultinomialLogisticRegression():
    """
    This class performs multinomial logistic regression using batch gradient descent,
    minibatch gradient descent or stochastic gradient descent.
    """

    # ---------------- Hyperparameters -------------------
    LEARNING_RATE = 0.1 # The learning rate
    CONVERGENCE_MARGIN = 0.001 # The convergence criterion
    MINIBATCH_SIZE = 100 # Minibatch size (only for minibatch gradient descent)
    LAMBDA = 0.001 # Parameter in regularization

    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array
        @param y The labels as a DATAPOINT array
        @param theta Our weights we want to optimize for our model
        Code from: https://github.com/filiporestav/DD1418/blob/main/NER/BinaryLogisticRegression.py
        """
        # Number of datapoints
        self.DATAPOINTS = len(x)

        # Number of features
        self.FEATURES = len(x[0]) + 1 # +1 because of "dummy feature"

        # Encoding of data points (as a DATAPOINTS x FEATURES size array)
        # We add the first column with ones since we have the "bias" term.
        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

        # Correct labels for the datapoints.
        self.y = np.array(y)

        # Number of classes in our data.
        self.CLASSES = len(np.unique(self.y))

        # The weights we want to learn in the training phase (each row associated with a class, and each column with a feature)
        self.theta = np.random.uniform(-1, 1, (self.FEATURES, self.CLASSES))

        # The current gradient
        self.gradient = np.zeros(self.FEATURES)

        # Change if regularization or not
        self.regularization = False

    # --------------------------------------------------

    def softmax(self, x):
        """
        The softmax function. An exponent normalization function which converts
        oour scores into positives and turn them into probabilities.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def score(self, x):
        """
        The score of the datapoint.
        Returns a vector x*theta which contains the weights for each of the k classes.
        """
        return x @ self.theta
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        Computes the cross-entropy loss between the predicted probabilities and the true labels.

        @param y_pred Predicted probabilities (output of softmax)
        @param y_true True labels (one-hot encoded)
        """
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss
    
    def compute_gradient(self, x, y_true, y_pred):
        """
        Computes the gradient of the cross-entropy loss with respect to the weights.

        @param x Input data
        @param y_true True labels (one-hot encoded)
        @param y_pred Predicted probabilities (output of softmax)
        """
        gradient = x.T @ (y_pred - y_true) / len(y_true)
        return gradient
    
    def gradient_descent(self, num_epochs=500, batch_size=None, stochastic=True):
        """
        Train the model using gradient descent.

        @param num_epochs Number of training epochs
        @param batch_size Batch size for minibatch gradient descent
        @param stochastic If True, use stochastic gradient descent
        """
        for epoch in range(num_epochs):
            if stochastic:
                # Shuffle data for stochastic gradient descent
                indices = np.random.permutation(self.DATAPOINTS)
                self.x = self.x[indices]
                self.y = self.y[indices]

            for batch_start in range(0, self.DATAPOINTS, batch_size or self.DATAPOINTS):
                batch_end = batch_start + batch_size if batch_size else self.DATAPOINTS
                x_batch = self.x[batch_start:batch_end]
                y_batch = self.y[batch_start:batch_end]

                # Forward pass
                scores = self.score(x_batch)
                probabilities = self.softmax(scores)

                # One-hot encode the true labels
                y_true_one_hot = np.eye(self.CLASSES)[y_batch]

                # Compute the loss
                loss = self.cross_entropy_loss(probabilities, y_true_one_hot)

                # Backward pass - Compute the gradient
                self.gradient = self.compute_gradient(x_batch, y_true_one_hot, probabilities)

                # Update weights using gradient descent
                self.theta -= self.LEARNING_RATE * self.gradient

            # Check for convergence
            if np.linalg.norm(self.gradient) < self.CONVERGENCE_MARGIN:
                print(f"Converged after {epoch + 1} epochs.")
                break

            # Print loss for monitoring
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        print(self.theta)


def main():
    x, y = preprocess_data("books")
    logisticReg = MultinomialLogisticRegression(x, y)
    logisticReg.gradient_descent()

def preprocess_data(data_path):
        """
        Preprocess the data from the given directory.

        @param data_path Path to the directory containing author text files.
        """
        authors = []
        x = []
        y = []

        for author_name in os.listdir(data_path):

            # Check if author already exist, if not create it
            author = None
            for author_instance in authors: # Check if author exist already
                if author_name == author_instance.get_name():
                    author = author_instance
            if author == None: # Check if author did not exist, if then create it
                author = Author(author_name)
                authors.append(author)

            author_path = os.path.join(data_path, author_name)
            for filename in os.listdir(author_path):
                with open(os.path.join(author_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    author.update_data(text)
                    author.update_stylometric_features()

                    # Extract stylometric features
                    features = [
                        author.average_word_length,
                        author.average_sentence_length,
                        author.unique_word_ratio,
                        author.function_words_ratio,
                        # Add more features as needed
                    ]

                    x.append(features)
                    y.append(author_name)

        # Convert y to numerical labels
        unique_authors = list(set(y))
        author_to_label = {author: i for i, author in enumerate(unique_authors)}
        y = [author_to_label[author] for author in y]

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y

if __name__== "__main__":
    main()