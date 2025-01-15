import java.util.*;
import java.io.FileWriter;
import java.io.IOException;



public class YNProject1 {
       // Network configuration constants
       public static final int D = 2;                                // Number of inputs (features) for the neural network
       public static final int K = 4;                                // Number of output categories (classes) for classification
       public static final String ACTIVATION_OUTPUT = "softmax";     // Activation function for the output layer, typically used for multi-class classification
       public static final int MIN_EPOCHS = 800;                     // Minimum number of training epochs to ensure sufficient learning
       public static final double ERROR_THRESHOLD = 1e-4;            // Error threshold to determine when training can stop early   


    public static void main(String[] args) {
        // Print message indicating the start of data generation
        System.out.println("Generating Training and Testing Data...");
        
        // Generate training data points with random features and corresponding categories
        List<Point> trainingData = generateClassificationData(4000); // Generate 4000 training data points
        
        // Generate testing data points with random features and corresponding categories
        List<Point> testingData = generateClassificationData(4000);  // Generate 4000 testing data points

        // Encode the categories of training data into one-hot representation
        List<EncodedPoint> encodedTrainingData = encodeCategories(trainingData);

        // Encode the categories of testing data into one-hot representation
        List<EncodedPoint> encodedTestingData = encodeCategories(testingData);

        // Define the possible values for the size of the first hidden layer (H1)
        int[] h1Values = {10, 20, 30};

        // Define the possible values for the size of the second hidden layer (H2)
        int[] h2Values = {10, 20, 30};

        // Define the possible values for the size of the third hidden layer (H3)
        int[] h3Values = {10, 20, 30};

        // Define the activation functions to be used for the hidden layers
        String[] activations = {"tanh", "relu"};

        // Define batch sizes for training, calculated as fractions of the training dataset size
        int[] batchSizes = {encodedTrainingData.size() / 20, encodedTrainingData.size() / 200};


        try (FileWriter writer = new FileWriter("results.csv")) {
            // Write the header of the CSV file to describe each column
            writer.write("Model,H1,H2,H3,Activation,BatchSize,Accuracy\n");
        
            // Step 1: Initialize variables to find the best configuration for the first hidden layer (H1)
            int bestH1 = 0;                 // Store the size of H1 that gives the best accuracy
            double bestAccuracyH1 = 0.0;    // Store the best accuracy achieved for H1
        
            // Loop over all values of H1, activation functions, and batch sizes
            for (int h1 : h1Values) {                      // Iterate through all candidate sizes for the first hidden layer
                for (String activation : activations) {    // Iterate through activation functions ("tanh", "relu")
                    for (int batchSize : batchSizes) {     // Iterate through different batch sizes
                        
                        // Create a new MLP model with the current configuration
                        MLP model = new MLP(new int[]{D, h1, K}, activation, ACTIVATION_OUTPUT);
                        
                        // Train the model using the encoded training data
                        model.train(encodedTrainingData, batchSize, 800, 200);
        
                        // Evaluate the model's accuracy using the encoded testing data
                        double accuracy = model.evaluate(encodedTestingData);
        
                        // Check if the current configuration achieves a higher accuracy than previously seen
                        if (accuracy > bestAccuracyH1) {
                            bestH1 = h1;                   // Update the best H1 value
                            bestAccuracyH1 = accuracy;     // Update the best accuracy value
                        }
        
                        // Write the current configuration and its corresponding accuracy to the CSV file
                        writer.write("PT2," + h1 + ",NA,NA," + activation + "," + batchSize + "," + accuracy + "\n");
        
                        // Print the current configuration and accuracy to the console for monitoring progress
                        System.out.println("H1=" + h1 + ", Activation=" + activation + ", BatchSize=" + batchSize + ", Accuracy=" + accuracy);
                    }
                }
            }       

            // Step 2: Find the best H2 given the best H1
            int bestH2 = 0;                 // Variable to store the size of H2 that gives the best accuracy
            double bestAccuracyH2 = 0.0;    // Variable to store the best accuracy achieved for H2

            // Loop through all candidate sizes for the second hidden layer (H2)
            for (int h2 : h2Values) {                      // Iterate through possible sizes for the second hidden layer
                for (String activation : activations) {    // Iterate through the activation functions (e.g., "tanh", "relu")
                    for (int batchSize : batchSizes) {     // Iterate through different batch sizes
                        
                        // Create a new MLP model using the best H1 and the current H2
                        MLP model = new MLP(new int[]{D, bestH1, h2, K}, activation, ACTIVATION_OUTPUT);
                        
                        // Train the model with the encoded training data
                        model.train(encodedTrainingData, batchSize, 800, 200);

                        // Evaluate the model's accuracy on the encoded testing data
                        double accuracy = model.evaluate(encodedTestingData);

                        // Update bestH2 and bestAccuracyH2 if the current configuration achieves a higher accuracy
                        if (accuracy > bestAccuracyH2) {
                            bestH2 = h2;                   // Update the best H2 value
                            bestAccuracyH2 = accuracy;     // Update the best accuracy value
                        }

                        // Write the current configuration and accuracy to the CSV file
                        writer.write("PT2," + bestH1 + "," + h2 + ",NA," + activation + "," + batchSize + "," + accuracy + "\n");

                        // Print the configuration and accuracy to the console for real-time feedback
                        System.out.println("H1=" + bestH1 + ", H2=" + h2 + ", Activation=" + activation + ", BatchSize=" + batchSize + ", Accuracy=" + accuracy);
                    }
                }
            }

            // Step 3: Find the best H3 given the best H1 and H2
            int bestH3 = 0;                 // Variable to store the size of H3 that gives the best accuracy
            double bestAccuracyH3 = 0.0;    // Variable to store the best accuracy achieved for H3

            // Loop through all candidate sizes for the third hidden layer (H3)
            for (int h3 : h3Values) {                      // Iterate through possible sizes for the third hidden layer
                for (String activation : activations) {    // Iterate through the activation functions (e.g., "tanh", "relu")
                    for (int batchSize : batchSizes) {     // Iterate through different batch sizes
                        
                        // Create a new MLP model using the best H1, best H2, and the current H3
                        MLP model = new MLP(new int[]{D, bestH1, bestH2, h3, K}, activation, ACTIVATION_OUTPUT);
                        
                        // Train the model with the encoded training data
                        model.train(encodedTrainingData, batchSize, 800, 200);

                        // Evaluate the model's accuracy on the encoded testing data
                        double accuracy = model.evaluate(encodedTestingData);

                        // Update bestH3 and bestAccuracyH3 if the current configuration achieves a higher accuracy
                        if (accuracy > bestAccuracyH3) {
                            bestH3 = h3;                   // Update the best H3 value
                            bestAccuracyH3 = accuracy;     // Update the best accuracy value
                        }

                        // Write the current configuration and accuracy to the CSV file
                        writer.write("PT3," + bestH1 + "," + bestH2 + "," + h3 + "," + activation + "," + batchSize + "," + accuracy + "\n");

                        // Print the configuration and accuracy to the console for real-time feedback
                        System.out.println("H1=" + bestH1 + ", H2=" + bestH2 + ", H3=" + h3 + ", Activation=" + activation + ", BatchSize=" + batchSize + ", Accuracy=" + accuracy);
                    }
                }
            }

            // Print the best configuration found across all layers to the console
            System.out.println("Best Configuration: H1=" + bestH1 + ", H2=" + bestH2 + ", H3=" + bestH3);

            // Handle any potential IO exceptions
            } catch (IOException e) {  
                // Print the error message to the console
                System.err.println("Error writing to file: " + e.getMessage());  
            }
    }


    // Method to generate classification data points
    public static List<Point> generateClassificationData(int count) {
        Random rand = new Random();              // Random object for generating random values
        List<Point> data = new ArrayList<>();    // List to store the generated data points


        // Continue generating data points until the desired count is reached
        while (data.size() < count) {
            // Random value for x1 in the range [-1, 1]
            double x1 = -1 + 2 * rand.nextDouble();  

            // Random value for x2 in the range [-1, 1]
            double x2 = -1 + 2 * rand.nextDouble();  

            // Determine the category of the point using the classification logic
            int category = classifyPoint(x1, x2);   

            // Only include points that belong to a valid category
            if (category > 0) {                     
                // Add the point to the data list
                data.add(new Point(x1, x2, category));  
            }
        }

        // Return the list of generated classification data points
        return data;  
    }


    // Method to classify a point based on its coordinates
    private static int classifyPoint(double x1, double x2) {
        // Check if the point lies within a circle centered at (0.5, 0.5) and above or below y=0.5
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 > 0.5) return 1;
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 <= 0.5) return 2;

        // Check if the point lies within a circle centered at (-0.5, -0.5) and above or below y=-0.5
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 > -0.5) return 1;
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 <= -0.5) return 2;

        // Check if the point lies within a circle centered at (-0.5, 0.5) and above or below y=-0.5
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 > -0.5) return 1;
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 <= -0.5) return 2;

        // Check if the point lies within a circle centered at (0.5, -0.5) and above or below y=0.5
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 > 0.5) return 1;
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 <= 0.5) return 2;

        // Classify points based on the product of x1 and x2
        if (x1 * x2 > 0) return 3; // Quadrants I and III
        if (x1 * x2 < 0) return 4; // Quadrants II and IV

        // Return -1 for unclassified points
        return -1;
    }


    // Method to encode categories of points into one-hot vectors
    public static List<EncodedPoint> encodeCategories(List<Point> data) {
        // Initialize a list to store encoded points
        List<EncodedPoint> encodedData = new ArrayList<>();

        // Iterate through each point in the input data
        for (Point point : data) {
            // Create a one-hot encoding array for the category
            double[] oneHot = new double[K];     // K is the number of categories
            oneHot[point.category - 1] = 1.0;    // Set the appropriate index to 1.0

            // Create an EncodedPoint object and add it to the encoded data list
            encodedData.add(new EncodedPoint(point.x1, point.x2, oneHot));
        }

        // Return the list of encoded points
        return encodedData;
    }




    // A class to represent a point in a 2D space with a category
    static class Point {
        double x1, x2;  // Coordinates of the point in 2D space
        int category;   // The category of the point (used for classification)

        // Constructor to initialize the point with coordinates and a category
        public Point(double x1, double x2, int category) {
            this.x1 = x1;              // Assign the first coordinate to x1
            this.x2 = x2;              // Assign the second coordinate to x2
            this.category = category;  // Assign the category to the point
        }
    }



    // A class to represent a point in 2D space with one-hot encoded category
    static class EncodedPoint {
        double x1, x2;            // Coordinates of the point in 2D space
        double[] categoryOneHot;  // One-hot encoded representation of the category

        // Constructor to initialize the point with coordinates and the one-hot encoded category
        public EncodedPoint(double x1, double x2, double[] categoryOneHot) {
            this.x1 = x1;                             // Assign the first coordinate to x1
            this.x2 = x2;                             // Assign the second coordinate to x2
            this.categoryOneHot = categoryOneHot;     // Assign the one-hot encoded category
        }
    }



    // A class to represent a Multi-Layer Perceptron (MLP) neural network
    static class MLP {
        private final int[] layerSizes;               // Array storing the size of each layer in the network
        private final List<double[][]> weights;       // List of weight matrices, each representing the weights between layers
        private final List<double[]> biases;          // List of bias vectors, each representing the biases for each layer
        private final String activationHidden;        // Activation function used in the hidden layers
        
        @SuppressWarnings("unused")
        private final String activationOutput;        // Activation function used in the output layer
    

        // Constructor to initialize the MLP with specified layer sizes and activation functions
        public MLP(int[] layerSizes, String activationHidden, String activationOutput) {
            this.layerSizes = layerSizes;                  // Set the layer sizes for the network
            this.activationHidden = activationHidden;      // Set the activation function for the hidden layers
            this.activationOutput = activationOutput;      // Set the activation function for the output layer
            this.weights = new ArrayList<>();              // Initialize the list to store weights
            this.biases = new ArrayList<>();               // Initialize the list to store biases
            initializeWeightsAndBiases();                  // Call method to initialize weights and biases for the network
        }


        // Method to initialize the weights and biases for the network
        private void initializeWeightsAndBiases() {
            // Random number generator for weight initialization
            Random rand = new Random();  

            for (int i = 0; i < layerSizes.length - 1; i++) {
                // Initialize weight matrix for the current layer connection
                double[][] weightMatrix = new double[layerSizes[i]][layerSizes[i + 1]];
                // Initialize bias vector for the next layer
                double[] biasVector = new double[layerSizes[i + 1]];

                // Initialize the weights using He initialization (scaled by the number of neurons in the previous layer)
                for (int j = 0; j < layerSizes[i]; j++) {
                    for (int k = 0; k < layerSizes[i + 1]; k++) {
                        // He initialization (Gaussian distribution scaled by sqrt(2 / n_in))
                        weightMatrix[j][k] = rand.nextGaussian() * Math.sqrt(2.0 / layerSizes[i]);
                    }
                }
                
                // Add the initialized weight matrix and bias vector to their respective lists
                weights.add(weightMatrix);
                biases.add(biasVector);
            }
        }


        // Method to train the neural network using backpropagation
        public void train(List<EncodedPoint> data, int batchSize, int epochs, double errorThreshold) {
            // To track the loss and check for convergence
            double previousLoss = Double.MAX_VALUE;  
            
            for (int epoch = 1; epoch <= epochs; epoch++) {
                double totalLoss = 0.0;     // To accumulate the loss over each batch
                Collections.shuffle(data);  // Shuffle the data before each epoch
                
                // Process the data in batches
                for (int i = 0; i < data.size(); i += batchSize) {
                    List<EncodedPoint> batch = data.subList(i, Math.min(i + batchSize, data.size()));
                    
                    // For each point in the batch, forward pass, compute loss, and backpropagate
                    for (EncodedPoint point : batch) {
                        double[] input = {point.x1, point.x2};                                // The input features
                        double[] target = point.categoryOneHot;                               // The one-hot encoded target
                        List<double[]> activations = forward(input);                          // Perform a forward pass
                        double[] output = softmax(activations.get(activations.size() - 1));   // Softmax for the output layer
                        totalLoss += computeCrossEntropyLoss(output, target);                 // Add the loss for this point
                        backpropagate(activations, target, 0.0001);              // Backpropagate to update the weights and biases
                    }
                }

                // Print the total loss every 10 epochs
                if (epoch % 10 == 0) {
                    System.out.println("Epoch " + epoch + " Total Loss: " + totalLoss);
                }

                // Check if the loss has converged (if the change in loss is smaller than the threshold)
                if (epoch > MIN_EPOCHS && Math.abs(previousLoss - totalLoss) < errorThreshold) {
                    System.out.println("Training terminated at epoch: " + epoch);
                    // Exit the loop if training is converging
                    break;  
                }
                
                // Update the previous loss for the next iteration
                previousLoss = totalLoss;  
            }
        }


        // Method to evaluate the accuracy of the trained model on a given dataset
        public double evaluate(List<EncodedPoint> data) {
            // Variable to count the number of correct predictions
            int correct = 0;  
            
            // Iterate through all data points in the test dataset
            for (EncodedPoint point : data) {
                double[] input = {point.x1, point.x2};          // The input features of the point
                List<double[]> activations = forward(input);    // Perform a forward pass to get the activations

                // Get the predicted category index by finding the index of the maximum output value
                int predicted = argMax(activations.get(activations.size() - 1));
                
                // Get the actual category index by finding the index of the maximum value in the one-hot encoded target
                int actual = argMax(point.categoryOneHot);
                
                // If the predicted category matches the actual category, increment the correct count
                if (predicted == actual) correct++;
            }

            // Return the accuracy as a percentage (correct predictions / total points)
            return (double) correct / data.size() * 100;
        }


        // Forward pass through the neural network to calculate activations for each layer
        private List<double[]> forward(double[] input) {
            // List to store activations of each layer
            List<double[]> activations = new ArrayList<>();     
            
            // Add input layer activations to the list (initial input values)
            activations.add(input);                             

            // Iterate over each layer's weights and biases
            for (int i = 0; i < weights.size(); i++) {
                // Array to store the weighted sums (z values) for each node in the current layer
                double[] z = new double[weights.get(i)[0].length];  
                
                // Calculate the weighted sum for each node in the layer
                for (int j = 0; j < z.length; j++) {
                    for (int k = 0; k < weights.get(i).length; k++) {
                        // Weighted sum
                        z[j] += activations.get(i)[k] * weights.get(i)[k][j];  
                    }

                    // Add the bias for the node
                    z[j] += biases.get(i)[j];  

                    // Apply activation function to the weighted sum if it's not the output layer
                    if (i < weights.size() - 1) {
                        // Apply the activation function for hidden layers
                        z[j] = applyActivation(z[j]);  
                    }
                }
                
                // Store the activations of the current layer
                activations.add(z);  
            }

            // Return the list of activations for all layers
            return activations;  
        }


        // Helper method to apply the activation function (ReLU or tanh)
        private double applyActivation(double value) {
            // Return the value after applying the chosen activation function
            return activationHidden.equals("relu") ? Math.max(0, value) : Math.tanh(value);
        }


        // Method to compute the cross-entropy loss between the predicted output and the target labels
        private double computeCrossEntropyLoss(double[] output, double[] target) {
            double loss = 0.0;  // Initialize loss to 0.0
            
            // Loop through each output value and compute the contribution to the total loss
            for (int i = 0; i < output.length; i++) {
                // Cross-entropy formula: loss += - target[i] * log(output[i])
                // Add a small epsilon to avoid log(0)
                loss -= target[i] * Math.log(output[i] + 1e-9);  
            }

            // Return the total loss value
            return loss;  
        }


        // Method to perform backpropagation and update weights and biases based on the error
        private void backpropagate(List<double[]> activations, double[] target, double learningRate) {
            // Get the total number of layers in the network
            int layers = activations.size();  

            // Initialize delta array to store the error gradients
            double[][] delta = new double[layers][];  

            // Iterate backwards through the layers (starting from the output layer)
            for (int l = layers - 1; l > 0; l--) {
                // Initialize delta for the current layer
                delta[l] = new double[activations.get(l).length];  

                // If it's the output layer
                if (l == layers - 1) {  
                    // Compute the error at the output layer (difference between predicted and target)
                    for (int j = 0; j < delta[l].length; j++) {
                        delta[l][j] = activations.get(l)[j] - target[j];
                    }

                // If it's a hidden layer
                } else {  
                    // Compute the error for hidden layers using the delta from the next layer
                    for (int j = 0; j < delta[l].length; j++) {
                        for (int k = 0; k < delta[l + 1].length; k++) {
                            delta[l][j] += delta[l + 1][k] * weights.get(l)[j][k];
                        }
                        
                        // Apply the derivative of the activation function (ReLU or tanh)
                        delta[l][j] *= activationHidden.equals("relu") 
                                        ? (activations.get(l)[j] > 0 ? 1 : 0)       // ReLU derivative
                                        : 1 - Math.pow(activations.get(l)[j], 2); // tanh derivative
                    }
                }

                // Update the weights for the current layer (using the gradient and learning rate)
                for (int j = 0; j < weights.get(l - 1).length; j++) {
                    for (int k = 0; k < weights.get(l - 1)[j].length; k++) {
                        weights.get(l - 1)[j][k] -= learningRate * delta[l][k] * activations.get(l - 1)[j];
                    }
                }

                // Update the biases for the current layer
                for (int j = 0; j < biases.get(l - 1).length; j++) {
                    biases.get(l - 1)[j] -= learningRate * delta[l][j];
                }
            }
        }


       // Method to get the index of the maximum value in the array (used for classification)
        private int argMax(double[] array) {
            // Initialize the index of the maximum value
            int index = 0;  

            for (int i = 1; i < array.length; i++) {
                // Compare each element with the current maximum
                if (array[i] > array[index]) {  
                    // Update the index if a new maximum is found
                    index = i;  
                }
            }
            
            // Return the index of the maximum value
            return index;  
        }

        
        // Method to apply the softmax function to an array of logits (used for output layer)
        private double[] softmax(double[] logits) {
            // Get the maximum value from the logits for numerical stability
            double maxLogit = Arrays.stream(logits).max().orElse(0.0); 
            
            // Apply the exponential function to each logit, subtracting the max value for numerical stability
            double[] expValues = Arrays.stream(logits).map(v -> Math.exp(v - maxLogit)).toArray();
            
            // Sum the exponentiated values
            double sumExp = Arrays.stream(expValues).sum();
            
            // Normalize the exponentiated values to get the probabilities
            return Arrays.stream(expValues).map(v -> v / sumExp).toArray();
        }
    }
}