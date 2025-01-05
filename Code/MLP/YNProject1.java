import java.util.*;
import java.io.FileWriter;
import java.io.IOException;

public class YNProject1 {
    // Network configuration
    public static final int D = 2; // Number of inputs
    public static final int K = 4; // Number of categories
    public static final String ACTIVATION_OUTPUT = "softmax"; // Activation for output layer
    public static final int MIN_EPOCHS = 800;
    public static final double ERROR_THRESHOLD = 1e-4;

    public static void main(String[] args) {
        System.out.println("Generating Training and Testing Data...");
        List<Point> trainingData = generateClassificationData(4000); // Generate training data
        List<Point> testingData = generateClassificationData(4000);  // Generate testing data

        List<EncodedPoint> encodedTrainingData = encodeCategories(trainingData);
        List<EncodedPoint> encodedTestingData = encodeCategories(testingData);

        int[] h1Values = {10, 20, 30};
        int[] h2Values = {10, 20, 30};
        int[] h3Values = {10, 20, 30};
        String[] activations = {"tanh", "relu"};
        int[] batchSizes = {encodedTrainingData.size() / 20, encodedTrainingData.size() / 200};

        try (FileWriter writer = new FileWriter("results.csv")) {
            writer.write("Model,H1,H2,H3,Activation,BatchSize,Accuracy\n");

            // Step 1: Find the best H1
            int bestH1 = 0;
            double bestAccuracyH1 = 0.0;
            for (int h1 : h1Values) {
                for (String activation : activations) {
                    for (int batchSize : batchSizes) {
                        MLP model = new MLP(new int[]{D, h1, K}, activation, ACTIVATION_OUTPUT);
                        model.train(encodedTrainingData, batchSize, 800, 200);
                        double accuracy = model.evaluate(encodedTestingData);
                        if (accuracy > bestAccuracyH1) {
                            bestH1 = h1;
                            bestAccuracyH1 = accuracy;
                        }
                        writer.write("PT2," + h1 + ",NA,NA," + activation + "," + batchSize + "," + accuracy + "\n");
                        System.out.println("H1=" + h1 + ", Activation=" + activation + ", BatchSize=" + batchSize + ", Accuracy=" + accuracy);
                    }
                }
            }

            // Step 2: Find the best H2 given the best H1
            int bestH2 = 0;
            double bestAccuracyH2 = 0.0;
            for (int h2 : h2Values) {
                for (String activation : activations) {
                    for (int batchSize : batchSizes) {
                        MLP model = new MLP(new int[]{D, bestH1, h2, K}, activation, ACTIVATION_OUTPUT);
                        model.train(encodedTrainingData, batchSize, 800, 200);
                        double accuracy = model.evaluate(encodedTestingData);
                        if (accuracy > bestAccuracyH2) {
                            bestH2 = h2;
                            bestAccuracyH2 = accuracy;
                        }
                        writer.write("PT2," + bestH1 + "," + h2 + ",NA," + activation + "," + batchSize + "," + accuracy + "\n");
                        System.out.println("H1=" + bestH1 + ", H2=" + h2 + ", Activation=" + activation + ", BatchSize=" + batchSize + ", Accuracy=" + accuracy);
                    }
                }
            }

            // Step 3: Find the best H3 given the best H1 and H2
            int bestH3 = 0;
            double bestAccuracyH3 = 0.0;
            for (int h3 : h3Values) {
                for (String activation : activations) {
                    for (int batchSize : batchSizes) {
                        MLP model = new MLP(new int[]{D, bestH1, bestH2, h3, K}, activation, ACTIVATION_OUTPUT);
                        model.train(encodedTrainingData, batchSize, 800, 200);
                        double accuracy = model.evaluate(encodedTestingData);
                        if (accuracy > bestAccuracyH3) {
                            bestH3 = h3;
                            bestAccuracyH3 = accuracy;
                        }
                        writer.write("PT3," + bestH1 + "," + bestH2 + "," + h3 + "," + activation + "," + batchSize + "," + accuracy + "\n");
                        System.out.println("H1=" + bestH1 + ", H2=" + bestH2 + ", H3=" + h3 + ", Activation=" + activation + ", BatchSize=" + batchSize + ", Accuracy=" + accuracy);
                    }
                }
            }

            System.out.println("Best Configuration: H1=" + bestH1 + ", H2=" + bestH2 + ", H3=" + bestH3);

        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }

    public static List<Point> generateClassificationData(int count) {
        Random rand = new Random();
        List<Point> data = new ArrayList<>();
        while (data.size() < count) {
            double x1 = -1 + 2 * rand.nextDouble();
            double x2 = -1 + 2 * rand.nextDouble();
            int category = classifyPoint(x1, x2);
            if (category > 0) {
                data.add(new Point(x1, x2, category));
            }
        }
        return data;
    }

    private static int classifyPoint(double x1, double x2) {
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 > 0.5) return 1;
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 <= 0.5) return 2;
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 > -0.5) return 1;
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 <= -0.5) return 2;
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 > -0.5) return 1;
        if ((Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) && x2 <= -0.5) return 2;
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 > 0.5) return 1;
        if ((Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) && x2 <= 0.5) return 2;
        if (x1 * x2 > 0) return 3;
        if (x1 * x2 < 0) return 4;
        return -1;
    }

    public static List<EncodedPoint> encodeCategories(List<Point> data) {
        List<EncodedPoint> encodedData = new ArrayList<>();
        for (Point point : data) {
            double[] oneHot = new double[K];
            oneHot[point.category - 1] = 1.0;
            encodedData.add(new EncodedPoint(point.x1, point.x2, oneHot));
        }
        return encodedData;
    }

    static class Point {
        double x1, x2;
        int category;

        public Point(double x1, double x2, int category) {
            this.x1 = x1;
            this.x2 = x2;
            this.category = category;
        }
    }

    static class EncodedPoint {
        double x1, x2;
        double[] categoryOneHot;

        public EncodedPoint(double x1, double x2, double[] categoryOneHot) {
            this.x1 = x1;
            this.x2 = x2;
            this.categoryOneHot = categoryOneHot;
        }
    }

    static class MLP {
        private final int[] layerSizes;
        private final List<double[][]> weights;
        private final List<double[]> biases;
        private final String activationHidden;
        private final String activationOutput;

        public MLP(int[] layerSizes, String activationHidden, String activationOutput) {
            this.layerSizes = layerSizes;
            this.activationHidden = activationHidden;
            this.activationOutput = activationOutput;
            this.weights = new ArrayList<>();
            this.biases = new ArrayList<>();
            initializeWeightsAndBiases();
        }

        private void initializeWeightsAndBiases() {
            Random rand = new Random();
            for (int i = 0; i < layerSizes.length - 1; i++) {
                double[][] weightMatrix = new double[layerSizes[i]][layerSizes[i + 1]];
                double[] biasVector = new double[layerSizes[i + 1]];
                for (int j = 0; j < layerSizes[i]; j++) {
                    for (int k = 0; k < layerSizes[i + 1]; k++) {
                        weightMatrix[j][k] = rand.nextGaussian() * Math.sqrt(2.0 / layerSizes[i]);
                    }
                }
                weights.add(weightMatrix);
                biases.add(biasVector);
            }
        }

        public void train(List<EncodedPoint> data, int batchSize, int epochs, double errorThreshold) {
            double previousLoss = Double.MAX_VALUE;
            for (int epoch = 1; epoch <= epochs; epoch++) {
                double totalLoss = 0.0;
                Collections.shuffle(data);
                for (int i = 0; i < data.size(); i += batchSize) {
                    List<EncodedPoint> batch = data.subList(i, Math.min(i + batchSize, data.size()));
                    for (EncodedPoint point : batch) {
                        double[] input = {point.x1, point.x2};
                        double[] target = point.categoryOneHot;
                        List<double[]> activations = forward(input);
                        double[] output = softmax(activations.get(activations.size() - 1));
                        totalLoss += computeCrossEntropyLoss(output, target);
                        backpropagate(activations, target, 0.0001);
                    }
                }
                if (epoch % 10 == 0) {
                    System.out.println("Epoch " + epoch + " Total Loss: " + totalLoss);
                }
                if (epoch > MIN_EPOCHS && Math.abs(previousLoss - totalLoss) < errorThreshold) {
                    System.out.println("Training terminated at epoch: " + epoch);
                    break;
                }
                previousLoss = totalLoss;
            }
        }

        public double evaluate(List<EncodedPoint> data) {
            int correct = 0;
            for (EncodedPoint point : data) {
                double[] input = {point.x1, point.x2};
                List<double[]> activations = forward(input);
                int predicted = argMax(activations.get(activations.size() - 1));
                int actual = argMax(point.categoryOneHot);
                if (predicted == actual) correct++;
            }
            return (double) correct / data.size() * 100;
        }

        private List<double[]> forward(double[] input) {
            List<double[]> activations = new ArrayList<>();
            activations.add(input);
            for (int i = 0; i < weights.size(); i++) {
                double[] z = new double[weights.get(i)[0].length];
                for (int j = 0; j < z.length; j++) {
                    for (int k = 0; k < weights.get(i).length; k++) {
                        z[j] += activations.get(i)[k] * weights.get(i)[k][j];
                    }
                    z[j] += biases.get(i)[j];
                    if (i < weights.size() - 1) {
                        z[j] = applyActivation(z[j]);
                    }
                }
                activations.add(z);
            }
            return activations;
        }

        private double applyActivation(double value) {
            return activationHidden.equals("relu") ? Math.max(0, value) : Math.tanh(value);
        }

        private double computeCrossEntropyLoss(double[] output, double[] target) {
            double loss = 0.0;
            for (int i = 0; i < output.length; i++) {
                loss -= target[i] * Math.log(output[i] + 1e-9);
            }
            return loss;
        }

        private void backpropagate(List<double[]> activations, double[] target, double learningRate) {
            int layers = activations.size();
            double[][] delta = new double[layers][];

            for (int l = layers - 1; l > 0; l--) {
                delta[l] = new double[activations.get(l).length];
                if (l == layers - 1) {
                    for (int j = 0; j < delta[l].length; j++) {
                        delta[l][j] = activations.get(l)[j] - target[j];
                    }
                } else {
                    for (int j = 0; j < delta[l].length; j++) {
                        for (int k = 0; k < delta[l + 1].length; k++) {
                            delta[l][j] += delta[l + 1][k] * weights.get(l)[j][k];
                        }
                        delta[l][j] *= activationHidden.equals("relu") ? (activations.get(l)[j] > 0 ? 1 : 0) : 1 - Math.pow(activations.get(l)[j], 2);
                    }
                }

                for (int j = 0; j < weights.get(l - 1).length; j++) {
                    for (int k = 0; k < weights.get(l - 1)[j].length; k++) {
                        weights.get(l - 1)[j][k] -= learningRate * delta[l][k] * activations.get(l - 1)[j];
                    }
                }
                for (int j = 0; j < biases.get(l - 1).length; j++) {
                    biases.get(l - 1)[j] -= learningRate * delta[l][j];
                }
            }
        }

        private int argMax(double[] array) {
            int index = 0;
            for (int i = 1; i < array.length; i++) {
                if (array[i] > array[index]) {
                    index = i;
                }
            }
            return index;
        }

        private double[] softmax(double[] logits) {
            double maxLogit = Arrays.stream(logits).max().orElse(0.0);
            double[] expValues = Arrays.stream(logits).map(v -> Math.exp(v - maxLogit)).toArray();
            double sumExp = Arrays.stream(expValues).sum();
            return Arrays.stream(expValues).map(v -> v / sumExp).toArray();
        }
    }
}