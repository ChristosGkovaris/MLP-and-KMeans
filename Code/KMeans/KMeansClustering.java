import java.util.Random;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;

public class KMeansClustering {
    private static final int[] CLUSTER_COUNTS = {4, 6, 8, 10, 12}; // Different values for M
    private static final int N = 1000; // Number of data points
    private static final int DIM = 2; // Dimension of data points

    public static void main(String[] args) {
        double[][] data = new double[N][DIM];
        generateDatasetSDO(data);

        try (FileWriter csvWriter = new FileWriter("best_clustering_results.csv")) {
            csvWriter.append("M,Error,Cluster,Center_X,Center_Y,X,Y\n");

            for (int M : CLUSTER_COUNTS) {
                System.out.println("Running K-Means for M = " + M);

                double[][] centroids = new double[M][DIM];
                int[] clusters = new int[N];
                double[][] bestCentroids = new double[M][DIM];
                double minError = Double.MAX_VALUE;

                for (int run = 0; run < 20; run++) { // 20 runs for different initializations
                    initializeCentroids(centroids, data, M);
                    double previousError, currentError = Double.MAX_VALUE;

                    do {
                        previousError = currentError;
                        assignClusters(data, clusters, centroids, M);
                        updateCentroids(data, clusters, centroids, M);
                        currentError = calculateClusteringError(data, clusters, centroids, M);
                    } while (Math.abs(previousError - currentError) > 1e-6);

                    System.out.printf("M = %d, Run = %d, Error = %.6f\n", M, run + 1, currentError);

                    if (currentError < minError) {
                        minError = currentError;
                        for (int i = 0; i < M; i++) {
                            bestCentroids[i] = Arrays.copyOf(centroids[i], DIM);
                        }
                    }
                }

                // Write the best results for this M to the CSV
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        if (clusters[j] == i) {
                            csvWriter.append(String.format("%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f\n", 
                                M, minError, i + 1, bestCentroids[i][0], bestCentroids[i][1], data[j][0], data[j][1]));
                        }
                    }
                }

                System.out.printf("Best clustering error for M = %d: %.6f\n", M, minError);
            }

            System.out.println("Best results written to best_clustering_results.csv");
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
        }
    }

    private static void generateDatasetSDO(double[][] data) {
        Random random = new Random();
        int index = 0;

        double[][] regions = {
            {-2, -1.6, 1.6, 2}, {-1.2, -0.8, 1.6, 2}, {-0.4, 0, 1.6, 2},
            {-1.8, -1.4, 0.8, 1.2}, {-0.6, -0.2, 0.8, 1.2}, {-2, -1.6, 0, 0.4},
            {-1.2, -0.8, 0, 0.4}, {-0.4, 0, 0, 0.4}
        };

        // Generate 100 points for each specific region
        for (double[] region : regions) {
            for (int i = 0; i < 100; i++) {
                data[index][0] = region[0] + random.nextDouble() * (region[1] - region[0]);
                data[index][1] = region[2] + random.nextDouble() * (region[3] - region[2]);
                index++;
            }
        }

        // Generate 200 additional points in the larger region
        for (int i = 0; i < 200; i++) {
            data[index][0] = -2 + random.nextDouble() * 2;
            data[index][1] = 0 + random.nextDouble() * 2;
            index++;
        }
    }

    private static void initializeCentroids(double[][] centroids, double[][] data, int M) {
        Random random = new Random();
        boolean[] used = new boolean[N];

        for (int i = 0; i < M; i++) {
            int idx;
            do {
                idx = random.nextInt(N);
            } while (used[idx]);
            used[idx] = true;
            centroids[i] = Arrays.copyOf(data[idx], DIM);
        }
    }

    private static void assignClusters(double[][] data, int[] clusters, double[][] centroids, int M) {
        for (int i = 0; i < N; i++) {
            double minDistance = Double.MAX_VALUE;
            int bestCluster = -1;
            for (int j = 0; j < M; j++) {
                double distance = 0.0;
                for (int k = 0; k < DIM; k++) {
                    distance += Math.pow(data[i][k] - centroids[j][k], 2);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = j;
                }
            }
            clusters[i] = bestCluster;
        }
    }

    private static void updateCentroids(double[][] data, int[] clusters, double[][] centroids, int M) {
        double[][] sums = new double[M][DIM];
        int[] counts = new int[M];

        for (int i = 0; i < N; i++) {
            int cluster = clusters[i];
            for (int j = 0; j < DIM; j++) {
                sums[cluster][j] += data[i][j];
            }
            counts[cluster]++;
        }

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < DIM; j++) {
                centroids[i][j] = counts[i] > 0 ? sums[i][j] / counts[i] : 0;
            }
        }
    }

    private static double calculateClusteringError(double[][] data, int[] clusters, double[][] centroids, int M) {
        double error = 0.0;
        for (int i = 0; i < N; i++) {
            int cluster = clusters[i];
            for (int j = 0; j < DIM; j++) {
                error += Math.pow(data[i][j] - centroids[cluster][j], 2);
            }
        }
        return error;
    }
}
