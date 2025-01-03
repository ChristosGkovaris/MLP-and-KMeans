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
        generateRandomData(data);

        try (FileWriter csvWriter = new FileWriter("clustering_results.csv")) {
            csvWriter.append("M,Run,Error,Cluster,Center_X,Center_Y,Point_X,Point_Y\n");

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

                    if (currentError < minError) {
                        minError = currentError;
                        for (int i = 0; i < M; i++)
                            bestCentroids[i] = Arrays.copyOf(centroids[i], DIM);
                    }

                    // Write centroids to CSV
                    for (int i = 0; i < M; i++) {
                        csvWriter.append(String.format("%d,%d,%f,%d,%f,%f,,\n", M, run + 1, currentError, i + 1, centroids[i][0], centroids[i][1]));
                    }

                    // Write points and their assigned clusters to CSV
                    for (int i = 0; i < N; i++) {
                        csvWriter.append(String.format("%d,%d,%f,%d,,,%.6f,%.6f\n", M, run + 1, currentError, clusters[i] + 1, data[i][0], data[i][1]));
                    }
                }

                System.out.println("Best clustering error for M = " + M + ": " + minError);
                System.out.println("Cluster centers for M = " + M + ":");
                for (int i = 0; i < M; i++) {
                    System.out.println("Center " + (i + 1) + ": " + Arrays.toString(bestCentroids[i]));
                }
                System.out.println();
            }

            System.out.println("Results written to clustering_results.csv");
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
        }
    }

    private static void generateRandomData(double[][] data) {
        Random random = new Random();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < DIM; j++) {
                data[i][j] = random.nextDouble() * 4 - 2; // Random values in [-2, 2]
            }
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
