import java.util.Random;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;



public class KMeansClustering {
    private static final int[] CLUSTER_COUNTS = {4, 6, 8, 10, 12};   
    // Different numbers of clusters (M)

    // Total number of data points
    private static final int N = 1000;
    
    // Number of dimensions (2D points)
    private static final int DIM = 2; 


    public static void main(String[] args) {
        double[][] data = new double[N][DIM];    // Array to hold data points
        generateDatasetSDO(data);                // Generate dataset using predefined regions

        try (FileWriter csvWriter = new FileWriter("clustering_results.csv")) {
            // Write CSV header
            csvWriter.append("M,Error,Cluster,Center_X,Center_Y,X,Y\n"); 

            // Iterate over different cluster counts (M)
            for (int M : CLUSTER_COUNTS) {
                System.out.println("Running K-Means for M = " + M);

                // Centroids for M clusters
                double[][] centroids = new double[M][DIM];
                
                // Cluster assignment for each data point
                int[] clusters = new int[N];              

                // Store the best centroids
                double[][] bestCentroids = new double[M][DIM];
                
                // Initialize minimum error
                double minError = Double.MAX_VALUE; 

                // Perform 20 runs with different initializations of centroids
                for (int run = 0; run < 20; run++) {

                    initializeCentroids(centroids, data, M); 
                    double previousError, currentError = Double.MAX_VALUE;

                    // Iterate until convergence (error change is small enough)
                    do {
                        previousError = currentError;

                        // Assign points to closest centroid
                        assignClusters(data, clusters, centroids, M); 

                        // Update centroids based on cluster assignments
                        updateCentroids(data, clusters, centroids, M); 

                        // Calculate clustering error
                        currentError = calculateClusteringError(data, clusters, centroids, M); 
                    } while (Math.abs(previousError - currentError) > 1e-6);   // Check for convergence

                    System.out.printf("M = %d, Run = %d, Error = %.6f\n", M, run + 1, currentError);

                    // Track best centroids and minimum error
                    if (currentError < minError) {
                        minError = currentError;
                        for (int i = 0; i < M; i++) {
                            // Save best centroids
                            bestCentroids[i] = Arrays.copyOf(centroids[i], DIM); 
                        }
                    }
                }

                // Write results for this value of M to the CSV file
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        
                        // Check if data point belongs to cluster i
                        if (clusters[j] == i) { 
                            csvWriter.append(String.format("%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f\n", 
                                M, minError, i + 1, bestCentroids[i][0], bestCentroids[i][1], data[j][0], data[j][1]));
                        }
                    }
                }
                System.out.printf("Best clustering error for M = %d: %.6f\n", M, minError);
            }

            // Notify user that results are saved
            System.out.println("Best results written to clustering_results.csv"); 

        } catch (IOException e) {
            // Error handling if writing fails
            System.err.println("Error writing to CSV file: " + e.getMessage()); 
        }
    }


    // Method to generate dataset based on predefined regions for clustering
    private static void generateDatasetSDO(double[][] data) {
        Random random = new Random();
        int index = 0;

        // Define regions for data points with specified boundaries
        double[][] regions = {
            {-2, -1.6, 1.6, 2}, {-1.2, -0.8, 1.6, 2}, {-0.4, 0, 1.6, 2},
            {-1.8, -1.4, 0.8, 1.2}, {-0.6, -0.2, 0.8, 1.2}, {-2, -1.6, 0, 0.4},
            {-1.2, -0.8, 0, 0.4}, {-0.4, 0, 0, 0.4}
        };

        // Generate 100 points for each region
        for (double[] region : regions) {
            for (int i = 0; i < 100; i++) {
                // X-coordinate
                data[index][0] = region[0] + random.nextDouble() * (region[1] - region[0]); 

                // Y-coordinate
                data[index][1] = region[2] + random.nextDouble() * (region[3] - region[2]); 
                index++;
            }
        }

        // Generate 200 additional points in a larger region
        for (int i = 0; i < 200; i++) {
            // X-coordinate
            data[index][0] = -2 + random.nextDouble() * 2;
            
            // Y-coordinate
            data[index][1] = 0 + random.nextDouble() * 2; 
            index++;
        }
    }


    // Method to initialize centroids randomly from data points
    private static void initializeCentroids(double[][] centroids, double[][] data, int M) {
        Random random = new Random();

        // Array to track which data points are used for centroids
        boolean[] used = new boolean[N]; 

        for (int i = 0; i < M; i++) {
            int idx;
            do {
                // Select a random data point
                idx = random.nextInt(N); 
            } while (used[idx]);  // Ensure the point is not reused
            
            used[idx] = true;
            // Set centroid to the selected data point
            centroids[i] = Arrays.copyOf(data[idx], DIM); 
        }
    }


    // Method to assign each data point to the closest centroid
    private static void assignClusters(double[][] data, int[] clusters, double[][] centroids, int M) {
        for (int i = 0; i < N; i++) {
            double minDistance = Double.MAX_VALUE;
            int bestCluster = -1;
            // Find the closest centroid for each data point
            for (int j = 0; j < M; j++) {
                double distance = 0.0;
                for (int k = 0; k < DIM; k++) {
                    // Euclidean distance
                    distance += Math.pow(data[i][k] - centroids[j][k], 2); 
                }
                if (distance < minDistance) {
                    minDistance = distance;

                    // Update closest cluster
                    bestCluster = j; 
                }
            }

            // Assign data point to the best cluster
            clusters[i] = bestCluster; 
        }
    }


    // Method to update centroids based on current cluster assignments
    private static void updateCentroids(double[][] data, int[] clusters, double[][] centroids, int M) {
        double[][] sums = new double[M][DIM];    // Sum of data points for each cluster
        int[] counts = new int[M];               // Number of data points in each cluster

        for (int i = 0; i < N; i++) {
            int cluster = clusters[i];
            for (int j = 0; j < DIM; j++) {
                // Add data point to the sum for the respective cluster
                sums[cluster][j] += data[i][j]; 
            }

            // Increment count for the respective cluster
            counts[cluster]++; 
        }

        // Update centroids by averaging the data points in each cluster
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < DIM; j++) {
                // Average
                centroids[i][j] = counts[i] > 0 ? sums[i][j] / counts[i] : 0; 
            }
        }
    }


    // Method to calculate the clustering error (sum of squared distances from data points to centroids)
    private static double calculateClusteringError(double[][] data, int[] clusters, double[][] centroids, int M) {
        double error = 0.0;
        for (int i = 0; i < N; i++) {
            int cluster = clusters[i];
            for (int j = 0; j < DIM; j++) {
                // Sum of squared differences
                error += Math.pow(data[i][j] - centroids[cluster][j], 2); 
            }
        }

        // Return total error
        return error; 
    }
}