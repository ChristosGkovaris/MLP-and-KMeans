import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.stream.Collectors;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;



public class KMeansPlot extends Frame {


    // Inner class to hold the result data for each data point
    static class ResultData {
        double x, y;              // Coordinates of the data point
        int cluster;              // The cluster this point belongs to
        double centerX, centerY;  // Coordinates of the cluster center

        // Constructor to initialize the result data
        public ResultData(double x, double y, int cluster, double centerX, double centerY) {
            this.x = x;
            this.y = y;
            this.cluster = cluster;
            this.centerX = centerX;
            this.centerY = centerY;
        }
    }


    // List to hold all the result data (data points and centers)
    private java.util.List<ResultData> resultData;


    // Constructor that initializes the result data and GUI setup
    public KMeansPlot(java.util.List<ResultData> resultData) {
        this.resultData = resultData;
        prepareGUI();
    }


    // Prepare the GUI, setting up the window size and close event
    private void prepareGUI() {
        // Set the size of the window
        setSize(800, 800); 
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent windowEvent) {
                // Close the program when the window is closed
                System.exit(0); 
            }
        });
    }


    // Override the paint method to draw the plot
    @Override
    public void paint(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;

        // Enable anti-aliasing for smooth lines
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON); 

        // Define the plot dimensions and origin
        int plotWidth = getWidth() - 100;
        int plotHeight = getHeight() - 100;
        int originX = 50;
        int originY = getHeight() - 50;

        // Draw the axes
        g2.drawLine(originX, originY, originX + plotWidth, originY);      // X-axis
        g2.drawLine(originX, originY, originX, originY - plotHeight);     // Y-axis
        g2.drawString("X", originX + plotWidth - 10, originY + 20);   // Label for X-axis
        g2.drawString("Y", originX - 20, originY - plotHeight + 10);  // Label for Y-axis

        // Draw axis ticks and labels
        // Number of ticks for each axis
        int numTicks = 10; 
        for (int i = 0; i <= numTicks; i++) {
            // Draw X-axis ticks and labels
            int xTick = originX + (int) (i * plotWidth / numTicks);
            double xValue = -2 + i * (4.0 / numTicks);

            // Draw tick
            g2.drawLine(xTick, originY, xTick, originY - 5); 

            // Label for X value
            g2.drawString(String.format("%.1f", xValue), xTick - 10, originY + 15); 

            // Draw Y-axis ticks and labels
            int yTick = originY - (int) (i * plotHeight / numTicks);
            double yValue = -2 + i * (4.0 / numTicks);

            // Draw tick
            g2.drawLine(originX, yTick, originX + 5, yTick); 

            // Label for Y value
            g2.drawString(String.format("%.1f", yValue), originX - 30, yTick + 5); 
        }

        // Determine the minimum and maximum values of X and Y to scale the plot
        double minX = resultData.stream().mapToDouble(d -> d.x).min().orElse(-2);
        double maxX = resultData.stream().mapToDouble(d -> d.x).max().orElse(2);
        double minY = resultData.stream().mapToDouble(d -> d.y).min().orElse(-2);
        double maxY = resultData.stream().mapToDouble(d -> d.y).max().orElse(2);

        // Scale factors for X and Y axes
        double scaleX = plotWidth / (maxX - minX);
        double scaleY = plotHeight / (maxY - minY);

        // Assign predefined colors to each cluster
        Map<Integer, Color> clusterColors = new HashMap<>();
        Color[] predefinedColors = {Color.RED, Color.BLUE, Color.GREEN, Color.MAGENTA, Color.CYAN, Color.ORANGE, Color.PINK, Color.YELLOW};
        
        // Map each cluster to a color
        int colorIndex = 0;
        for (int cluster : resultData.stream().map(d -> d.cluster).distinct().collect(Collectors.toList())) {
            clusterColors.put(cluster, predefinedColors[colorIndex % predefinedColors.length]);
            colorIndex++;
        }

        // Plot the data points with '+' symbol and cluster-specific colors
        for (ResultData data : resultData) {
            // X-coordinate on the plot
            int px = originX + (int) ((data.x - minX) * scaleX);     

            // Y-coordinate on the plot
            int py = originY - (int) ((data.y - minY) * scaleY);    
            
            // Set the color based on the cluster
            g2.setColor(clusterColors.get(data.cluster));  
            
            // Draw data point with '+' symbol
            g2.drawString("+", px - 3, py + 3);                 
        }

        // Plot the cluster centers with '*' symbol in black
        g2.setColor(Color.BLACK);
        Map<Integer, ResultData> centers = resultData.stream()
            // Keep only one center per cluster
            .collect(Collectors.toMap(d -> d.cluster, d -> d, (a, b) -> a)); 

        for (ResultData center : centers.values()) {
            // X-coordinate of the cluster center
            int cx = originX + (int) ((center.centerX - minX) * scaleX); 

            // Y-coordinate of the cluster center
            int cy = originY - (int) ((center.centerY - minY) * scaleY);
            
            // Draw cluster center with '*' symbol
            g2.drawString("*", cx - 3, cy + 3); 
        }
    }


    // Read the CSV file containing clustering data and return a list of ResultData
    private static java.util.List<ResultData> readCSV(String filename) {
        java.util.List<ResultData> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;

            // Skip the header line
            boolean firstLine = true; 

            while ((line = br.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                String[] tokens = line.split(",");
                // Skip lines with insufficient data
                if (tokens.length < 6) continue; 

                // Parse the CSV line into relevant values
                int cluster = Integer.parseInt(tokens[2]);
                double centerX = Double.parseDouble(tokens[3]);
                double centerY = Double.parseDouble(tokens[4]);
                double x = Double.parseDouble(tokens[5]);
                double y = Double.parseDouble(tokens[6]);

                // Add parsed data to the list
                data.add(new ResultData(x, y, cluster, centerX, centerY)); 
            }

        } catch (IOException e) {
            // Handle file reading errors
            System.err.println("Error reading file: " + e.getMessage()); 
        }
        return data;
    }


    // Main method to execute the program
    public static void main(String[] args) {
        // The CSV file containing clustering results
        String filename = "clustering_results.csv"; 

        // Read the CSV data
        java.util.List<ResultData> resultData = readCSV(filename); 

        if (resultData.isEmpty()) {
            // Error if no valid data was found
            System.err.println("No valid data to plot."); 
            return;
        }

        // Create the plot with the data
        KMeansPlot plot = new KMeansPlot(resultData); 

        // Set the window title
        plot.setTitle("Enhanced K-Means Clustering Plot"); 
        
        // Display the plot window
        plot.setVisible(true); 
    }
}
