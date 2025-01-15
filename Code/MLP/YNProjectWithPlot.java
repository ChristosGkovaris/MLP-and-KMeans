import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;



public class YNProjectWithPlot extends Frame {


    // Class to hold result data for each MLP configuration and its performance (accuracy)
    static class ResultData {
        int h1, h2, h3, batchSize;     // Hyperparameters: h1, h2, h3 represent neurons in hidden layers, and batchSize for training
        String activation;             // Activation function used ("relu", "tanh")
        double accuracy;               // Accuracy achieved by the configuration


        // Constructor to initialize a ResultData object with hyperparameters, activation function, batch size, and accuracy
        public ResultData(int h1, int h2, int h3, String activation, int batchSize, double accuracy) {
            this.h1 = h1;                   // Assign h1 (neurons in the first hidden layer)
            this.h2 = h2;                   // Assign h2 (neurons in the second hidden layer)
            this.h3 = h3;                   // Assign h3 (neurons in the third hidden layer)
            this.activation = activation;   // Assign the activation function used
            this.batchSize = batchSize;     // Assign the batch size used for training
            this.accuracy = accuracy;       // Assign the accuracy achieved by this configuration
        }
    }


    // List to store multiple ResultData objects, each representing a configuration's result
    private List<ResultData> resultData; 


    // Constructor for the YNProjectWithPlot class that accepts a list of ResultData
    public YNProjectWithPlot(List<ResultData> resultData) {
        this.resultData = resultData;  // Initialize the resultData list with the provided data
        prepareGUI();                  // Prepare the graphical user interface (GUI)
    }


    // Method to prepare and initialize the graphical user interface (GUI) for the plot
    private void prepareGUI() {
        setSize(800, 600); 
        // Set the size of the window to 800px width and 600px height
        
        // Add a window listener to handle the window closing event
        addWindowListener(new WindowAdapter() {
            // Override the windowClosing method to define the behavior when the window is closed
            public void windowClosing(WindowEvent windowEvent) {
                // Exit the program when the window is closed
                System.exit(0); 
            }
        });
    }


    // Override the paint method to perform custom drawing for the frame
    public void paint(Graphics g) {
        // Cast the Graphics object to Graphics2D for advanced drawing
        Graphics2D g2 = (Graphics2D) g; 

        // Enable anti-aliasing for smoother graphics
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON); 

        // Set font style and size for labels
        Font font = new Font("Serif", Font.PLAIN, 10); 
        g2.setFont(font);

        // Plot area dimensions and origin
        int plotWidth = getWidth() - 100;   // Plot width with padding
        int plotHeight = getHeight() - 150; // Plot height with padding
        int originX = 50;                   // X coordinate of the origin of the plot
        int originY = getHeight() - 100;    // Y coordinate of the origin of the plot

        // Draw the X and Y axes of the plot
        g2.drawLine(originX, originY, originX + plotWidth, originY);                    // X axis
        g2.drawLine(originX, originY, originX, originY - plotHeight);                   // Y axis
        g2.drawString("Accuracy", originX - 40, originY - plotHeight + 10);         // Y axis label
        g2.drawString("Configurations", originX + plotWidth - 40, originY + 40);    // X axis label

        // Calculate scaling factors for plotting
        // Get the maximum accuracy value
        double maxAccuracy = resultData.stream().mapToDouble(r -> r.accuracy).max().orElse(100); 
        
        // Get the number of configurations
        int numConfigs = resultData.size(); 

        // Calculate the width of each bar
        int barWidth = plotWidth / numConfigs; 

        // Calculate the scaling factor for the accuracy
        double scale = plotHeight / maxAccuracy; 

        // Draw bars for each configuration in the result data
        for (int i = 0; i < resultData.size(); i++) {
            ResultData data = resultData.get(i);           // Get the result data for the current configuration
            int barHeight = (int) (data.accuracy * scale); // Calculate the height of the bar based on accuracy
            int x = originX + i * barWidth;                // X coordinate for the bar
            int y = originY - barHeight;                   // Y coordinate for the bar (origin is at the bottom)

            // Set the color for the bar (red for 'relu' activation, blue for others)
            g2.setPaint(data.activation.equals("relu") ? Color.RED : Color.BLUE);

            // Draw the bar
            g2.fillRect(x, y, barWidth - 2, barHeight); 
            
            // Reset color to black for labels
            g2.setPaint(Color.BLACK); 

            // Rotate and draw the labels for each bar
            Graphics2D g2d = (Graphics2D) g;

            // Rotate the label 45 degrees
            g2d.rotate(Math.toRadians(45), x, originY + 5); 

            // Display the configuration parameters as label
            g2d.drawString(String.format("H1=%d,H2=%d,H3=%d", data.h1, data.h2, data.h3), x, originY + 10); 
            
            // Reset the rotation back to normal
            g2d.rotate(Math.toRadians(-45), x, originY + 5); 
        }
    }


   // Method to read data from a CSV file and store it in a list of ResultData objects
    private static List<ResultData> readCSV(String filename) {
        // Initialize a list to store the result data
        List<ResultData> data = new ArrayList<>(); 


        // Try-with-resources to automatically close the BufferedReader
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) { 
            String line;

            // Flag to skip the header line of the CSV file
            boolean firstLine = true; 

            // Read each line of the CSV file
            while ((line = br.readLine()) != null) { 
                // Skip the header line
                if (firstLine) { 
                    firstLine = false;
                    continue;
                }

                // Split the line into an array of strings (tokens) based on commas
                String[] tokens = line.split(","); 


                try {
                    // Parse each column from the tokens array
                    // If "NA", set to 0, otherwise parse as integer
                    int h1 = tokens[1].equals("NA") ? 0 : Integer.parseInt(tokens[1]); 
                    int h2 = tokens[2].equals("NA") ? 0 : Integer.parseInt(tokens[2]);
                    int h3 = tokens[3].equals("NA") ? 0 : Integer.parseInt(tokens[3]);
                    
                    // Get activation function
                    String activation = tokens[4]; 

                    // Parse batch size
                    int batchSize = Integer.parseInt(tokens[5]); 

                    // Parse accuracy as double
                    double accuracy = Double.parseDouble(tokens[6]); 

                    // Add a new ResultData object with the parsed values to the data list
                    data.add(new ResultData(h1, h2, h3, activation, batchSize, accuracy));
                
                // Catch any number format exceptions
                } catch (NumberFormatException e) {   
                    // Print error message
                    System.err.println("Error parsing numeric values: " + Arrays.toString(tokens)); 
                }
            }

        // Catch any IOExceptions while reading the file
        } catch (IOException e) {      
            // Print error message
            System.err.println("Error reading file: " + e.getMessage()); 
        }

        // Return the list of parsed data
        return data; 
    }


    public static void main(String[] args) {
        // Set the file name for the CSV file
        String filename = "results.csv"; 

        // Call the readCSV method to parse the CSV file into result data
        List<ResultData> resultData = readCSV(filename); 

        // If no valid data is found
        if (resultData.isEmpty()) { 
            // Print an error message
            System.err.println("No valid data to plot."); 
            
            // Exit the method
            return; 
        }

        // Create a new plot object with the result data
        YNProjectWithPlot plot = new YNProjectWithPlot(resultData);
        
        // Set the plot title
        plot.setTitle("MLP Configuration vs Accuracy"); 
        
        // Make the plot window visible
        plot.setVisible(true); 
    }
}