import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Arrays;

public class KMeansPlot extends Frame {

    // Class to hold clustering result data
    static class ResultData {
        double x;
        double y;
        int cluster;
        double centerX;
        double centerY;

        public ResultData(double x, double y, int cluster, double centerX, double centerY) {
            this.x = x;
            this.y = y;
            this.cluster = cluster;
            this.centerX = centerX;
            this.centerY = centerY;
        }
    }

    private java.util.List<ResultData> resultData;

    public KMeansPlot(java.util.List<ResultData> resultData) {
        this.resultData = resultData;
        prepareGUI();
    }

    private void prepareGUI() {
        setSize(800, 600);
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent windowEvent) {
                System.exit(0);
            }
        });
    }

    @Override
    public void paint(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Plot area
        int plotWidth = getWidth() - 100;
        int plotHeight = getHeight() - 100;
        int originX = 50;
        int originY = getHeight() - 50;

        // Draw axes
        g2.drawLine(originX, originY, originX + plotWidth, originY);
        g2.drawLine(originX, originY, originX, originY - plotHeight);
        g2.drawString("X", originX + plotWidth - 10, originY + 20);
        g2.drawString("Y", originX - 20, originY - plotHeight + 10);

        // Scale data
        double minX = resultData.stream().mapToDouble(d -> d.x).min().orElse(-2);
        double maxX = resultData.stream().mapToDouble(d -> d.x).max().orElse(2);
        double minY = resultData.stream().mapToDouble(d -> d.y).min().orElse(-2);
        double maxY = resultData.stream().mapToDouble(d -> d.y).max().orElse(2);

        double scaleX = plotWidth / (maxX - minX);
        double scaleY = plotHeight / (maxY - minY);

        // Draw points
        Map<Integer, Color> clusterColors = new HashMap<>();
        Random random = new Random();
        resultData.stream().map(d -> d.cluster).distinct().forEach(cluster -> {
            clusterColors.put(cluster, new Color(random.nextInt(256), random.nextInt(256), random.nextInt(256)));
        });

        for (ResultData data : resultData) {
            int px = originX + (int) ((data.x - minX) * scaleX);
            int py = originY - (int) ((data.y - minY) * scaleY);
            g2.setColor(clusterColors.get(data.cluster));
            g2.fillOval(px - 3, py - 3, 6, 6);
        }

        // Draw centers
        g2.setColor(Color.BLACK);
        for (ResultData data : resultData) {
            int cx = originX + (int) ((data.centerX - minX) * scaleX);
            int cy = originY - (int) ((data.centerY - minY) * scaleY);
            g2.fillRect(cx - 5, cy - 5, 10, 10);
        }
    }

    private static java.util.List<ResultData> readCSV(String filename) {
        java.util.List<ResultData> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean firstLine = true; // Skip the header
            while ((line = br.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue; // Skip header
                }
                String[] tokens = line.split(",");
                try {
                    double x = Double.parseDouble(tokens[0]);
                    double y = Double.parseDouble(tokens[1]);
                    int cluster = Integer.parseInt(tokens[3]);
                    double centerX = Double.parseDouble(tokens[4]);
                    double centerY = Double.parseDouble(tokens[5]);
                    data.add(new ResultData(x, y, cluster, centerX, centerY));
                } catch (NumberFormatException e) {
                    System.err.println("Error parsing numeric values: " + Arrays.toString(tokens));
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
        return data;
    }

    public static void main(String[] args) {
        String filename = "clustering_results.csv";
        java.util.List<ResultData> resultData = readCSV(filename);
        if (resultData.isEmpty()) {
            System.err.println("No valid data to plot.");
            return;
        }
        KMeansPlot plot = new KMeansPlot(resultData);
        plot.setTitle("K-Means Clustering Results");
        plot.setVisible(true);
    }
}
