import java.util.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;

public class YNProjectWithPlot {

    // Class to hold result data
    static class ResultData {
        double x1, x2;
        boolean correct;

        public ResultData(double x1, double x2, boolean correct) {
            this.x1 = x1;
            this.x2 = x2;
            this.correct = correct;
        }
    }

    private List<ResultData> resultData;

    public YNProjectWithPlot(List<ResultData> resultData) {
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

        int width = getWidth();
        int height = getHeight();

        // Plot bounds
        int margin = 50;
        int plotWidth = width - 2 * margin;
        int plotHeight = height - 2 * margin;

        // Draw axes
        g2.drawLine(margin, margin, margin, margin + plotHeight);
        g2.drawLine(margin, margin + plotHeight, margin + plotWidth, margin + plotHeight);
        
        // Scale points
        double xMin = -1.0, xMax = 1.0, yMin = -1.0, yMax = 1.0;
        for (ResultData point : resultData) {
            int x = margin + (int) ((point.x1 - xMin) / (xMax - xMin) * plotWidth);
            int y = margin + plotHeight - (int) ((point.x2 - yMin) / (yMax - yMin) * plotHeight);

            g2.setColor(point.correct ? Color.GREEN : Color.RED);
            g2.fillOval(x - 3, y - 3, 6, 6);
        }

        // Labels
        g2.setColor(Color.BLACK);
        g2.drawString("x1", width / 2, height - margin / 2);
        g2.drawString("x2", margin / 2, height / 2);
    }

    public static List<ResultData> evaluateModel(List<Point> testData, MLP model) {
        List<ResultData> results = new ArrayList<>();
        for (Point point : testData) {
            double[] input = {point.x1, point.x2};
            int predicted = model.predict(input);
            boolean correct = predicted == point.category;
            results.add(new ResultData(point.x1, point.x2, correct));
        }
        return results;
    }

    public static void main(String[] args) {
        // Load test data and model
        List<Point> testData = YNProject1.generateClassificationData(1000);
        MLP model = new MLP(new int[]{2, 10, 20, 4}, "relu", "softmax");

        // Assume model is trained (replace with actual training if needed)
        List<ResultData> resultData = evaluateModel(testData, model);

        // Display plot
        YNProjectWithPlot plot = new YNProjectWithPlot(resultData);
        plot.setTitle("Model Evaluation: Correct (Green) vs Incorrect (Red)");
        plot.setVisible(true);
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

    static class MLP {
        private int[] layerSizes;
        private String activationHidden;
        private String activationOutput;

        public MLP(int[] layerSizes, String activationHidden, String activationOutput) {
            this.layerSizes = layerSizes;
            this.activationHidden = activationHidden;
            this.activationOutput = activationOutput;
        }

        public int predict(double[] input) {
            // Dummy implementation for prediction
            return (int) (Math.random() * 4) + 1;
        }
    }
}