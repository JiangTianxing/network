package first;

public class Backpropagation {

    //学习率
    static double learningRate;

    //隐含层权重
    static double[][] hiddenToOutputWeights;

    //输入层权重
    static double[][] inputToHiddenWeights;

    //隐含层阈值
    static double[] outputLayerThresholds;

    //输入层阈值
    static double[] hiddenLayerThresholds;

    static int outputSize;

    static void init(int inputSize, int hiddenSize, int outputSize) {
        Backpropagation.outputSize = outputSize;
        inputToHiddenWeights = DoubleInit.init(inputSize, hiddenSize);
        hiddenToOutputWeights = DoubleInit.init(hiddenSize, outputSize);
        hiddenLayerThresholds = DoubleInit.init(hiddenSize);
        outputLayerThresholds = DoubleInit.init(outputSize);
        learningRate = DoubleInit.init();
    }

    static double[][] forecast(double[][] data) {
        double[][] result = new double[data.length][outputSize];
        for (int i = 0; i < data.length; i++) {
            double[] hiddenLayerResult = Caculate.multiplyAndSigmoid(data[i], hiddenLayerThresholds, inputToHiddenWeights);
            //正向计算结果
            double[] outputLayerResult = Caculate.multiplyAndSigmoid(hiddenLayerResult, hiddenLayerThresholds, hiddenToOutputWeights);
            result[i] = outputLayerResult;
        }
        return result;
    }

    static double study(double[] inputData, double[] validationSet) {
        double[] hiddenLayerResult = Caculate.multiplyAndSigmoid(inputData, hiddenLayerThresholds, inputToHiddenWeights);
        //正向计算结果
        double[] outputLayerResult = Caculate.multiplyAndSigmoid(hiddenLayerResult, outputLayerThresholds, hiddenToOutputWeights);

        //计算反向隐层与输出层之间的参数更新迭代基量
        double[] outputLayerGradient = Caculate.getOutputToHiddenGradient(outputLayerResult, validationSet);
        //计算反向输入层与隐层之间的参数更新迭代基量
        double[] hiddenLayerGradient = Caculate.getHiddenToInputGradient(hiddenLayerResult, outputLayerGradient, hiddenToOutputWeights);
        //更新
        updateHiddenToOutputWeights(outputLayerGradient, hiddenLayerResult);

        updateInputToHiddenWeights(hiddenLayerGradient, inputData);

        updateOutputLayerThresholds(outputLayerGradient);

        updateHiddenLayerThresholds(hiddenLayerGradient);

        return Caculate.getDistance(outputLayerResult, validationSet);
    }

    static void updateHiddenToOutputWeights(double[] outputLayerGradient, double[] hiddenLayerResult) {
        for (int i = 0; i < hiddenToOutputWeights.length; i++) {
            for (int j = 0; j < hiddenToOutputWeights[0].length; j++) {
                hiddenToOutputWeights[i][j] += learningRate * outputLayerGradient[j] * hiddenLayerResult[i];
            }
        }
    }

    static void updateInputToHiddenWeights(double[] hiddenLayerGradient, double[] inputData) {
        for (int i = 0; i < inputToHiddenWeights.length; i++) {
            for (int j = 0; j < inputToHiddenWeights[0].length; j++) {
                inputToHiddenWeights[i][j] += learningRate * hiddenLayerGradient[i] * inputData[i];
            }
        }
    }

    static void updateOutputLayerThresholds(double[] outputLayerGradient) {
        for (int i = 0; i < outputLayerGradient.length; i++) {
            outputLayerThresholds[i] -= learningRate * outputLayerGradient[i];
        }
    }

    static void updateHiddenLayerThresholds(double[] hiddenLayerGradient) {
        for (int i = 0; i < hiddenLayerGradient.length; i++) {
            hiddenLayerThresholds[i] -= learningRate * hiddenLayerGradient[i];
        }
    }

    public static void main(String[] args) {
        init(1, 6, 1);
        for (int i = 0; i < 10000000; i++) {
            double x = Math.random() * 10;
            double y = 0.5 * (Math.cos(x) + 1);
            double[] testSet = new double[]{x};
            double[] validationSet = new double[]{y};
            double result = study(testSet, validationSet);
            System.out.println("distance == " + result);
        }
        double x = 7.43;
        System.out.println(0.5 * (Math.cos(x) + 1));
        double[] hiddenLayerResult = Caculate.multiplyAndSigmoid(new double[]{x}, hiddenLayerThresholds, inputToHiddenWeights);
        //正向计算结果
        double[] outputLayerResult = Caculate.multiplyAndSigmoid(hiddenLayerResult, outputLayerThresholds, hiddenToOutputWeights);
        System.out.println(outputLayerResult[0]);
        x = 5.23;
        System.out.println(0.5 * (Math.cos(x) + 1));
        hiddenLayerResult = Caculate.multiplyAndSigmoid(new double[]{x}, hiddenLayerThresholds, inputToHiddenWeights);
        //正向计算结果
        outputLayerResult = Caculate.multiplyAndSigmoid(hiddenLayerResult, outputLayerThresholds, hiddenToOutputWeights);
        System.out.println(outputLayerResult[0]);
        x = 0.948;
        System.out.println(0.5 * (Math.cos(x) + 1));
        hiddenLayerResult = Caculate.multiplyAndSigmoid(new double[]{x}, hiddenLayerThresholds, inputToHiddenWeights);
        //正向计算结果
        outputLayerResult = Caculate.multiplyAndSigmoid(hiddenLayerResult, outputLayerThresholds, hiddenToOutputWeights);
        System.out.println(outputLayerResult[0]);

    }
}


class DoubleInit {

    public static double init() {
        return Math.random();
    }

    public static double[] init(int length) {
        double[] params = new double[length];
        for (int i = 0; i < params.length; i++) {
            params[i] = Math.random();
        }
        return params;
    }

    public static double[][] init(int xSize, int ySize) {
        double[][] params = new double[xSize][ySize];
        for (int i = 0; i < xSize; i++) {
            for (int j = 0; j < ySize; j++) {
                params[i][j] = Math.random();
            }
        }
        return params;
    }
}


class Caculate {

    public static double[] multiplyAndSigmoid(double[] inputData, double[] thresholds, double[][] weightData) {
        int size = thresholds.length;
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            double sum = 0;
            for (int j = 0; j < weightData.length; j++) {
                sum += inputData[j] * weightData[j][i];
            }
            result[i] = (1.0 / (1 + Math.exp(-1 * sum)));
        }
        return result;
    }

    public static double[] getOutputToHiddenGradient(double[] result, double[] validation) {
        int l = result.length;
        double[] gradients = new double[l];
        for (int i = 0; i < l; i++) {
            gradients[i] = result[i] * (1 - result[i]) * (validation[i] - result[i]);
        }
        return gradients;
    }


    public static double[] getHiddenToInputGradient(double[] hiddenLayerResult, double[] outputLayerGradient, double[][] hiddenToOutputWeights) {
        int l = hiddenLayerResult.length;
        double[] gradient = new double[l];
        for (int i = 0; i < l; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenToOutputWeights[0].length; j++) {
                sum += hiddenToOutputWeights[i][j] * outputLayerGradient[j];
            }
            gradient[i] = sum * hiddenLayerResult[i] * (1 - hiddenLayerResult[i]);
        }
        return gradient;
    }

    public static double getDistance(double[] data, double[] result) {
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum += Math.pow((data[i] - result[i]), 2);
        }
        return sum / 2;
    }
}