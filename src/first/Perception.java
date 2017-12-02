package first;

public class Perception {

    //学习率
    static double learningRate;

    //权重
    static double[] weight;

    //阈值
    static double threshold;

    /**
     * sigmod函数  1.0/(1+exp(-x))
     * @param result
     * @return
     */
    private static int sigmoid(double result) {
//        System.out.println(result);
        return (1.0 / (Math.exp(-1 * result) + 1)) >= 0.5 ? 1 : -1;
//        return Math.sin(result) >= 0 ? 1 : -1;
    }

    /**
     * 启动
     * @param testSet 测试集
     * @param validationSet 验证集
     */
    public static void start(int[][] testSet, int[] validationSet) {
        int latitude = validationSet.length;
        init(testSet[0].length);
        int count = 0;
        //检查是否已经收敛
        boolean ifIterae = false;
        while (!ifIterae) {
//        for (int j = 0; j < 300; j++) {
           for (int i = 0; i < latitude && !ifIterae; i++) {
                boolean stuStatus = study(testSet[i], validationSet[i]);
                if (!stuStatus) count = 0;
                else count++;
                if (count == latitude) ifIterae = true;
           }
        }
    }

    /**
     * 学习
     * @param testData 测试集
     * @param validation 验证数据
     * @return 验证结构
     */
    private static boolean study(int[] testData, int validation) {
        double sum = 0;
        for (int i = 0; i < testData.length; i++) {
            sum += testData[i] * weight[i];
        }
        sum += threshold;
        int result = sigmoid(sum);
        for (int i = 0; i < testData.length; i++) {
            weight[i] = weight[i] + learningRate * (validation - result) * testData[i];
        }
        threshold = threshold + learningRate * (validation - result);
        return result == validation;
    }

    /**
     * 初始化
     * @param latitude 数组纬度
     */
    private static void init(int latitude) {
        threshold = Math.random();
        learningRate = Math.random();
        weight = new double[latitude];
        for (int i = 0; i < latitude; i++) {
            weight[i] = Math.random();
        }
    }

    public static void main(String[] args) {
        int[][] testSet = {
                {0, 1, 1},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1},
                {1, 0, 0},
                {0, 1, 0}
        };
        int[] validationSet = {1, 1, 1, 1, -1, -1};
        start(testSet, validationSet);
        StringBuilder shizi = new StringBuilder();
        shizi.append("result = sigmod(").append(weight[0]).append("x1 + ").append(weight[1]).append("x2 + ").append(weight[2]).append("x3 + " + threshold + ")");
        System.out.println(shizi.toString());
        int[][] test = {
                {0, 0, 1},
                {0, 0, 0}
        };
//        int[] result = {-1, -1};
        int[] result = getResult(test);
        for (int i = 0; i < result.length; i++) {
            System.out.println(result[i]);
        }
    }

    public static int[] getResult(int[][] testData) {
        int[] result = new int[testData.length];

        for (int i = 0; i < testData.length; i++) {
            double sum = 0;
            for (int j = 0; j < testData[0].length; j++) {
                sum+=weight[j] * testData[i][j];
            }
            sum += threshold;
            result[i] = sigmoid(sum);
        }
        return result;
    }
}
