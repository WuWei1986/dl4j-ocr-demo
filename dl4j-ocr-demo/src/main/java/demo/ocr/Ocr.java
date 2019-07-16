package demo.ocr;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 文本识别
 * 
 * @author wuwei
 *
 */
public class Ocr {
	private static long seed = 123;
	private static int epochs = 100;
	private static int batchSize = 15;
	private static int height = 60;
	private static int width = 160;
	private static int channels = 1;
	private static int timeSeriesLength = 16; // 时序长度
	private static String dirPath = "E:/deeplearning/captcha/";
	private static String textCharStr = "_234578acdefgmnpwxy";
	private static String modelFileName = "ocrCtcModel.json";
	private static int maxLabelLength = 8; // 标签最大长度（支持不定长度训练）
	
	public static void main(String[] args) throws IOException, InterruptedException {
		MultiDataSetIterator trainMulIterator = new MultiRecordDataSetIterator(batchSize, "train");
		//ComputationGraph model = ModelSerializer.restoreComputationGraph(modelFileName);
		ComputationGraph model = createModel();
		model.init();
		System.out.println(model.summary(InputType.convolutional(height, width, channels)));
		train(model,trainMulIterator);
		//modelPredict(model, trainMulIterator);
		ModelSerializer.writeModel(model, modelFileName, true);
		System.out.println("end...");
	}
	
	/**
	 * 创建模型
	 * 
	 * @return
	 */
	public static ComputationGraph createModel(){
		ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder().seed(seed)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.l2(1e-3)
				.updater(new Adam(1e-4))
				.weightInit(WeightInit.XAVIER_UNIFORM)
				.graphBuilder()
				.addInputs("trainFeatures")
				.setInputTypes(InputType.convolutional(height, width, 1))
				.setOutputs("rnnout")
				.addLayer("cnn1",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
	                .nIn(1).nOut(64).activation( Activation.RELU).build(), "trainFeatures")
	            .addLayer("maxpool1",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
	                .build(), "cnn1")
	            .addLayer("cnn2",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
	                .nOut(128).activation( Activation.RELU).build(), "maxpool1")
	            .addLayer("bn1", new BatchNormalization.Builder(false).build(), "cnn2")
	            .addLayer("maxpool2",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,1}, new int[]{2, 1}, new int[]{0, 0})
	                .build(), "bn1")
	            .addLayer("cnn3",  new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0})
	                .nOut(256).activation( Activation.RELU).build(), "maxpool2")
	            .addLayer("maxpool3",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
	                .build(), "cnn3")
	            .addLayer("cnn4",  new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0})
	                .nOut(512).activation( Activation.RELU).build(), "maxpool3")
	            .addLayer("bn2", new BatchNormalization.Builder(false).build(), "cnn4")
	            .addLayer("maxpool4",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
	                .build(), "bn2")
				.addLayer("lstm0", new LSTM.Builder().nIn(512).nOut(256).activation(Activation.TANH).build(), "maxpool4")
				.addLayer("lstm1", new LSTM.Builder().nIn(256).nOut(256).activation(Activation.TANH).build(), "lstm0")
				.addLayer("rnnout", new RnnOutputLayer.Builder(new LossCTC(timeSeriesLength,batchSize)).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
						.nIn(256).nOut(textCharStr.length()).build(),"lstm1")
				.inputPreProcessor("lstm0", new OcrCnnToRnnPreProcessor(1,16,512))
				.build();
		ComputationGraph model = new ComputationGraph(config);
		return model;
	}
	
	/**
	 * 训练
	 * 
	 * @param model
	 * @param trainMulIterator
	 * @throws IOException
	 */
	public static void train(ComputationGraph model,MultiDataSetIterator trainMulIterator) throws IOException{
		UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10), new StatsListener( statsStorage));
		for ( int i = 0; i < epochs; i ++ ) {
            System.out.println("Epoch=====================" + i);
            model.fit(trainMulIterator);
            // 每10轮保存一次模型
            if ((i+1)%10 == 0 && i+1 !=epochs) {
            	ModelSerializer.writeModel(model, modelFileName, true);
            }
        }
	}
	
	/**
	 * 预测
	 * 
	 * @param model
	 * @param iterator
	 */
	public static void modelPredict(ComputationGraph model, MultiDataSetIterator iterator) {
        int sumCount = 0;
        int correctCount = 0;

        while (iterator.hasNext()) {
        	org.nd4j.linalg.dataset.api.MultiDataSet mds = iterator.next();
            INDArray[]  output = model.output(true,mds.getFeatures());
            INDArray[] labels = mds.getLabels();
            long[] shap = output[0].shape();
            int dataNum = (int)shap[0];
            for (int dataIndex = 0;  dataIndex < dataNum; dataIndex ++) {
                String reLabel = "";
                String peLabel = "";
                INDArray preOutput = null;
                INDArray realLabel = null;
                for (int digit = 0; digit < (int)output[0].shape()[2]; digit ++) {
                    preOutput = output[0].getRow(dataIndex).getColumn(digit);
                    preOutput.putScalar(new int[]{0,0}, 0);
                    peLabel += textCharStr.charAt((Nd4j.argMax(preOutput, 0).getInt(0)));
                }
                peLabel = simpleParseOutput(peLabel);
                for (int digit = 0; digit < (int)labels[0].shape()[2]; digit ++) {
                	realLabel =  labels[0].getRow(dataIndex).getColumn(digit);
                	// label不对齐处理
                	if (Nd4j.sum(realLabel, 0).getInt(0) == 0) {
    					continue;
    				}
 	                reLabel += textCharStr.charAt((Nd4j.argMax(realLabel, 0).getInt(0)));
                }
                if (peLabel.equals(reLabel)) {
                    correctCount ++;
                }
                sumCount ++;
                System.out.println("real image "+reLabel+"  prediction "+peLabel+" status "+peLabel.equals(reLabel)+"");
            }
        }
        iterator.reset();
        System.out.println("validate result : sum count =" + sumCount + " correct count=" + correctCount );
    }
	
	static class MultiRecordDataSetIterator implements MultiDataSetIterator {
	    /**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		private int batchSize = 0;
	    private int batchNum = 0;
	    private int numExample = 0;
	    private MulRecordDataLoader load;
	    private MultiDataSetPreProcessor preProcessor;

	    public MultiRecordDataSetIterator(int batchSize, String dataSetType) {
	        this(batchSize, null, dataSetType);
	    }
	    public MultiRecordDataSetIterator(int batchSize, ImageTransform imageTransform, String dataSetType) {
	        this.batchSize = batchSize;
	        load = new MulRecordDataLoader(height,width,channels,imageTransform, dataSetType);
	        numExample = load.totalExamples();
	    }


	    @Override
	    public MultiDataSet next(int i) {
	        batchNum += i;
	        MultiDataSet mds = load.next(i);
	        if (preProcessor != null) {
	            preProcessor.preProcess(mds);
	        }
	        return mds;
	    }

	    @Override
	    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
	        this.preProcessor = multiDataSetPreProcessor;
	    }

	    @Override
	    public MultiDataSetPreProcessor getPreProcessor() {
	        return preProcessor;
	    }

	    @Override
	    public boolean resetSupported() {
	        return true;
	    }

	    @Override
	    public boolean asyncSupported() {
	        return true;
	    }

	    @Override
	    public void reset() {
	        batchNum = 0;
	        load.reset();
	    }

	    @Override
	    public boolean hasNext() {
	        if(batchNum < numExample){
	            return true;
	        } else {
	            return false;
	        }
	    }

	    @Override
	    public MultiDataSet next() {
	        return next(batchSize);
	    }
	}
	
	static class MulRecordDataLoader extends NativeImageLoader implements Serializable {
	    /**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		private static final Logger log = LoggerFactory.getLogger(MulRecordDataLoader.class);

	    private File fullDir = new File(dirPath);
	    private Iterator<File> fileIterator;
	    private int numExample = 0;

	    public MulRecordDataLoader(int height, int width, int channels, ImageTransform imageTransform, String dataSetType) {
	        super(height, width, channels, imageTransform);
	        this.height = height;
	        this.width = width;
	        this.channels = channels;
	        this.fullDir = new File(fullDir, dataSetType);
	        load();
	    }

	    protected void load() {
	        try {
	            List<File> dataFiles = (List<File>) FileUtils.listFiles(fullDir, new String[]{"jpg"}, true );
	            Collections.shuffle(dataFiles);
	            fileIterator = dataFiles.iterator();
	            numExample = dataFiles.size();
	        } catch (Exception var4) {
	            throw new RuntimeException( var4 );
	        }
	    }

	    public MultiDataSet convertDataSet(int num) throws Exception {
	        int batchNumCount = 0;

	        INDArray[] featuresMask = null;
	        INDArray[] labelMask = null;

	        List<MultiDataSet> multiDataSets = new ArrayList<>();

	        while (batchNumCount != num && fileIterator.hasNext()) {
	            File image = fileIterator.next();
	            String imageName = image.getName().substring(0,image.getName().lastIndexOf('.'));
	            String[] imageNames = imageName.split("");
	            INDArray feature = asMatrix(image);
	            INDArray[] features = new INDArray[]{feature};
	            INDArray label = Nd4j.zeros(1, textCharStr.length(),maxLabelLength);
	            INDArray[] labels = new INDArray[]{label};
	            Nd4j.getAffinityManager().ensureLocation(feature, AffinityManager.Location.DEVICE);
	            for (int i = 0; i < imageNames.length; i ++) {
	            	int digit = textCharStr.indexOf(imageNames[i]);
	            	label.putScalar(new int[]{0, digit,i}, 1);
	            }
	            feature =  feature.muli(1.0/255.0);
	            multiDataSets.add(new MultiDataSet(features, labels, featuresMask, labelMask));

	            batchNumCount ++;
	        }
	        MultiDataSet result = MultiDataSet.merge(multiDataSets);
	        return result;
	    }

	    public MultiDataSet next(int batchSize) {
	        try {
	            MultiDataSet result = convertDataSet( batchSize );
	            return result;
	        } catch (Exception e) {
	            log.error("the next function shows error", e);
	        }
	        return null;
	    }

	    public void reset() {
	        load();
	    }
	    public int totalExamples() {
	        return numExample;
	    }
	    
	}
	
	/**
	 * 简单处理预测结果
	 * 每个时序除去blank后最大的做为该时序的预测结果
	 * 
	 * @param out
	 * @return
	 */
	private static String simpleParseOutput(String out) {
		StringBuilder sb = new StringBuilder();
		char temp = out.charAt(0);
		sb.append(temp);
		for (int i=1;i<out.length();i++) {
			if (out.charAt(i) != temp) {
				sb.append(out.charAt(i));
				temp = out.charAt(i);
			}
		}
		return sb.toString();
	}
	
}
