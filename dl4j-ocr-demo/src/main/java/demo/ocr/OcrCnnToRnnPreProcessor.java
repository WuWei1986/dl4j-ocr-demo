package demo.ocr;
import java.util.Arrays;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * CNN与RNN结合预处理
 * 
 * @author wuwei
 *
 */
public class OcrCnnToRnnPreProcessor implements InputPreProcessor {
	private long inputHeight;
    private long inputWidth;
    private long numChannels;
    private long product;
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public OcrCnnToRnnPreProcessor(){
	}
	
	@JsonCreator
    public OcrCnnToRnnPreProcessor(@JsonProperty("inputHeight") long inputHeight,
                    @JsonProperty("inputWidth") long inputWidth, @JsonProperty("numChannels") long numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        this.product = inputHeight * inputWidth * numChannels;
    }

	@Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
		if (input.rank() != 4)
            throw new IllegalArgumentException(
                            "Invalid input: expect CNN activations with rank 4 (received input with shape "
                                            + Arrays.toString(input.shape()) + ")");
        //Input: 4d activations (CNN)
        //Output: 3d activations (RNN)

        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input))
            input = input.dup('c');

        long[] shape = input.shape(); //[timeSeriesLength*miniBatchSize, numChannels, inputHeight, inputWidth]
        //First: reshape 4d to 2d, as per CnnToFeedForwardPreProcessor
        INDArray twod = input.reshape('c', input.size(0), ArrayUtil.prod(input.shape()) / input.size(0));
        //Second: reshape 2d to 3d, as per FeedForwardToRnnPreProcessor
        INDArray reshaped = workspaceMgr.dup(ArrayType.ACTIVATIONS, twod, 'f');
        reshaped = reshaped.reshape('f', miniBatchSize, shape[0] / miniBatchSize*inputWidth, product/inputWidth);
        return reshaped.permute(0, 2, 1);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
    	 if (output.ordering() == 'c' || !Shape.hasDefaultStridesForShape(output))
             output = output.dup('f');

    	 long[] shape = output.shape();
         INDArray output2d;
         if (shape[0] == 1) {
             //Edge case: miniBatchSize = 1
             output2d = output.tensorAlongDimension(0, 1, 2).permutei(1, 0);
         } else if (shape[2] == 1) {
             //Edge case: timeSeriesLength = 1
             output2d = output.tensorAlongDimension(0, 1, 0);
         } else {
             //As per FeedForwardToRnnPreprocessor
             INDArray permuted3d = output.permute(0, 2, 1);
             output2d = permuted3d.reshape('f', shape[0] * shape[2]/inputWidth, shape[1]*inputWidth);
         }
         if (shape[1] != product/inputWidth)
             throw new IllegalArgumentException("Invalid input: expected output size(1)=" + shape[1]
                             + " must be equal to " + inputHeight + " x columns " + inputWidth + " x channels "
                             + numChannels + " = " + product + ", received: " + shape[1]);
         INDArray ret = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, output2d, 'c');
         ret = ret.reshape('c', output2d.size(0), numChannels, inputHeight, inputWidth);
         return ret;
    }

    @Override
    public FeedForwardToRnnPreProcessor clone() {
        try {
            FeedForwardToRnnPreProcessor clone = (FeedForwardToRnnPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) {
    	if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got " + inputType);
        }

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        long outSize = c.getChannels() * c.getHeight() * c.getWidth();
        return InputType.recurrent(outSize);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
    	 //Assume mask array is 4d - a mask array that has been reshaped from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1, 1, 1]
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else {
            //Need to reshape mask array from [minibatch*timeSeriesLength, 1, 1, 1] to [minibatch,timeSeriesLength]
            return new Pair<>(TimeSeriesUtils.reshapeCnnMaskToTimeSeriesMask(maskArray, minibatchSize),currentMaskState);
        }
    }

	public long getInputHeight() {
		return inputHeight;
	}

	public void setInputHeight(long inputHeight) {
		this.inputHeight = inputHeight;
	}

	public long getInputWidth() {
		return inputWidth;
	}

	public void setInputWidth(long inputWidth) {
		this.inputWidth = inputWidth;
	}

	public long getNumChannels() {
		return numChannels;
	}

	public void setNumChannels(long numChannels) {
		this.numChannels = numChannels;
	}

	public long getProduct() {
		return product;
	}

	public void setProduct(long product) {
		this.product = product;
	}
    
    
}
