package demo.ocr;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import onnx.OnnxProto3.AttributeProto;
import onnx.OnnxProto3.GraphProto;
import onnx.OnnxProto3.NodeProto;

/**
 * CTC损失函数
 * 
 * @author wuwei
 *
 */
public class LossCTC extends DifferentialFunction implements ILossFunction{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private static final double DEFAULT_SOFTMAX_CLIPPING_EPSILON = 1e-10;
	private static int timeSeriesLength;
	private static int miniBatchSize;
	
	@JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray weights;

    private double softmaxClipEps;
    public LossCTC(int timeSeriesLength,int miniBatchSize) {
        this(null);
        this.timeSeriesLength = timeSeriesLength;
        this.miniBatchSize = miniBatchSize;
    }
    
    public LossCTC(INDArray weights) {
        this(DEFAULT_SOFTMAX_CLIPPING_EPSILON, weights);
    }
    
    public LossCTC(@JsonProperty("softmaxClipEps") double softmaxClipEps, @JsonProperty("weights") INDArray weights) {
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        if(softmaxClipEps < 0 || softmaxClipEps > 0.5){
            throw new IllegalArgumentException("Invalid clipping epsilon: epsilon should be >= 0 (but near zero). Got: "
                    + softmaxClipEps);
        }
        this.weights = weights;
        this.softmaxClipEps = softmaxClipEps;
    }

	@Override
	public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
			boolean average) {
		INDArray output = activationFn.getActivation(preOutput.dup(), true);
        if(activationFn instanceof ActivationSoftmax && softmaxClipEps > 0.0){
            BooleanIndexing.replaceWhere(output, softmaxClipEps, Conditions.lessThan(softmaxClipEps));
            BooleanIndexing.replaceWhere(output, 1.0-softmaxClipEps, Conditions.greaterThan(1.0-softmaxClipEps));
        }
        return computeCtcLoss(labels,output);
	}

	@Override
	public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        throw new UnsupportedOperationException("not supported");
	}

	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        if (activationFn instanceof ActivationSoftmax) {
            if (mask != null && LossUtil.isPerOutputMasking(output, mask)) {
                throw new UnsupportedOperationException("Per output masking for CTC + softmax: not supported");
            }
            return computeCtcGradient(labels,output);
        }
        throw new UnsupportedOperationException("not supported");
	}

	@Override
	public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
			INDArray mask, boolean average) {
		return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
	}

	@Override
	public String name() {
		return "LossCTC()";
	}
	
	@Override
    public String toString() {
        return "LossCTC()";
    }

	@Override
	public SDVariable[] outputVariables(String baseName) {
		return new SDVariable[0];
	}

	@Override
	public List<SDVariable> doDiff(List<SDVariable> f1) {
		return null;
	}

	@Override
	public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode,
			GraphDef graph) {
		
	}

	@Override
	public void initFromOnnx(NodeProto node, SameDiff initWith, Map<String, AttributeProto> attributesForNode,
			GraphProto graph) {
		
	}

	@Override
	public String onnxName() {
		return "CTCLoss";
	}

	@Override
	public String tensorflowName() {
		return "CTCLoss";
	}
	
	/**
	 * 计算梯度
	 * 
	 * @param labels
	 * @param preOutput
	 * @return
	 */
	private INDArray computeCtcGradient(INDArray labels, INDArray preOutput){
		INDArray grad =  Nd4j.zeros(preOutput.shape());
		long[] lablesShape = labels.shape();
		INDArray reshapedLabels = labels.reshape('f', miniBatchSize, lablesShape[0] / miniBatchSize, lablesShape[1]);
		reshapedLabels = reshapedLabels.permute(0, 2, 1);
		lablesShape = reshapedLabels.shape();
		INDArray reshapedPreOutput = preOutput.reshape('f', miniBatchSize, preOutput.shape()[0] / miniBatchSize, preOutput.shape()[1]);
		reshapedPreOutput = reshapedPreOutput.permute(0, 2, 1);
		for (int i = 0; i < (int)lablesShape[0]; i ++) {
			int llength = 0;
			// label不对齐处理
			for (int j = 0; j < (int)lablesShape[2]; j ++) {
				INDArray label =  reshapedLabels.getRow(i).getColumn(j);
				if (Nd4j.sum(label, 0).getInt(0) > 0) {
					llength ++;
				}
	        }
			// 扩展的label数组
			int[] l = new int[llength*2+1];
			l[0] = 0;
			for (int j = 0; j < llength; j ++) {
				INDArray label =  reshapedLabels.getRow(i).getColumn(j);
				l[j*2+1]=((Nd4j.argMax(label, 0).getInt(0)));
				l[j*2+2]=0;
	        }
			Node node0 = new Node(0,0);
			Node node1 = new Node(0,1);
			Map<Node,Node> validNodes = new HashMap<>();
			computeValidNodes(l,node0,validNodes);
			computeValidNodes(l,node1,validNodes);
			Map<Integer,List<Node>> timeSeriesMap = validNodes.values().stream().collect(Collectors.groupingBy(Node::getKey,LinkedHashMap::new, Collectors.toList()));
			for (Node node:timeSeriesMap.get(timeSeriesLength -1)) {
				computeNodeForwardProbability(l, node, reshapedPreOutput.getRow(i), validNodes);
			} 
			computeNodeBackProbability(l,node0,reshapedPreOutput.getRow(i),validNodes);
			computeNodeBackProbability(l,node1,reshapedPreOutput.getRow(i),validNodes);
			//for (Node node:validNodes.values().stream().sorted().collect(Collectors.toList())) {
			//	System.out.println(node);
			//}
			float plx = 0f;
			if (validNodes.get(new Node(timeSeriesLength -1,l.length-1)) != null) {
				plx += validNodes.get(new Node(timeSeriesLength -1,l.length-1)).getFp();
			}
			if (validNodes.get(new Node(timeSeriesLength -1,l.length-2)) != null) {
				plx += validNodes.get(new Node(timeSeriesLength -1,l.length-2)).getFp();
			}
			for (int t=0;t<timeSeriesLength;t++) {
				INDArray temp = Nd4j.zeros(1, preOutput.shape()[1]);
				INDArray kArray = reshapedPreOutput.getRow(i).getColumn(t);
				List<Node> nodes = timeSeriesMap.get(t);
				for (int k=0;k<kArray.shape()[0];k++) {
					List<Node> lkNodes = getNodes(nodes,k,l);
					float ytk = 0f;
					float ykt = kArray.getFloat(new int[]{k,0});
					if (lkNodes != null && !lkNodes.isEmpty()) {
						for (Node lkNode:lkNodes) {
							ytk += lkNode.getFp()*lkNode.getBp()/ykt;
						}
					}
					temp.putScalar(new int[]{0, k}, ykt-ytk/plx);
				}
				grad.putRow(i+t*miniBatchSize,temp);
			}
		}
		return grad;
	}
	
	/**
	 * 计算损失
	 * 
	 * @param labels
	 * @param preOutput
	 * @return
	 */
	private double computeCtcLoss(INDArray labels, INDArray preOutput){
		long[] lablesShape = labels.shape();
		INDArray reshapedLabels = labels.reshape('f', miniBatchSize, lablesShape[0] / miniBatchSize, lablesShape[1]);
		reshapedLabels = reshapedLabels.permute(0, 2, 1);
		lablesShape = reshapedLabels.shape();
		INDArray reshapedPreOutput = preOutput.reshape('f', miniBatchSize, preOutput.shape()[0] / miniBatchSize, preOutput.shape()[1]);
		reshapedPreOutput = reshapedPreOutput.permute(0, 2, 1);
		double loss = 0d;
		for (int i = 0; i < (int)lablesShape[0]; i ++) {
			int llength = 0;
			for (int j = 0; j < (int)lablesShape[2]; j ++) {
				INDArray label =  reshapedLabels.getRow(i).getColumn(j);
				if (Nd4j.sum(label, 0).getInt(0) > 0) {
					llength ++;
				}
	        }
			// 扩展的label数组
			int[] l = new int[llength*2+1];
			l[0] = 0;
			for (int j = 0; j < llength; j ++) {
				INDArray label =  reshapedLabels.getRow(i).getColumn(j);
				l[j*2+1]=((Nd4j.argMax(label, 0).getInt(0)));
				l[j*2+2]=0;
	        }
			Node node0 = new Node(0,0);
			Node node1 = new Node(0,1);
			Map<Node,Node> validNodes = new HashMap<>();
			computeValidNodes(l,node0,validNodes);
			computeValidNodes(l,node1,validNodes);
			Map<Integer,List<Node>> timeSeriesMap = validNodes.values().stream().collect(Collectors.groupingBy(Node::getKey,LinkedHashMap::new, Collectors.toList()));
			for (Node node:timeSeriesMap.get(timeSeriesLength -1)) {
				computeNodeForwardProbability(l, node, reshapedPreOutput.getRow(i), validNodes);
			} 
			float plx = 0f;
			if (validNodes.get(new Node(timeSeriesLength -1,l.length-1)) != null) {
				plx += validNodes.get(new Node(timeSeriesLength -1,l.length-1)).getFp();
			}
			if (validNodes.get(new Node(timeSeriesLength -1,l.length-2)) != null) {
				plx += validNodes.get(new Node(timeSeriesLength -1,l.length-2)).getFp();
			}
			loss += -Math.log(plx);
		}
		return loss;
	}
	
	private List<Node> getNodes(List<Node> nodes, int k, int[] l) {
		List<Node> result = new ArrayList<>();
		for (Node node:nodes) {
			if (l[node.getValue()] == k) {
				result.add(node);
			}
		}
		return result;
	}
	
	/**
	 * 获取当前节点的后续可能的节点
	 * 
	 * @param l
	 * @param node
	 * @return
	 */
	private static List<Node> getNextNodes(int[] l , Node node){
		int t = node.getKey();
		int s = node.getValue();
		if (t == timeSeriesLength -1) {
			return null;
		}
		List<Node> nextNodes = new ArrayList<>();
		for (int i=0;i<l.length;i++) {
			// 转换只能往右下方向
			if (i >= s){
				// 相同字符间起码要有一个空字符
				if (i != s && l[i] == l[s]) {
					break;
				}
				// 非空字符不能跳过
				if (i != s && l[i] != 0) {
					Node newNode = new Node(t+1,i);
					nextNodes.add(newNode);
					break;
				}
				Node newNode = new Node(t+1,i);
				nextNodes.add(newNode);
			}
		}
		return nextNodes;
	} 
	
	/**
	 * 计算前向概率
	 * 
	 * @param l
	 * @param node
	 * @param preOutput
	 * @param nodeMap
	 */
	private static void computeNodeForwardProbability(int[] l ,Node node, INDArray preOutput, Map<Node,Node> nodeMap) {
		if (node.getKey() == 0 && (node.getValue() == 0 || node.getValue() == 1)) {
			node.setFp(preOutput.getScalar(new int[]{l[node.getValue()], node.getKey()}).getFloat(0));
			return ;
		}
		if (!nodeMap.containsKey(node)) {
			return ;
		}
		// 当前seq(s)为空符号或者当前seq(s) == seq(s-2)
		if (l[node.getValue()] == 0 || (node.getValue() -2 >=0 && l[node.getValue()] ==l[node.getValue() -2]) ) {
			float p1 = 0f;
			float p2 = 0f;
			Node pre1 = nodeMap.get(new Node(node.getKey()-1,node.getValue()));
			Node pre2 = nodeMap.get(new Node(node.getKey()-1,node.getValue()-1));
			if (pre1 != null) {
				if (pre1.getFp() == 0f) {
					computeNodeForwardProbability(l, pre1, preOutput, nodeMap);
				}
				p1 = pre1.getFp();
			}
			if (pre2 != null) {
				if (pre2.getFp() == 0f) {
					computeNodeForwardProbability(l, pre2, preOutput, nodeMap);
				}
				p2 = pre2.getFp();
			}
			node.setFp((p1+p2)*preOutput.getScalar(new int[]{l[node.getValue()], node.getKey()}).getFloat(0));
		} else {
			float p1 = 0f;
			float p2 = 0f;
			float p3 = 0f;
			Node pre1 = nodeMap.get(new Node(node.getKey()-1,node.getValue()));
			Node pre2 = nodeMap.get(new Node(node.getKey()-1,node.getValue()-1));
			Node pre3 = nodeMap.get(new Node(node.getKey()-1,node.getValue()-2));
			if (pre1 != null) {
				if (pre1.getFp() == 0f) {
					computeNodeForwardProbability(l, pre1, preOutput, nodeMap);
				}
				p1 = pre1.getFp();
			}
			if (pre2 != null) {
				if (pre2.getFp() == 0f) {
					computeNodeForwardProbability(l, pre2, preOutput, nodeMap);
				}
				p2 = pre2.getFp();
			}
			if (pre3 != null) {
				if (pre3.getFp() == 0f) {
					computeNodeForwardProbability(l, pre3, preOutput, nodeMap);
				}
				p3 = pre3.getFp();
			}
			node.setFp((p1+p2+p3)*preOutput.getScalar(new int[]{l[node.getValue()], node.getKey()}).getFloat(0));
		}
		return ;
	}
	
	/**
	 * 计算后向概率
	 * 
	 * @param l
	 * @param node
	 * @param preOutput
	 * @param nodeMap
	 */
	private static void computeNodeBackProbability(int[] l, Node node, INDArray preOutput, Map<Node,Node> nodeMap) {
		if (node.getKey() == timeSeriesLength -1 && (node.getValue() == l.length-1 || node.getValue() == l.length-2)) {
			node.setBp(preOutput.getScalar(new int[]{l[node.getValue()], node.getKey()}).getFloat(0));
			return ;
		}
		if (!nodeMap.containsKey(node)) {
			return ;
		}
		// 当前seq(s)为空符号或者当前seq(s) == seq(s+2)
		if (l[node.getValue()] == 0 || (node.getValue() +2 <= l.length-1 && l[node.getValue()] ==l[node.getValue() +2]) ) {
			float p1 = 0f;
			float p2 = 0f;
			Node pre1 = nodeMap.get(new Node(node.getKey()+1,node.getValue()));
			Node pre2 = nodeMap.get(new Node(node.getKey()+1,node.getValue()+1));
			if (pre1 != null) {
				if (pre1.getBp() == 0f) {
					computeNodeBackProbability(l, pre1, preOutput, nodeMap);
				}
				p1 = pre1.getBp();
			}
			if (pre2 != null) {
				if (pre2.getBp() == 0f) {
					computeNodeBackProbability(l, pre2, preOutput, nodeMap);
				}
				p2 = pre2.getBp();
			}
			node.setBp((p1+p2)*preOutput.getScalar(new int[]{l[node.getValue()], node.getKey()}).getFloat(0));
		} else {
			float p1 = 0f;
			float p2 = 0f;
			float p3 = 0f;
			Node pre1 = nodeMap.get(new Node(node.getKey()+1,node.getValue()));
			Node pre2 = nodeMap.get(new Node(node.getKey()+1,node.getValue()+1));
			Node pre3 = nodeMap.get(new Node(node.getKey()+1,node.getValue()+2));
			if (pre1 != null) {
				if (pre1.getBp() == 0f) {
					computeNodeBackProbability(l, pre1, preOutput, nodeMap);
				}
				p1 = pre1.getBp();
			}
			if (pre2 != null) {
				if (pre2.getBp() == 0f) {
					computeNodeBackProbability(l, pre2, preOutput, nodeMap);
				}
				p2 = pre2.getBp();
			}
			if (pre3 != null) {
				if (pre3.getBp() == 0f) {
					computeNodeBackProbability(l, pre3, preOutput, nodeMap);
				}
				p3 = pre3.getBp();
			}
			node.setBp((p1+p2+p3)*preOutput.getScalar(new int[]{l[node.getValue()], node.getKey()}).getFloat(0));
		}
		return ;
	}
	
	/**
	 * 计算有效节点
	 * 
	 * @param l
	 * @param node
	 * @param validNodes
	 * @return
	 */
	private static boolean computeValidNodes(int[] l , Node node,Map<Node,Node> validNodes){
		int t = node.getKey();
		int s = node.getValue();
		if (t == timeSeriesLength -1) {
			// 终点必须落在最后两个字符
			if (s < l.length-2){
				return false;
			}
		}
		List<Node> nextNodes = getNextNodes(l,node);
		if (nextNodes == null) {
			validNodes.put(node,node);
			return true;
		}
		int count = 0;
		for (Node next:nextNodes) {
			if (computeValidNodes(l,next,validNodes)) {
				validNodes.put(node,node);
			} else {
				count ++;
			}
		}
		if (count == nextNodes.size()) {
			return false;
		}
		validNodes.put(node,node);
		return true;
	} 
	
	public static void main(String[] args){
		int[] l =  new int[] {0, 10, 0, 9, 0, 1, 0, 10, 0};
		Node node0 = new Node(0,0);
		Node node1 = new Node(0,1);
		Map<Node,Node> validNodes = new HashMap<>();
		//validNodes.put(node0,node0);
		//validNodes.put(node1,node1);
		computeValidNodes(l,node0,validNodes);
		computeValidNodes(l,node1,validNodes);
		//Map<Integer,List<Node>> timeSeriesMap = validNodes.values().stream().collect(Collectors.groupingBy(Node::getKey,LinkedHashMap::new, Collectors.toList()));
		for (Node node:validNodes.values().stream().sorted().collect(Collectors.toList())) {
			System.out.println(node);
		}
	}
	
	//-	o	o	o
	//x	o	o	o	o
	//-		o	o	o	
	//5		o	o	o	o	
	//-			o	o	o	
	//g			o	o	o	o
	//-				o	o	o
	//n				o	o	o	o
	//-					o	o	o
	//x					o	o	o	o
	//-						o	o	o
	//	1	2	3	4	5	6	7	8
	
	//-	o	o	o
	//1	o	o	o	o	o
	//-		o	o	o	o
	//2		o	o	o	o	o
	//-			o	o	o	o
	//3			o	o	o	o	o
	//-				o	o	o	o
	//4				o	o	o	o	o
	//-					o	o	o	o
	//	1	2	3	4	5	6	7	8
	
	static class Node implements Comparable<Node>{
		private int key;
		private int value;
		private float fp;
		private float bp;
		
		public Node(int k, int v) {
			this.key = k;
			this.value = v;
		}

		public int getKey() {
			return key;
		}

		public void setKey(int key) {
			this.key = key;
		}

		public int getValue() {
			return value;
		}

		public void setValue(int value) {
			this.value = value;
		}

		public float getFp() {
			return fp;
		}

		public void setFp(float fp) {
			this.fp = fp;
		}

		public float getBp() {
			return bp;
		}

		public void setBp(float bp) {
			this.bp = bp;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (this.getKey() != ((Node)obj).getKey()) {
				return false;
			}
			if (this.getValue() != ((Node)obj).getValue()) {
				return false;
			}
			return true;
		}
		
		@Override
		public int hashCode() {
			int result = this.getKey();
			result = 31 * result + this.getValue();
			return result;
		}

		@Override
		public String toString() {
			return this.getKey()+"-"+this.getValue()+"-"+this.getFp()+"-"+this.getBp()+",";
		}

		@Override
		public int compareTo(Node o) {
			if (this.getKey() < o.getKey()) {
				return 1;
			} else if(this.getKey() == o.getKey()){
				if (this.getValue() < o.getValue()) {
					return 1;
				} else {
					return -1;
				}
			} else {
				return -1;
			}
		}
		
	}
	
}
