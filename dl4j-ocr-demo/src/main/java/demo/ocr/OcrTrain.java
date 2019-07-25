package demo.ocr;

import java.io.IOException;

public class OcrTrain {
	public static void main(String[] args) {
		Ocr ocr = new Ocr.Builder()
				.setEpochs(100)
				.setBatchSize(15)
				.setDataSetType("train")
				.setTimeSeriesLength(16)
				.setMaxLabelLength(8)
				.setTextChars("_234578acdefgmnpwxy")
				.setDirPath("E:/deeplearning/captcha/")
				.setModelFileName("ocrModel.json")
				.toOcr();
		try {
			ocr.train();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			ocr.predict();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
