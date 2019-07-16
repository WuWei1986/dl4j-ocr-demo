package demo.captch;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

/**
 * 验证码生成工具
 * 
 * @author wuwei
 *
 */
public class CaptchaProducer {

	private final static int captchaLength = 4;
	
	private static final String base = "234578acdefgmnpwxy";
	
	private static Random rand = new Random();

	/**
	 * 创建文本
	 * 
	 * @return
	 */
	public static String createText() {
		StringBuilder sb = new StringBuilder();
		String temp = "";
		for (int i = 0; i < captchaLength; i++) {
			String ch = base.charAt(new Random().nextInt(base.length())) + " ";
			if (ch.equals(temp)) {
				i--;
				continue;
			}
			sb.append(ch);
			temp = ch;
		}
		return sb.toString();
	}

	/**
	 * 创建坐标
	 * 
	 * @return
	 */
	public static List<String> createCoordinates(int width, int height) {
		List<Integer> randomIntList = new ArrayList<Integer>();
		Random random = new Random();
		while (randomIntList.size() < captchaLength) {
			int number = random.nextInt(captchaLength) + 1;
			if (!randomIntList.contains(number)) {
				randomIntList.add(number);
			}
		}
		List<String> coordinates = new ArrayList<>();
		for (int i = 0; i < captchaLength; i++) {
			int x = 0, y = 0;
			int j = randomIntList.get(i);
			x = new Random().nextInt(20) + width * (j-1) / captchaLength;
			y = new Random().nextInt(60) + height-60-10;
			coordinates.add(x + "_" + y);
		}
		return coordinates;
	}

	/**
	 * 创建图片
	 * 
	 * @param captchaText
	 * @param width
	 * @param height
	 * @param coordinates
	 * @param captchaConfig
	 * @return
	 */
	public static BufferedImage createImage(String captchaText, int width, int height, List<String> coordinates,CaptchaConfig captchaConfig) {
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics graphics = img.getGraphics();
		// 设置背景
		if(captchaConfig.getBackGroundColor() != null){
			graphics.setColor(captchaConfig.getBackGroundColor());
			graphics.fillRect(0, 0, width, height);
		}
		
		// 干扰
		if(captchaConfig.getNoise() != null && captchaConfig.getNoise()){
			// 干扰线
			createRandomLine(width, height,50,graphics,100);  
			// 干扰点
			//createRandomPoint(width, height,1000,graphics,100);
		}
		
		// 设置边框
		if(captchaConfig.getBorderColor() != null){
			// 设置边框颜色
			graphics.setColor(captchaConfig.getBorderColor());
			// 设置边框区域
			graphics.drawRect(captchaConfig.getBorderX(), captchaConfig.getBorderY(), width - captchaConfig.getBorderX()<<1, height - captchaConfig.getBorderY()<<1);
		}
		
		// 设置字体颜色
		graphics.setColor(captchaConfig.getFontColor());
		// 设置字体
		graphics.setFont(captchaConfig.getFont());

		for (int i = 0; i < captchaText.length(); i++) {
			String[] xy = coordinates.get(i).split("_");
			int x = Integer.valueOf(xy[0]);
			int y = Integer.valueOf(xy[1]);
			graphics.drawString(String.valueOf(captchaText.charAt(i)), x, y);
		}
		return img;
	}
	
	/**
	 * 创建图片
	 * 
	 * @param captchaText
	 * @param width
	 * @param height
	 * @param x
	 * @param y
	 * @param captchaConfig
	 * @return
	 */
	public static BufferedImage createImage(String captchaText, int width, int height, int x,int y,CaptchaConfig captchaConfig) {
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		
		Graphics graphics = img.getGraphics();
		// 设置背景颜色
		if(captchaConfig.getBackGroundColor() != null){
			graphics.setColor(captchaConfig.getBackGroundColor());
			graphics.fillRect(0, 0, width, height);
		}
		
		// 干扰
		if(captchaConfig.getNoise() != null && captchaConfig.getNoise()){
			// 干扰线
			//createRandomLine(width, height,50,graphics,100);  
			// 干扰点
			//createRandomPoint(width, height,1000,graphics,100);
		}
		
		// 设置边框
		if(captchaConfig.getBorderColor() != null){
			// 设置边框颜色
			graphics.setColor(captchaConfig.getBorderColor());
			// 设置边框区域
			graphics.drawRect(captchaConfig.getBorderX(), captchaConfig.getBorderY(), width - captchaConfig.getBorderX()<<1, height - captchaConfig.getBorderY()<<1);
		}
		
		// 设置字体颜色
		graphics.setColor(captchaConfig.getFontColor());
		// 设置字体
		graphics.setFont(captchaConfig.getFont());
		graphics.drawString(captchaText, x, y);
		return img;
	}
	
	/***
	 * 随机返回一种颜色,透明度0~255 0表示全透
	 * 
	 * @return 随机返回一种颜色
	 * @param alpha
	 *            透明度0~255 0表示全透
	 */
	private static Color getColor(int alpha) {
		int R = (int) (Math.random() * 255);
		int G = (int) (Math.random() * 255);
		int B = (int) (Math.random() * 255);
		return new Color(R, G, B, alpha);
	}

	
	/**
	 * 随机产生干扰线条
	 * 
	 * @param width
	 * @param height
	 * @param minMany
	 *            最少产生的数量
	 * @param g
	 * @param alpha
	 *            透明度0~255 0表示全透
	 */
	private static void createRandomLine(int width, int height, int minMany, Graphics g, int alpha) { // 随机产生干扰线条
		for (int i = 0; i < getIntRandom(minMany, minMany + 6); i++) {
			int x1 = getIntRandom(0, (int) (width * 0.6));
			int y1 = getIntRandom(0, (int) (height * 0.9));
			int x2 = getIntRandom((int) (width * 0.4), width);
			int y2 = getIntRandom((int) (height * 0.2), height);
			g.setColor(getColor(alpha));
			g.drawLine(x1, y1, x2, y2);
		}
	}

	/***
	 * @return 随机返一个指定区间的数字
	 */
	private static int getIntRandom(int start, int end) {
		if (end < start) {
			int t = end;
			end = start;
			start = t;
		}
		int i = start + (int) (Math.random() * (end - start));
		return i;
	}
	
	/**
	 * 随机产生干扰点
	 * 
	 * @param width
	 * @param height
	 * @param many
	 * @param g
	 * @param alpha
	 *            透明度0~255 0表示全透
	 */
	private static void createRandomPoint(int width, int height, int many, Graphics g, int alpha) { // 随机产生干扰点
		for (int i = 0; i < many; i++) {
			int x = rand.nextInt(width);
			int y = rand.nextInt(height);
			g.setColor(getColor(alpha));
			g.drawOval(x, y, rand.nextInt(3), rand.nextInt(3));
		}
	}
	
	/**
	 * 生成验证码
	 * 
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		FileOutputStream out = null;
		for (int i = 0; i < 100; i++) {
			String captchaText = CaptchaProducer.createText();
			CaptchaConfig config = new CaptchaConfig();
			Font font = new Font("微软雅黑", Font.BOLD, 38);
			Color fontColor = new Color(255, 136, 30);
			config.setBackGroundColor(Color.white);
			config.setFont(font);
			config.setFontColor(fontColor);
			config.setNoise(true);
			BufferedImage bi = CaptchaProducer.createImage(captchaText, 160, 60, 5, font.getSize()+8, config);
			out = new FileOutputStream(new File("E://deeplearning/captcha/train/" + captchaText.replaceAll(" ", "") + ".jpg"));
			ImageIO.write(bi, "jpg", out);
			out.close();
		}
	}

}
