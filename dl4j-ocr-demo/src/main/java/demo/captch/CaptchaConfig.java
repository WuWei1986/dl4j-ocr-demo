package demo.captch;

import java.awt.Color;
import java.awt.Font;

public class CaptchaConfig {
	
	private Color backGroundColor;
	
	private Color borderColor;
	
	private int borderX;
	
	private int borderY;
	
	private Color fontColor;
	
	private Font font;
	
	private Boolean noise;

	public Color getBackGroundColor() {
		return backGroundColor;
	}

	public void setBackGroundColor(Color backGroundColor) {
		this.backGroundColor = backGroundColor;
	}

	public Color getBorderColor() {
		return borderColor;
	}

	public void setBorderColor(Color borderColor) {
		this.borderColor = borderColor;
	}

	public int getBorderX() {
		return borderX;
	}

	public void setBorderX(int borderX) {
		this.borderX = borderX;
	}

	public int getBorderY() {
		return borderY;
	}

	public void setBorderY(int borderY) {
		this.borderY = borderY;
	}

	public Color getFontColor() {
		return fontColor;
	}

	public void setFontColor(Color fontColor) {
		this.fontColor = fontColor;
	}

	public Font getFont() {
		return font;
	}

	public void setFont(Font font) {
		this.font = font;
	}

	public Boolean getNoise() {
		return noise;
	}

	public void setNoise(Boolean noise) {
		this.noise = noise;
	}


}
