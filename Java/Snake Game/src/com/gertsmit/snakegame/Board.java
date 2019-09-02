package com.gertsmit.snakegame;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

import javax.swing.ImageIcon;
import javax.swing.JPanel;
import javax.swing.Timer;

public class Board extends JPanel implements ActionListener {

	private final int B_WIDTH = 300;
	private final int B_HEIGHT = 300;
	private final int DOT_SIZE = 10;
	private final int ALL_DOTS = 900;
	private final int RAND_POS = 29;
	private final int DELAY = 140;
	
	private final int x[] = new int[ALL_DOTS];
	private final int y[] = new int[ALL_DOTS];
	
	private int dots;
	private int apple_x;
	private int apple_y;
	
	private boolean leftDirection = false;
	private boolean rightDirection = true;
	private boolean upDirection = false;
	private boolean downDirection = false;
	private boolean inGame = true;
	
	private Timer timer;
	private Image ball;
	private Image apple;
	private Image head;
	
	public Board() {
		addKeyListener(new TAdapter());
		setBackground(Color.black);
		setFocusable(true);
		setPreferredSize(new Dimension(B_WIDTH, B_HEIGHT));
		loadImages();
		initGame();
	}
	
	//load necessary images/sprites
	private void loadImages() {
		ImageIcon iid = new ImageIcon(getClass().getResource("res/dot.png"));
		ball = iid.getImage();
		
		ImageIcon iia = new ImageIcon(getClass().getResource("res/apple.png"));
		apple = iia.getImage();
		
		ImageIcon iih = new ImageIcon(getClass().getResource("res/head.png"));
		head = iih.getImage();
	}
	
	//initiate the game
	private void initGame() {
		//set initial snake body size
		dots = 3;
		//place the dots on the game to create the snake
		for (int z = 0; z < dots; z++) {
			x[z] = 50 - z * 10;
			y[z] = 50;
		}
		
		locateApple();
		
		//we use a timer on a timer to call action performed method at fixed delay
		timer = new Timer(DELAY, this);
		timer.restart();
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		doDrawing(g);
	}
	//draw the images in the game (head and body)
	private void doDrawing(Graphics g) {
		//while the game is not over, draw a new apple on the board and add the extra body part
		if (inGame) {
			g.drawImage(apple, apple_x, apple_y, this);
			
			for (int z = 0; z < dots; z++) {
				if (z == 0) {
					g.drawImage(head, x[z], y[z], this);
				} else {
					g.drawImage(ball, x[z], y[z], this);
				}
			}
			
			Toolkit.getDefaultToolkit().sync();
		} else {
			gameOver(g);
		}
	}
	
	private void gameOver(Graphics g) {
		//set the game over text if the game has ended
		String msg = "Game Over";
		Font small = new Font("Helvetica", Font.BOLD, 14);
		FontMetrics metr = getFontMetrics(small);
		
		g.setColor(Color.white);
		g.setFont(small);
		g.drawString(msg, (B_WIDTH - metr.stringWidth(msg)) / 2, B_HEIGHT / 2);
	}
	
	private void checkApple() {
		//check if the player hit the apple and add a body part
		if((x[0] == apple_x) && (y[0] == apple_y)) {
			dots++;
			locateApple();
		}
	}
	
	private void move() {
		//move the player and check what directional button the player pressed
		for(int z = dots; z > 0; z--) {
			x[z] = x[z-1];
			y[z] = y[z-1];
		}
		
		if(leftDirection) {
			x[0] -= DOT_SIZE;
		}
		
		if(rightDirection) {
			x[0] += DOT_SIZE;
		}
		
		if(upDirection) {
			y[0] -= DOT_SIZE;
		}
		
		if(downDirection) {
			y[0] += DOT_SIZE;
		}
	}
	
	private void checkCollision() {
		//check if the player crashed and end the game if so
		for (int z = dots; z > 0; z--) {
			if ((z > 4) && (x[0] == x[z]) && (y[0] == y[z])) {
				inGame = false;
			}
		}
		
		if(y[0] >= B_HEIGHT) {
			inGame = false;
		}
		
		if(y[0] < 0) {
			inGame = false;
		}
		
		if(x[0] >= B_WIDTH) {
			inGame = false;
		}
		
		if(x[0] < 0) {
			inGame = false;
		}
		
		if(!inGame) {
			timer.stop();
		}
	}
	
	private void locateApple() {
		//set the new x and y values for the random apple
		int r = (int)(Math.random() * RAND_POS);
		apple_x = r * DOT_SIZE;
		
		r = (int)(Math.random() * RAND_POS);
		apple_y = r * DOT_SIZE;
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		//create the game movement
		if (inGame) {
			checkApple();
			checkCollision();
			move();
		}
		
		repaint();
	}
	
	private class TAdapter extends KeyAdapter {
		//get the pressed key from the keyboard
		@Override
		public void keyPressed(KeyEvent e) {
			int key = e.getKeyCode();
			
			if(key == KeyEvent.VK_LEFT && !rightDirection) {
				leftDirection = true;
				upDirection = false;
				downDirection = false;
			}
			
			if(key == KeyEvent.VK_RIGHT && !leftDirection) {
				rightDirection = true;
				upDirection = false;
				downDirection = false;
			}
			
			if(key == KeyEvent.VK_UP && !downDirection) {
				upDirection = true;
				rightDirection = false;
				leftDirection = false;
			}
			
			if(key == KeyEvent.VK_DOWN && !upDirection) {
				downDirection = true;
				rightDirection = false;
				leftDirection = false;
			}
		}
	}

}
