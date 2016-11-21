package org.atheby.graphoroid.model;

import org.opencv.core.*;
import org.opencv.imgproc.*;
import java.util.*;

public class GrayInput {

    private Mat inputFrame;
    private int threshold = 50;

    public void setInputFrame(Mat m) {
        this.inputFrame = m;
    }

    public void setThreshold(int t) {
        this.threshold = t;
    }

    public Mat getBinary(Mat input) {
        Imgproc.equalizeHist(input, input);
        Imgproc.GaussianBlur(input, input, new Size(3, 3), 0);
        Imgproc.adaptiveThreshold(input, input, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, threshold);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size (5, 5), new Point(1, 1));
        Imgproc.morphologyEx(input, input, Imgproc.MORPH_CLOSE, kernel);
        return input;
    }

    public Mat getLargestContourMask(Mat input) {
        Mat mask = Mat.zeros(input.size(), CvType.CV_8UC1);
        double area = 0;
        int largestContour = 0;
        List<MatOfPoint> contours = getContours(input, Imgproc.RETR_EXTERNAL);

        for(int x = 0; x < contours.size(); x++)
            if(Imgproc.contourArea(contours.get(x)) > area && Imgproc.contourArea(contours.get(x)) > 1000) {
                area = Imgproc.contourArea(contours.get(x));
                largestContour = x;
            }

        if(area != 0) {
            Imgproc.drawContours(mask, contours, largestContour, new Scalar(255, 255, 255), -1);
        }
        return mask;
    }

    public List<MatOfPoint> getCirclesContours(Mat input) {
        List<MatOfPoint> contours = getContours(input, Imgproc.RETR_LIST);
        List<MatOfPoint> circles = new ArrayList<>();

        if(contours.size() < 25)
            for(int x = 0; x < contours.size(); x++) {
                MatOfPoint2f mMOP2f1 = new MatOfPoint2f();
                MatOfPoint2f mMOP2f2 = new MatOfPoint2f();
                Point center = new Point();
                float[] radius = new float[1];
                double contourArea = Imgproc.contourArea(contours.get(x));

                contours.get(x).convertTo(mMOP2f1, CvType.CV_32FC1);
                Imgproc.minEnclosingCircle(mMOP2f1, center, radius);
                if((Math.PI * Math.pow(radius[0], 2)) / contourArea < 2.7 && contourArea > 50 && contourArea < 1000) {
                    Imgproc.approxPolyDP(mMOP2f1, mMOP2f2, Imgproc.arcLength(mMOP2f1, true) * 0.02, true);
                    if(mMOP2f2.rows() > 6)
                        circles.add(contours.get(x));
                }
            }

        return circles;
    }

    private List<MatOfPoint> getContours(Mat input, int mode) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat copy = new Mat();
        input.copyTo(copy);
        Imgproc.findContours(copy, contours, new Mat(), mode, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours;
    }
}
