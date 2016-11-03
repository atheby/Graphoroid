package org.atheby.graphoroid;

import android.view.*;
import android.app.Activity;
import android.os.Bundle;
import org.opencv.core.*;
import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class CameraActivity extends Activity implements CvCameraViewListener2 {

    private CameraBridgeViewBase mOpenCvCameraView;
    private static final String TAG = "CameraActivity";
    private Mat grayFrame, rgbFrame, cannyEdges;
    private int thresholdCanny = 80;
    private List<MatOfPoint> contours;
    private Rect boundingRect;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.camera_activity);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CaptureView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        cannyEdges = new Mat(width, height, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        grayFrame = inputFrame.gray();
        rgbFrame = inputFrame.rgba();
        Imgproc.pyrDown(grayFrame, grayFrame);
        Imgproc.pyrDown(rgbFrame, rgbFrame);

        Imgproc.equalizeHist(grayFrame, grayFrame);
        Imgproc.GaussianBlur(grayFrame, grayFrame, new Size(5, 5), 0);
        Imgproc.adaptiveThreshold(grayFrame, grayFrame, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, 40);
        Imgproc.Canny(grayFrame, cannyEdges, thresholdCanny, thresholdCanny * 3);
        contours = new ArrayList<>();
        Imgproc.findContours(cannyEdges, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double area = 0;
        int largestContour = 0;
        boundingRect = null;
        for(int x = 0; x < contours.size(); x++)
            if(Imgproc.contourArea(contours.get(x)) > area && Imgproc.contourArea(contours.get(x)) > 1000) {
                area = Imgproc.contourArea(contours.get(x));
                boundingRect = Imgproc.boundingRect(contours.get(x));
                largestContour = x;
            }

        if(boundingRect != null) {
            Point br = boundingRect.br();
            Point tl = boundingRect.tl();
            br.x += 5;
            br.y += 5;
            tl.x -= 5;
            tl.y -= 5;
            Imgproc.rectangle(rgbFrame, br, tl, new Scalar(0, 255, 0), 1);

            Mat mask = Mat.zeros(grayFrame.size(), CvType.CV_8UC1);
            Mat result = Mat.zeros(grayFrame.size(), CvType.CV_8UC1);
            Imgproc.drawContours(mask, contours, largestContour, new Scalar(255, 255 ,255), -1);

            Imgproc.erode(mask, mask, new Mat());
            Core.bitwise_xor(grayFrame, mask, result, mask);
            Imgproc.GaussianBlur(result, result, new Size(3, 3), 0);
            Imgproc.Canny(result, cannyEdges, thresholdCanny, thresholdCanny * 3);
            contours = new ArrayList<>();
            Imgproc.findContours(cannyEdges, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            Point center = new Point();
            float[] radius = new float[1];
            MatOfPoint2f mMOP2f = new MatOfPoint2f();
            for(int x = 0; x < contours.size(); x++) {
                contours.get(x).convertTo(mMOP2f, CvType.CV_32FC2);
                Imgproc.minEnclosingCircle(mMOP2f, center, radius);
                Imgproc.circle(rgbFrame, center, (int)radius[0], new Scalar(80, 200, 255), -1);
                Imgproc.putText(rgbFrame, Integer.toString(x + 1), new Point(center.x - 5, center.y + 5), Core.FONT_HERSHEY_DUPLEX, 0.5, new Scalar(0, 0 ,0));
            }
        }

        Imgproc.pyrUp(rgbFrame, rgbFrame);
        return rgbFrame;
    }
}
