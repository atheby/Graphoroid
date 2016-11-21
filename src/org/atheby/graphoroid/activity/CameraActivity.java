package org.atheby.graphoroid.activity;

import android.view.*;
import android.app.Activity;
import android.os.Bundle;
import android.widget.SeekBar;
import android.widget.SeekBar.*;
import org.atheby.graphoroid.R;
import org.atheby.graphoroid.model.*;
import org.opencv.core.*;
import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.*;
import org.opencv.imgproc.Imgproc;
import java.util.*;

public class CameraActivity extends Activity implements CvCameraViewListener2 {

    private CameraBridgeViewBase mOpenCvCameraView;
    private GrayInput grayInput;
    private RGBInput rgbInput;

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

        SeekBar thresholdSeekBar = (SeekBar) findViewById(R.id.threshold);
        thresholdSeekBar.setProgress(40);
        thresholdSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {

            @Override
            public void onProgressChanged(SeekBar seekBar, int value, boolean b) {
                grayInput.setThreshold(value);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
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
        rgbInput = new RGBInput();
        grayInput = new GrayInput();
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        grayInput.setInputFrame(inputFrame.gray());
        rgbInput.setInputFrame(inputFrame.rgba());
        Mat binary = grayInput.getBinary(inputFrame.gray());
        Mat largestContourMask = grayInput.getLargestContourMask(binary);

        Mat empty = Mat.zeros(binary.size(), CvType.CV_8UC1);
        Core.bitwise_and(binary, largestContourMask, binary);

        List<MatOfPoint> circlesContours = grayInput.getCirclesContours(binary);
        Mat circlesMask = Mat.zeros(binary.size(), CvType.CV_8UC1);
        for(int x = 0; x < circlesContours.size(); x++)
            Imgproc.drawContours(circlesMask, circlesContours, x, new Scalar(255, 255, 255), -1);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size (5, 5));
        Imgproc.dilate(circlesMask, circlesMask, kernel, new Point(-1, -1), 3);
        Core.bitwise_and(binary, empty, binary, circlesMask);

        for(int x = 0; x < circlesContours.size(); x++) {
            MatOfPoint2f mMOP2f1 = new MatOfPoint2f();
            Point center = new Point();
            float[] radius = new float[1];
            circlesContours.get(x).convertTo(mMOP2f1, CvType.CV_32FC1);
            Imgproc.minEnclosingCircle(mMOP2f1, center, radius);
            Imgproc.drawContours(rgbInput.getInputFrame(), circlesContours, x, new Scalar(255, 0, 0), -1);
            Imgproc.circle(rgbInput.getInputFrame(), center, (int)  radius[0], new Scalar(0, 200, 255), 1);
            Imgproc.putText(rgbInput.getInputFrame(), Integer.toString(x), new Point(center.x - 5, center.y + 5), Core.FONT_HERSHEY_DUPLEX, 0.5, new Scalar(0, 0, 0));
        }

        return rgbInput.getInputFrame();
    }
}
