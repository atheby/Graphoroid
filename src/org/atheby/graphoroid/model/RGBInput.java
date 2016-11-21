package org.atheby.graphoroid.model;

import org.opencv.core.*;

public class RGBInput {

    private Mat inputFrame;

    public void setInputFrame(Mat m) {
        this.inputFrame = m;
    }

    public Mat getInputFrame() {
        return inputFrame;
    }
}
