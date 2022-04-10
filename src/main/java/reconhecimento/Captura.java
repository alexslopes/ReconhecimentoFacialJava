package reconhecimento;

import org.bytedeco.javacv.*;

import java.awt.event.KeyEvent;

public class Captura {
    public static void main(String[] args) throws FrameGrabber.Exception {
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();

        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;
        while((frameCapturado = camera.grab()) != null) {
            if(cFrame.isVisible()){
                cFrame.showImage(frameCapturado);
            }
        }
        cFrame.dispose();
        camera.stop();
    }
}
