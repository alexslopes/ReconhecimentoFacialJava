package reconhecimento;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.awt.event.KeyEvent;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Captura {
    public static void main(String[] args) throws FrameGrabber.Exception {
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src/main/java/recursos/haarcascade_frontalface_alt.xml");

        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;
        Mat imagemColorida = new Mat();

        while((frameCapturado = camera.grab()) != null) {
            imagemColorida = new Mat();
            imagemColorida = converteMat.convert(frameCapturado);
            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
            RectVector facesDetectadas = new RectVector();
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0,
                    new Size(150, 150), //Tamanho mínimo que uma face pode ter para detectar
                    new Size(500, 500)); //tamanho máximo que uma face pode ter
            for(int i =0 ; i< facesDetectadas.size(); i++) {
                Rect dadosFace = facesDetectadas.get(0);//Variável que desenha retangulo em volta da face
                rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));
            }

            if(cFrame.isVisible()){
                cFrame.showImage(frameCapturado);
            }
        }
        cFrame.dispose();
        camera.stop();
    }
}
