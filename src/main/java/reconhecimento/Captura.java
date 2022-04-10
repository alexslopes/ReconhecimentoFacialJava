package reconhecimento;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.awt.event.KeyEvent;
import java.util.Scanner;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
//Para iniciar a camera, digite um inteiro que representará um id,
//Quando a janela da cam abrir, selecion-a e digite 'q' para capturar a foto
public class Captura {
    public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException {
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src/main/java/recursos/haarcascade_frontalface_alt.xml");

        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;
        Mat imagemColorida = new Mat();
        int numeroAmostras = 25;
        int amostra = 1;
        System.out.println("Digite seu id: ");
        Scanner cadastro = new Scanner(System.in);
        int idPessoa = cadastro.nextInt();
        while((frameCapturado = camera.grab()) != null) {
            imagemColorida = new Mat();
            imagemColorida = converteMat.convert(frameCapturado);
            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
            RectVector facesDetectadas = new RectVector();
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0,
                    new Size(150, 150), //Tamanho mínimo que uma face pode ter para detectar
                    new Size(500, 500)); //tamanho máximo que uma face pode ter
            if (tecla == null) {
                tecla = cFrame.waitKey(5);
            }

            for(int i =0 ; i< facesDetectadas.size(); i++) {
                Rect dadosFace = facesDetectadas.get(0);//Variável que desenha retangulo em volta da face
                rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));//Exibe retangulo em torno da face
                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new Size(160,160));//redimensiona as imagens para um tamanho padrao
                if (tecla == null) {
                    tecla = cFrame.waitKey(5);
                }

                if(tecla != null) {
                    if(tecla.getKeyChar() == 'q') {
                        if(amostra <= numeroAmostras) {
                            imwrite("src\\fotos\\pessoa." + idPessoa + "." + amostra + ".jpg", faceCapturada);
                            System.out.println("Foto " + amostra + " capturada\n");
                            amostra++;
                        }
                    }
                    tecla = null;
                }
            }
            if (tecla == null) {
                tecla = cFrame.waitKey(20);
            }

            if(cFrame.isVisible()){
                cFrame.showImage(frameCapturado);
            }

            if (amostra > numeroAmostras) {
                break;
            }
        }
        cFrame.dispose();
        camera.stop();
    }
}
