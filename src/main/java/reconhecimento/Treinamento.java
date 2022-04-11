package reconhecimento;

import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.opencv.core.CvType.CV_32SC1;

public class Treinamento {
    public static void main(String[] args) {
        File diretorio = new File("src\\fotos");
        FilenameFilter filtroImagem = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String nome) {//Obtem arquivos de imagens
                return nome.endsWith(".jpg") || nome.endsWith(".gif") || nome.endsWith(".png");
            }
        };

        File[] arquivos = diretorio.listFiles(filtroImagem);
        MatVector fotos = new MatVector(arquivos.length);//Cria um vetor de matrizes com a quantidade total de fotos
        Mat rotulos  = new Mat(arquivos.length, 1, CV_32SC1);//Vai especificar a classe da imagem
        IntBuffer rotulosBuffer = rotulos.createBuffer();// armazena corretamente os rotulos
        int contador = 0;
        for (File imagem: arquivos) {
            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);//Obtem o id da imagem
            System.out.println(imagem.getName().split("\\.")[1] + "  " + imagem.getAbsolutePath());
            resize(foto, foto, new Size(160,160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }

        //Gera e salva os classificadores de TODAS as fotos
        FaceRecognizer eigenfaces = EigenFaceRecognizer.create();
        FaceRecognizer fisherfaces = FisherFaceRecognizer.create();
        FaceRecognizer lbph = LBPHFaceRecognizer.create(2,9,9,9,1);

        eigenfaces.train(fotos, rotulos);
        eigenfaces.save("src/main/java/recursos/classificadorEigenFaces.yml");
        fisherfaces.train(fotos, rotulos);
        fisherfaces.save("src/main/java/recursos/classificadorFisherFaces.yml");
        lbph.train(fotos, rotulos);
        lbph.save("src/main/java/recursos/classificadorLBPH.yml");
    }

}
