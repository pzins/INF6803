#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <fstream>
#include "common.hpp"

#define NB_TEST_IMG 48
#define NB_KNOWN_TEST_IMG 40
#define NB_PERSONS 40
#define NB_IMAGES_PER_PERSON 9
#define NB_IMAGES NB_PERSONS * NB_IMAGES_PER_PERSON

#define IMG_W 92
#define IMG_H 112

#define DISTANCE_FACE_THRESHOLD 400
#define DISTANCE_SPACE_THRESHOLD 0.42
#define SHOW_EIGENFACE true //No to activate if K is high
#define ALL_TRAINING_IMG true
#define THRESHOLD_K 1


//class representing a Face
class Face{
private:
    const std::string name;
    cv::Mat data;
    int numPerson; //numero of the person in the training set
    int numImage; //numero of the image for the person
    std::vector<double> wi; //wi coefficient
    int nbImg; //total number of images for one person

public:
    Face(const std::string& _name, cv::Mat& _data, int _numPerson, int _numImage, int _nbImg) :
        name(_name), data(_data), numPerson(_numPerson), numImage(_numImage), nbImg(_nbImg) {}
    int getNumPerson() const {return numPerson;}
    int getNumImage() const {return numImage;}
    int getNbImage() const {return nbImg;}
    cv::Mat& getData(){return data;}
    std::vector<double>& getWi(){return wi;}
    const std::string getName() const {return name;}
    void setWi(int i, double v){wi.at(i) = v;}
    void initWi(int nb){wi.assign(nb, 0);}
};


//declaration of functions
double dist(const cv::Mat& matA, const cv::Mat& matB);
cv::Mat train(cv::Mat& images, cv::Mat& eigenVectors, cv::Mat& avgImg);
void test(cv::Mat weightsTrain, cv::Mat& eigenVectors, cv::Mat &avgImg);
std::vector<std::tuple<int, float> > identify(cv::Mat &weightsTrain, cv::Mat &wis);
void loadData(cv::Mat &images, cv::Mat& avgImg);




int main()
{
    //training
    //load data into faces
    std::vector<Face> faces;
    cv::Mat avgImg(IMG_H, IMG_W, CV_64F, cv::Scalar(0));
    cv::Mat images(IMG_H*IMG_W, NB_IMAGES, CV_64F);
    loadData(images, avgImg);

    cv::Mat eigenVectors;
    cv::Mat weightsTrain;
    //train the model
    weightsTrain = train(images, eigenVectors, avgImg);

    //test with new images
    test(weightsTrain, eigenVectors, avgImg);
    return 0;
}


//display image stored in vector (eigenface, image projection in face space)
void displayImgVec(const cv::Mat& img, int index){
    /*
    cv::Mat tmp;
    img.copyTo(tmp);
    tmp = tmp.reshape(0,IMG_H);

    //normalize between 0 and 1
    double mi, ma;
    cv::minMaxLoc(tmp, &mi, &ma);
    tmp += fabs(mi);
    tmp /= fabs(ma)+fabs(mi);
*/
    //display
    std::stringstream ss;
    ss << "Window_" << index;
    cv::namedWindow(ss.str());
    cv::imshow(ss.str(), img);
}


//load training data in faces, and compute the mean of all the images
void loadData(cv::Mat& images, cv::Mat &avgImg){
    for(int i = 1; i <= NB_PERSONS; ++i)
    {
        //load each face in the vector
        for(int j = 1; j <= NB_IMAGES_PER_PERSON; ++j)
        {
            std::stringstream ss, s;
            ss << i << "_" << j <<".pgm";
            s << TRAINING_DATA_PATH << ss.str();
            cv::Mat img = cv::imread(s.str(), cv::IMREAD_GRAYSCALE);
            img.convertTo(img, CV_64F);
            avgImg += img;
            img.reshape(0, IMG_H*IMG_W).copyTo(images.col((i-1)*NB_IMAGES_PER_PERSON + j-1));
        }
    }
    //compue the mean
    avgImg /= NB_IMAGES;
}

std::vector<std::tuple<int, float> > identify(cv::Mat& weightsTrain, cv::Mat& wis)
{
    std::vector<std::tuple<int,float>> results;
    for(int i = 0; i < wis.cols; ++i)
    {
        double best_dist = std::numeric_limits<int>::max(), best_idx = -1;
        for(int j = 0; j < weightsTrain.cols; ++j)
        {
            double res = dist(wis.col(i), weightsTrain.col(j));
            if(res < best_dist){
                best_dist = res;
                best_idx = j;
            }
        }
        results.push_back(std::make_tuple(int(best_idx/9), best_dist));
    }
    std::cout << results.size() << std::endl;
    return results;
}


//try to recognize test faces
void test(cv::Mat weightsTrain, cv::Mat& eigenVectors, cv::Mat& avgImg)
{
    //test with images
    float score = 0;
    cv::Mat testImages(IMG_H*IMG_W, NB_TEST_IMG, CV_64F);
    //loop over the test images
    for(int i = 1; i <= NB_TEST_IMG; ++i)
    {
        //load image
        std::stringstream ss;
        ss << TEST_DATA_PATH << i << ".pgm";
        cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
        image.convertTo(image, CV_64F);
        //subtract mean and reshape into vector
        cv::subtract(image, avgImg, image);
        image = image.reshape(0,IMG_H*IMG_W);
        image.copyTo(testImages.col(i-1));
    }
    cv::Mat wis = eigenVectors.t() * testImages;

    //identification of the face
    std::vector<std::tuple<int, float>> results = identify(weightsTrain, wis);

    for(int i = 0; i < results.size(); ++i)
    {
        std::cout << "Image test person " << i << " => " << std::get<0>(results.at(i)) << "   (distance " << std::get<1>(results.at(i))<< ")" << std::endl;
        if(i == std::get<0>(results.at(i)))
            score++;

    }
    //show results
    std::cout << "=============================" << std::endl;
    std::cout << "Identifications : " << score << "/" << NB_TEST_IMG << std::endl;
    std::cout << "Pourcentage : " << score / NB_TEST_IMG << std::endl;

}

cv::Mat train(cv::Mat &images, cv::Mat& eigenVectors, cv::Mat& avgImg)
{
    /*
    cv::namedWindow("ol");
    cv::normalize(avgImg, avgImg, 0, 1, cv::NORM_MINMAX);
    cv::imshow("ol", avgImg);
    cv::waitKey();
    */

    //subtract the images mean, reshape images into vectors and store them in a matrix
    for(int i = 0; i < images.cols; ++i)
        images.col(i) -= avgImg.reshape(0, IMG_H*IMG_W);
    cv::Mat matrix;
    images.copyTo(matrix);

    //transpose matrix
    cv::Mat matrixT = matrix.t();

    //compute covariance matrix
    //matrixT * matrix instead of matrix * matrixT : to limit the dimensionnality
    cv::Mat cov = (matrixT * matrix) / matrix.cols;

    //compute eigenvalues and eigenvectors of matrixT * matrix
    cv::Mat eVa, eVe;
    cv::eigen(cov, eVa, eVe);

    //find the best number K
    double deno = cv::sum(eVa)[0];
    double num = 0;
    int K = 0;
    do
    {
        num += eVa.at<double>(K++,0);
    } while(num / deno <= THRESHOLD_K);
    K++;
//    K = eVe.cols;
    std::cout << K << std::endl;
    //init eigenVector, because now we know its size
    eigenVectors.create(IMG_H*IMG_W, K, CV_64F);

    eigenVectors = matrix * eVe;
    //compute real Eigenvector of matrix * matrixT
    for(int i = 0; i < eigenVectors.cols; ++i){
//        cv::normalize(eigenVectors.col(i), eigenVectors.col(i),1,0,cv::NORM_L2,-1, cv::noArray());
        cv::normalize(eigenVectors.col(i), eigenVectors.col(i),0,1,cv::NORM_MINMAX);
    }

    /*
    // print eigenfaces
    for(int i = 0; i < 10; ++i){
    cv::Mat tmp = eigenVectors.col(i).clone();
    cv::normalize(tmp, tmp, 0, 1, cv::NORM_MINMAX);
    tmp = tmp.reshape(0, IMG_H);
    cv::namedWindow("l");
    cv::imshow("l", tmp);
    cv::waitKey();
    }
    */

    cv::Mat wis(K,images.cols, CV_64F, cv::Scalar(0));
    wis = eigenVectors.t() * images;


    for(int im = 0;  im < images.cols; ++im)
    {
        cv::Mat res(IMG_H*IMG_W, 1, CV_64F, cv::Scalar(0));
        for(int l = 0; l < K; ++l){
            cv::Mat tmp;
            eigenVectors.col(l).copyTo(tmp);
            res += wis.at<double>(l, im) * tmp;
        }
        res = res.reshape(0, IMG_H);
        cv::normalize(res, res, 0, 1, cv::NORM_MINMAX);

        /*
        cv::namedWindow("res");
        cv::Mat t = images.col(im).clone();
        cv::normalize(t,t,0,1, cv::NORM_MINMAX);
//        cv::imshow("res", t.reshape(0, IMG_H));
        cv::imshow("res", res);
        cv::waitKey();
        */
    }
    std::cout << "Training done" << std::endl;
    return wis;
}

//compute euclidian distance between two cv::mat
double dist(const cv::Mat &matA, const cv::Mat &matB)
{
    cv::Mat t = matA - matB, res;
    cv::sqrt(t.mul(t), res);
    return res.at<double>(0,0) / matA.rows;
}









/*
using namespace cv;
using namespace std;

int taille=112*92;
int nb_image = 40 * 9;
string root = "/home/pierre/Dev/INF6803/EigenFaces/DATA/training/";
string root_test = "/home/pierre/Dev/INF6803/EigenFaces/DATA/test/";
double mini, maxi, Dist,energie,seuil=0;
vector<float> resultats, eigenvalues;

int main() {

    // Déclaration Variable
    namedWindow("mywindow");
    namedWindow("mywindow2");

    Mat img,D, Deigenvectors, base_connaissance, rebuild_data, rebuild_vector, rebuild_image, test,test_proj,comp;
    Mat rec(nb_image,taille, CV_64F);

    img.convertTo(img, CV_64F);
    D.convertTo(D, CV_64F);
    Deigenvectors.convertTo(Deigenvectors, CV_64F);
    base_connaissance.convertTo(base_connaissance, CV_64F);
    rebuild_data.convertTo(rebuild_data, CV_64F);
    rebuild_vector.convertTo(rebuild_vector, CV_64F);
    rebuild_image.convertTo(rebuild_image, CV_64F);
    test_proj.convertTo(test_proj, CV_64F);
    comp.convertTo(comp, CV_64F);




    // Conversion en ton de gris + reshape
    for (int i = 1; i < 41; i++) {
        for (int j = 1; j < 10; j++) {
            stringstream sstr;
            sstr << root  << i << "_" << j << ".pgm";
            string path = sstr.str();
            img = imread(path,IMREAD_GRAYSCALE);
            //imshow("mywindow", img_gray);
            //waitKey(0);

            img.reshape(0, 1).copyTo(rec.row((i - 1) * 9 + j - 1));
        }
    }


    // Calcul mean et eigenvector
    Scalar tempVal = mean(rec);
    float mean_rec = tempVal.val[0];
    Mat rec_mean(nb_image,taille, CV_64F, mean_rec);



    rec = rec - rec_mean;
    D = rec*rec.t() / nb_image;
    eigen(D, eigenvalues, Deigenvectors);

    energie = sum(eigenvalues)[0];

    int i = 0;
    while (seuil < 0.85*energie) {
        seuil += eigenvalues.at(i);
        i++;
    }
    i = rec.rows;
    cout << "On a conserve " << i << " vecteurs propre" << endl;
    Mat Ceigenvectors(nb_image, i, CV_64F);

    for (int k = 0; k < i; k++) {
        for (int j = 0; j < Deigenvectors.rows; j++) {
            Ceigenvectors.at<float>(j, k) = Deigenvectors.at<float>(j, k);
        }
    }
    for (int i = 0; i < Ceigenvectors.cols; i++) {
        normalize(Ceigenvectors.col(i), Ceigenvectors.col(i), 1, 0, NORM_L2, -1, noArray());
    }

    std::cout << Ceigenvectors.size() << std::endl;
    std::cout << rec.size() << std::endl;
    // Création base de données

    base_connaissance = Ceigenvectors.t()*rec;
    std::cout << base_connaissance.size() << std::endl;
    //base_connaissance = matrice avc chaque lg le vec propre refait (celui avc la bonne taille de N*N)
    rebuild_data = Ceigenvectors*base_connaissance;

    rebuild_data.row(nb_image-1).copyTo(rebuild_vector);
    rebuild_vector.reshape(0, 112).copyTo(rebuild_image);

    minMaxLoc(rebuild_image, &mini, &maxi);

    rebuild_image = rebuild_image - mini;

    minMaxLoc(rebuild_image, &mini, &maxi);

    rebuild_image = rebuild_image / maxi;

    imshow("mywindow", img);
    imshow("mywindow2", rebuild_image);


    cv::waitKey(0);

    // TEST image dataset et conversion

    stringstream sstr;
    sstr << root_test << "1.pgm";
    string path = sstr.str();
    test = imread(path,IMREAD_GRAYSCALE);

    Mat test_mean(112, 92, CV_64F, mean_rec);

    test.convertTo(test, CV_64F);

    test = test - test_mean;

    // Projection et calcul distance

    test_proj = test.reshape(0, 1)*base_connaissance.t();
    comp = rec*base_connaissance.t();

    for (int i = 0; i < Ceigenvectors.rows; i++) {
        Dist = 0.0;
        for (int j = 0; j < Ceigenvectors.cols; j++){
            Dist += pow((test_proj.at<double>(0,j) - comp.at<double>(i,j)),2);
        }
        Dist = pow(Dist, 0.5);
        resultats.push_back(Dist);
    }
    mini= min_element(resultats.begin(), resultats.end()) - resultats.begin();
    cout << mini/9+1 << endl;
    waitKey(0);
    return 0;
}
*/
