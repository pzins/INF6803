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

#define NB_TEST_IMG 40
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
void test(cv::Mat weightsTrain, cv::Mat& eigenVectors, cv::Mat &avgImg, cv::Mat matrix);
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
    test(weightsTrain, eigenVectors, avgImg, images);
    return 0;
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
    std::cout << weightsTrain.size() << " " << wis.size() << std::endl;
    std::vector<std::tuple<int,float>> results;
    //loop over test images
    for(int i = 0; i < wis.rows; ++i)
    {
        //loop over train images
        double best_dist = std::numeric_limits<int>::max(), best_idx = -1;
        for(int j = 0; j < weightsTrain.rows; ++j)
        {
            double res = dist(wis.row(i), weightsTrain.row(j));
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
void test(cv::Mat weightsTrain, cv::Mat& eigenVectors, cv::Mat& avgImg, cv::Mat matrix)
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
//    cv::Mat wis = eigenVectors.t() * testImages;
    cv::Mat wis = testImages.t() * eigenVectors;
    cv::Mat comp = matrix.t()*eigenVectors;
    std::cout << wis.size() << " " << comp.size() << std::endl;
    //identification of the face
    std::vector<std::tuple<int, float>> results = identify(comp, wis);

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

    cv::Mat eigVec(eVe.rows, K, CV_64F);
    //normalization of the eigenvector
    for(int i = 0; i < K; ++i){
        cv::normalize(eVe.col(i), eVe.col(i),1,0,cv::NORM_L2,-1, cv::noArray());
        eVe.col(i).copyTo(eigVec.col(i));
    }
    eigenVectors.create(IMG_H*IMG_W, K, CV_64F);

    eigenVectors = matrix * eigVec;

    cv::Mat res = eigVec * eigenVectors.t();
    for(int im = 0;  im < images.cols; ++im)
    {
        cv::Mat reconstruction(1, IMG_H*IMG_W, CV_64F);
        res.row(im).copyTo(reconstruction);
        reconstruction= reconstruction.reshape(0, IMG_H);
        reconstruction += avgImg;
        reconstruction /= 255;

//        cv::namedWindow("res");
//        cv::imshow("res", reconstruction);
//        cv::waitKey();

    }
    std::cout << "Training done" << std::endl;
    return eigVec;
}

//compute euclidian distance between two cv::mat
double dist(const cv::Mat &matA, const cv::Mat &matB)
{
    cv::Mat t = matA - matB, res;
    cv::sqrt(t.mul(t), res);
    return cv::sum(res)[0]/matA.rows;
}

