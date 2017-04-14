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

#define NB_PERSONS 40
#define NB_IMAGES_PER_PERSON 9
#define NB_IMAGES NB_PERSONS * NB_IMAGES_PER_PERSON

#define IMG_W 92
#define IMG_H 112

//number of test image
#define NB_TEST_IMG 49
//number of known face among the test set
#define NB_KNOWN_TEST_IMG 40

//two thresholds
#define DISTANCE_FACE_THRESHOLD 3.4e8
#define DISTANCE_SPACE_THRESHOLD 4e8

#define THRESHOLD_K 0.95

#define SHOW_RECONSTRUCTION false


//declaration of functions
cv::Mat train(cv::Mat& images, cv::Mat& wis, cv::Mat& avgImg);
void test(cv::Mat eigVect, cv::Mat &avgImg, cv::Mat matrix);
std::vector<std::tuple<int, float> > identify(cv::Mat &weightsTrain, cv::Mat &weightsTest);
double dist(const cv::Mat& matA, const cv::Mat& matB);

int main()
{

    cv::Mat avgImg(IMG_H, IMG_W, CV_64F, cv::Scalar(0));
    cv::Mat images(IMG_H*IMG_W, NB_IMAGES, CV_64F);
    cv::Mat wis;

    //train the model
    train(images, wis, avgImg);

    //test with new images
    test(wis, avgImg, images);
    return 0;
}

//perform the training phase
cv::Mat train(cv::Mat &images, cv::Mat& wis, cv::Mat& meanImage)
{
    //load data
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
            meanImage += img;
            img.reshape(0, IMG_H*IMG_W).copyTo(images.col((i-1)*NB_IMAGES_PER_PERSON + j-1));
        }
    }
    //compute the mean
    meanImage /= NB_IMAGES;

    //subtract the images mean, reshape images into vectors and store them in a matrix
    for(int i = 0; i < images.cols; ++i)
        images.col(i) -= meanImage.reshape(0, IMG_H*IMG_W);

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
    std::cout << "Nous conservons " << K << " vecteurs propres." << std::endl;

    // matrix with only K eigenvectors
    cv::Mat eigVec(eVe.rows, K, CV_64F);

    //normalization of the eigenvector
    for(int i = 0; i < K; ++i){
        cv::normalize(eVe.col(i), eVe.col(i),1,0,cv::NORM_L2,-1, cv::noArray());
        eVe.col(i).copyTo(eigVec.col(i));
    }

    wis.create(IMG_H*IMG_W, K, CV_64F);
    wis = matrix * eigVec;

    //reconstruct images
    cv::Mat rebuildImages = eigVec * wis.t();
    for(int im = 0;  im < images.cols; ++im)
    {
        cv::Mat reconstruction(1, IMG_H*IMG_W, CV_64F);
        rebuildImages.row(im).copyTo(reconstruction);
        reconstruction = reconstruction.reshape(0, IMG_H);
        reconstruction += meanImage;
        reconstruction /= 255;
        //to show reconstruction of the training images
        if(SHOW_RECONSTRUCTION){
            cv::namedWindow("res");
            cv::imshow("res", reconstruction);
            cv::waitKey();
        }
    }
    return eigVec;
}


//try to recognize test faces
void test(cv::Mat eigVect, cv::Mat& avgImg, cv::Mat matrix)
{
    float score = 0;
    cv::Mat testImages(IMG_H*IMG_W, NB_TEST_IMG, CV_64F);
    //load test images
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

    //compute weights for test and train images
    cv::Mat weightsTest = testImages.t() * eigVect;
    cv::Mat weightsTrain = matrix.t()*eigVect;

    //identification of the face (tuple : face numero, distance value)
    std::vector<std::tuple<int, float>> results = identify(weightsTrain, weightsTest);

    //loop over test faces to show the identification result
    for(int i = 1; i <= NB_TEST_IMG; ++i)
    {
        double distance = std::get<1>(results.at(i-1));
        int face = std::get<0>(results.at(i-1));
        if(distance > DISTANCE_SPACE_THRESHOLD){
            std::cout << "Image test person " << i << " => Not a face (distance " << distance << ")" << std::endl;
        } else if(distance > DISTANCE_FACE_THRESHOLD){
            std::cout << "Image test person " << i << " => face is unknown (distance " << distance << ")" << std::endl;

        } else {
            std::cout << "Image test person " << i << " => " << face << "   (distance " << distance << ")";
            if(i == face){
                std::cout << std::endl;
                score++;
            } else {
                std::cout << " Error" << std::endl;
            }
        }

    }
    //show results
    std::cout << "=============================" << std::endl;
    std::cout << "Identifications : " << score << "/" << NB_KNOWN_TEST_IMG << std::endl;
    std::cout << "Pourcentage : " << score / NB_KNOWN_TEST_IMG << std::endl;

}


//identify test faces
std::vector<std::tuple<int, float> > identify(cv::Mat& weightsTrain, cv::Mat& weightsTest)
{
    std::vector<std::tuple<int,float>> results;
    //loop over test images
    for(int i = 0; i < weightsTest.rows; ++i)
    {
        //loop over train images
        double best_dist = std::numeric_limits<int>::max(), best_idx = -1;
        for(int j = 0; j < weightsTrain.rows; ++j)
        {
            //compute distance
            double res = dist(weightsTest.row(i), weightsTrain.row(j));
            if(res < best_dist){
                best_dist = res;
                best_idx = j;
            }
        }
        results.push_back(std::make_tuple(int(best_idx/9)+1, best_dist));
    }
    return results;
}


//compute euclidian distance between the weights
double dist(const cv::Mat &matA, const cv::Mat &matB)
{
    cv::Mat t = matA - matB, res;
    cv::sqrt(t.mul(t), res);
    return cv::sum(res)[0]/matA.rows;
}

