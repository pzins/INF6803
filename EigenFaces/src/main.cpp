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
#define NB_PERSONS 40

#define IMG_W 92
#define IMG_H 112

#define DISTANCE_FACE_THRESHOLD 400
#define DISTANCE_SPACE_THRESHOLD 0.1
#define SHOW_EIGENFACE false //No to activate if K is high
#define ALL_TRAINING_IMG true
#define THRESHOLD_K 0.95

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
double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi);
double dist(const cv::Mat& matA, const cv::Mat& matB);
void train(std::vector<Face>& faces, cv::Mat& eigenVectors, cv::Mat& avgImg);
void test(std::vector<Face>& faces, cv::Mat& eigenVectors, cv::Mat &avgImg);
void computeOmegaPhi(cv::Mat& image, cv::Mat& eigenVectors, std::vector<double>& omega, cv::Mat& phi);
int identify(std::vector<Face>& faces, std::vector<double>& omega, cv::Mat &phi, int testImage);
void loadData(std::vector<Face> &faces, cv::Mat& avgImg);




int main()
{
    //training
    //load data into faces
    std::vector<Face> faces;
    cv::Mat avgImg(IMG_H, IMG_W, CV_64F, cv::Scalar(0));
    loadData(faces, avgImg);

    cv::Mat eigenVectors;
    //train the model
    train(faces, eigenVectors, avgImg);

    //test with new images
    test(faces, eigenVectors, avgImg);
    return 0;
}

//load training data in faces, and compute the mean of all the images
void loadData(std::vector<Face>& faces, cv::Mat &avgImg){
    for(int i = 1; i <= NB_PERSONS; ++i)
    {
        //count how many faces for each person
        int counter = 0;
        while(1){
            std::stringstream ss;
            ss << TRAINING_DATA_PATH << i << "_" << ++counter <<".pgm";
            std::ifstream iff(ss.str());
            //test if file exists
            if(!iff){
                counter--;
                break;
            }
        }
        //load each face in the vector
        for(int j = 1; j <= counter; ++j)
        {
            std::stringstream ss, s;
            ss << i << "_" << j <<".pgm";
            s << TRAINING_DATA_PATH << ss.str();
            cv::Mat img = cv::imread(s.str(), cv::IMREAD_GRAYSCALE);
            img.convertTo(img, CV_64F);
            cv::accumulate(img, avgImg); //accumulate images to compute the mean

            Face f(ss.str(), img, i, j, counter);
            faces.push_back(f);

        }
    }
    //compue the mean
    avgImg /= faces.size();
}

//check if the face is recognized, or if it is unknow or if it's not a face
//return 0 : if the new image is unknow or if it's not a face
//return 1:  if the new image has been recognized
int identify(std::vector<Face> &faces, std::vector<double>& omega, cv::Mat& phi, int testImage)
{
    double best_dist = std::numeric_limits<int>::max(), best_idx = -1;
    //we consider 1 omega for 1 person in the training set (mean of wi of all image of the same person)
    if(!ALL_TRAINING_IMG)
    {
        int K = faces.at(0).getWi().size();
        //compute wi mean for each person in training set
        std::vector<std::vector<double>> omegas(NB_PERSONS, std::vector<double>(K, 0));
        for(int i = 0; i < faces.size(); ++i)
        {
            //sum all wi for each person
            for(int k = 0; k < K; ++k)
                omegas.at(faces.at(i).getNumPerson()-1).at(k) += faces.at(i).getWi().at(k);
            //divide by the number of image for each person
            for(int k = 0; k < K; ++k)
                omegas.at(faces.at(i).getNumPerson()-1).at(k) /= faces.at(i).getNbImage();
        }
        //look for the smallest distance between each person in the training set and the new image
        for(int i = 0; i < omegas.size(); ++i)
        {
            double res = dist(omega, omegas.at(i));
            if(res < best_dist){
                best_dist = res;
                best_idx = i;
            }
        }
    }
    //we consider n images for 1 person
    else {
        //look for the shortest distance between each training image (several images for one same person) and hte new image
        for(int i = 0; i < faces.size(); ++i)
        {
            double res = dist(omega, faces.at(i).getWi());
            if(res < best_dist){
                best_dist = res;
                best_idx = faces.at(i).getNumPerson();
            }
        }
    }
    //compute the difference between the image and its projection in the new space
    double res = dist(faces.at(testImage).getData(), phi);

    //if the image and its projection in the new space are too different
    if(res > DISTANCE_SPACE_THRESHOLD){
        std::cout << "Face " << testImage << " : this image is not a face " << best_dist << " " << res << std::endl;
    }
    //if image and its proejction are close but the face is unknown
    else if(best_dist > DISTANCE_FACE_THRESHOLD)
    {
        std::cout << "Face " << testImage << " : this face is unknown " << best_dist << " " << res << std::endl;
    }
    //the face is recognized
    else
    {
        std::cout << "Face " << testImage << " : " << best_idx  << "  " << best_dist << " " << res << std::endl;
        if(testImage == best_idx) return 1;
    }
    return  0;
}


//compute the wi [w1, w2, ...] for the image and compute the projection of the image in the new space
void computeOmegaPhi(cv::Mat &image, cv::Mat& eigenVectors, std::vector<double>& omega, cv::Mat& phi)
{
    for(int i = 0; i < eigenVectors.cols; ++i)
    {
        cv::Mat tmpT, wi;
        cv::Mat tmp = eigenVectors.col(i);
        cv::transpose(tmp,tmpT);
        wi = tmpT * image;
        omega.push_back(wi.at<double>(0,0));
        cv::accumulate(wi.at<double>(0,0) * eigenVectors.col(i), phi);
    }

}

//try to recognize test faces
void test(std::vector<Face>& faces, cv::Mat& eigenVectors, cv::Mat& avgImg)
{
    //test with images
    float score = 0;
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

        //compute omega [w1 w2 ... wK] and phi (the projection of the image in the new space) for the new image
        std::vector<double> omega;
        cv::Mat phi(IMG_H*IMG_W, 1, CV_64F, cv::Scalar(0));
        computeOmegaPhi(image, eigenVectors, omega, phi);

        //identification of the face
        score += identify(faces, omega, phi, i);

    }
    //show results
    std::cout << "=============================" << std::endl;
    std::cout << "Identifications : " << score << "/" << NB_TEST_IMG << std::endl;
    std::cout << "Pourcentage : " << score / NB_TEST_IMG << std::endl;

}

void train(std::vector<Face>& faces, cv::Mat& eigenVectors, cv::Mat& avgImg)
{
    //subtract the images mean, reshape images into vectors and store them in a matrix
    cv::Mat matrix(IMG_H*IMG_W, faces.size(), CV_64F);
    for(int i = 0; i < faces.size(); ++i)
    {
        cv::subtract(faces.at(i).getData(), avgImg, faces.at(i).getData());
        faces.at(i).getData() = faces.at(i).getData().reshape(0,IMG_H*IMG_W);
        faces.at(i).getData().col(0).copyTo(matrix.col(i));
    }

    //transpose matrix
    cv::Mat matrixT;
    cv::transpose(matrix, matrixT);

    //compute covariance matrix
    //matrixT * matrix instead of matrix * matrixT : to limit the dimensionnality
    cv::Mat cov, mu;
    cov = matrixT * matrix;

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
    //init eigenVector, because now we know its size
    eigenVectors.create(IMG_H*IMG_W, K, CV_64F);
    //compute real Eigenvector of matrix * matrixT
    for(int i = 0; i < K; ++i){
        eigenVectors.col(i) = matrix * eVe.col(i);
        //normalization
        cv::normalize(eigenVectors.col(i), eigenVectors.col(i),1);

        //print Eigenfaces
        if(SHOW_EIGENFACE) {
            cv::Mat tmp(eigenVectors.rows,1,CV_64F);
            eigenVectors.col(i).copyTo(tmp);
            tmp = tmp.reshape(0,IMG_H);
            //normalize between 0 and 1
            double mi, ma;
            cv::minMaxLoc(tmp, &mi, &ma);
            tmp += fabs(mi);
            tmp /= fabs(ma)+fabs(mi);
            //display
            std::stringstream ss;
            ss << "EigenVector_" << i;
            cv::namedWindow(ss.str());
            cv::imshow(ss.str(), tmp);
        }
    }


    //compute wi for all training set images
    for(int i = 0; i < faces.size(); ++i)
    {
        faces.at(i).initWi(K);
        for(int k = 0; k < K; ++k)
        {
            cv::Mat ui = eigenVectors.col(k);
            cv::transpose(ui,ui);
            cv::Mat wi =  ui*faces.at(i).getData();
            faces.at(i).setWi(k, wi.at<double>(0,0));
        }
    }
    std::cout << "Training done" << std::endl;
}

//compute euclidian distance between two cv::mat
double dist(const cv::Mat &matA, const cv::Mat &matB)
{
    cv::Mat t = matA - matB, res;
    cv::sqrt(t.mul(t), res);
    return res.at<double>(0,0) / matA.rows;
}

//compute euclidian distance between two vectors
double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi)
{
    double res = 0;
    for(int i = 0; i < currImgWi.size(); ++i)
        res += sqrt(pow(datasetImgWi.at(i) - currImgWi.at(i),2));
    return res/currImgWi.size();
}
