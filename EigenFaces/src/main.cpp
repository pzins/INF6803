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
#define NB_IMG_PER_PERSON 9
#define DATASET NB_IMG_PER_PERSON * NB_PERSONS
#define IMG_W 92
#define IMG_H 112
#define K 10
#define DISTANCE_THRESHOLD 25000*3
#define SHOW_EIGENFACE false //No to activate if K is high
#define ALL_TRAINING_IMG true

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
        name(_name), data(_data), numPerson(_numPerson), numImage(_numImage), nbImg(_nbImg) {wi.assign(K, 0);}
    int getNumPerson() const {return numPerson;}
    int getNumImage() const {return numImage;}
    int getNbImage() const {return nbImg;}
    cv::Mat& getData(){return data;}
    std::vector<double>& getWi(){return wi;}
    const std::string getName() const {return name;}
    void setWi(int i, double v){wi.at(i) = v;}
};


double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi);
double dist(const cv::Mat& matA, const cv::Mat& matB);
void EigenFaces(cv::Mat& eigenVector, cv::Mat& avgImg, std::vector<Face>& faces);
void computeOmegaPhi(cv::Mat& image, std::vector<double>& omega, cv::Mat& eigenVectors, cv::Mat& phi);
int identify(std::vector<Face>& faces, std::vector<double>& omega, cv::Mat &phi, int testImage);


void loadData(std::vector<Face>& faces){
    for(int i = 1; i <= NB_PERSONS; ++i)
    {
        int counter = 0;
        while(1){
            const std::string filename(TRAINING_DATA_PATH);
            std::stringstream ss;
            ss << filename << i << "_" << ++counter <<".pgm";
            std::ifstream iff(ss.str());
            if(!iff){
                counter--;
                break;
            }
        }
        std::vector<cv::Mat> tmp;
        for(int j = 1; j <= counter; ++j)
        {
            const std::string filename(TRAINING_DATA_PATH);
            std::stringstream ss, s;
            ss << i << "_" << j <<".pgm";
            s << filename << ss.str();
            cv::Mat img = cv::imread(s.str(), cv::IMREAD_GRAYSCALE);
            img.convertTo(img, CV_64F);
            int index = (i-1) * NB_IMG_PER_PERSON + j - 1;
            Face f(ss.str(), img, i, j, counter);
            faces.push_back(f);
        }
    }
}



int main()
{
    //training
    cv::Mat eigenVectors(IMG_H*IMG_W, K, CV_64F);
    std::map<std::string, std::vector<double>> dataset_wi;
    cv::Mat avgImg;
    std::vector<Face> faces;
    loadData(faces);
    EigenFaces(eigenVectors, avgImg, faces);
    std::cout << "Training done" << std::endl;
    std::cout << "============================" << std::endl;

    //test with images
    float score = 0;
    for(int i = 1; i <= NB_TEST_IMG; ++i)
    {
        std::stringstream ss;
        ss << TEST_DATA_PATH << i << ".pgm";
        cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
        image.convertTo(image, CV_64F);
        cv::subtract(image, avgImg, image);
        image = image.reshape(0,IMG_H*IMG_W);


        //compute omega [w1 w2 ... wK] for the new image
        std::vector<double> omega;
        cv::Mat phi(IMG_H*IMG_W, 1, CV_64F, cv::Scalar(0));
        computeOmegaPhi(image, omega, eigenVectors, phi);

        //identification
        score += identify(faces, omega, phi, i); //res = person index

    }
    std::cout << "=============================" << std::endl;
    std::cout << "Identifications : " << score << "/" << NB_TEST_IMG << std::endl;
    std::cout << "Pourcentage : " << score / NB_TEST_IMG << std::endl;


    return 0;
}

int identify(std::vector<Face> &faces, std::vector<double>& omega, cv::Mat& phi, int testImage)
{
    //check if the face is known and who it is
    double best_dist = std::numeric_limits<int>::max(), best_idx = -1;
    if(!ALL_TRAINING_IMG)
    {
        //compute wi mean for each person in training set
        std::vector<std::vector<double>> omegas(NB_PERSONS, std::vector<double>(K, 0));
        for(int i = 0; i < faces.size(); ++i)
        {
            for(int k = 0; k < K; ++k)
                omegas.at(faces.at(i).getNumPerson()-1).at(k) += faces.at(i).getWi().at(k);
            for(int k = 0; k < K; ++k)
                omegas.at(faces.at(i).getNumPerson()-1).at(k) /= faces.at(i).getNbImage();

        }
        for(int i = 0; i < omegas.size(); ++i)
        {
            double res = dist(omega, omegas.at(i));
            if(res < best_dist){
                best_dist = res;
                best_idx = i;
            }
        }
    } else {
        for(int i = 0; i < faces.size(); ++i)
        {
            double res = dist(omega, faces.at(i).getWi());
            if(res < best_dist){
                best_dist = res;
                best_idx = faces.at(i).getNumPerson();
            }
        }
    }
    //compute difference the image and its projection

    double res = dist(faces.at(testImage).getData(), phi);
    if(res > DISTANCE_THRESHOLD){
        std::cout << "This image is not a face" << std::endl;
    }
    else if(best_dist > DISTANCE_THRESHOLD)
    {
        std::cout << "Ce visage est inconnu" << std::endl;
    }
    else
    {
        std::cout << "Visage " << testImage << " : " << best_idx << std::endl;
        if(testImage == best_idx) return 1;
    }
    return  0;

}


void computeOmegaPhi(cv::Mat &image, std::vector<double>& omega, cv::Mat& eigenVectors, cv::Mat& phi)
{
    for(int i = 0; i < K; ++i)
    {
        cv::Mat tmpT, wi;
        cv::Mat tmp = eigenVectors.col(i);
        cv::transpose(tmp,tmpT);
        wi = tmpT * image;
        omega.push_back(wi.at<double>(0,0));
        cv::accumulate(wi.at<double>(0,0) * eigenVectors.col(i), phi);
    }

}

void EigenFaces(cv::Mat& eigenVector, cv::Mat& avgImg, std::vector<Face>& faces){
    avgImg.create(IMG_H, IMG_W, CV_64F);

    for(int i = 0; i < faces.size(); ++i)
    {
        cv::accumulate(faces.at(i).getData(), avgImg);
    }
    //compue the mean
    avgImg /= DATASET;

    //compute mean for each face images + reshape + store in a matrix
    cv::Mat matrix(IMG_H*IMG_W, DATASET, CV_64F);
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


    //compute real Eigenvector of matrix * matrixT
    for(int i = 0; i < K; ++i){
        eigenVector.col(i) = matrix * eVe.col(i);
        //normalization
        cv::normalize(eigenVector.col(i), eigenVector.col(i),1);

        //print Eigenfaces
        if(SHOW_EIGENFACE) {
            cv::Mat tmp(eigenVector.rows,1,CV_64F);
            eigenVector.col(i).copyTo(tmp);
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
        for(int k = 0; k < K; ++k)
        {
            cv::Mat ui = eigenVector.col(k);
            cv::transpose(ui,ui);
            faces.at(i).getData().convertTo(faces.at(i).getData(), CV_64F);
            cv::Mat wi =  ui*faces.at(i).getData();
            faces.at(i).setWi(k, wi.at<double>(0,0));
        }
    }

}

double dist(const cv::Mat &matA, const cv::Mat &matB)
{
    cv::Mat t = matA - matB, res;
    cv::sqrt(t.mul(t), res);
    return res.at<double>(0,0);
}


double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi)
{
    double res = 0;
    for(int i = 0; i < currImgWi.size(); ++i)
        res += sqrt(pow(datasetImgWi.at(i) - currImgWi.at(i),2));
    return res;
}
