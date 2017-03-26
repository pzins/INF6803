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

class Face{
private:
    const std::string name;
    cv::Mat data;
    int numPerson;
    int numImage;
    std::vector<double> wi;
    cv::Mat phi;
    int nbImg;
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
    void setPhi(cv::Mat& _phi){phi = _phi;}
};


double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi);
void EigenFaces(cv::Mat& eigenVector, cv::Mat& avgImg, std::vector<Face>& faces);
void computeOmega(cv::Mat& image, std::vector<double>& omega, cv::Mat& eigenVectors);
int identify(std::vector<Face>& faces, std::vector<double>& omega);


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
        computeOmega(image, omega, eigenVectors);

        //identification
        int res = identify(faces, omega); //res = person index
        if(res == -1)
        {
            std::cout << "Ce visage est inconnu" << std::endl;
        }
        else
        {
            std::cout << "Visage " << i << " : " << res << std::endl;
            if(i == res) score++;
        }
    }
    std::cout << "=============================" << std::endl;
    std::cout << "Identifications : " << score << "/" << NB_PERSONS << std::endl;
    std::cout << "Pourcentage : " << score / NB_PERSONS << std::endl;


    return 0;
}

int identify(std::vector<Face> &faces, std::vector<double>& omega)
{
    //check if the face is known and who it is
    double best_dist = std::numeric_limits<int>::max(), best_idx = -1;
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
//        std::cout << res << " -> " << i << std::endl;
        if(res < best_dist){
            best_dist = res;
            best_idx = i;
        }
    }
    if(best_dist < DISTANCE_THRESHOLD) return best_idx;
    return  -1;

}


void computeOmega(cv::Mat &image, std::vector<double>& omega, cv::Mat& eigenVectors)
{
    for(int i = 0; i < K; ++i)
    {
        cv::Mat tmpT, wi;
        cv::Mat tmp = eigenVectors.col(i);
        cv::transpose(tmp,tmpT);
        wi = tmpT * image;
        omega.push_back(wi.at<double>(0,0));
    }
}

void EigenFaces(cv::Mat& eigenVector, cv::Mat& avgImg, std::vector<Face>& faces){
    avgImg.create(IMG_H, IMG_W, CV_64F);

    for(int i = 0; i < faces.size(); ++i)
    {
        cv::accumulate(faces.at(i).getData(), avgImg);
        /*
        //compute all files for each person
        std::vector<int> nbImgPerPerson;
        int counter = 0;
        std::ifstream iff;
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
            std::stringstream ss;
            ss << filename << i << "_" << j <<".pgm";
            cv::Mat img = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
            int index = (i-1) * NB_IMG_PER_PERSON + j - 1;

            //save images of faces
//            faces.insert(std::pair<int, cv::Mat>(index, img));
            tmp.push_back(img);

            //sum all faces to compute the mean

        }
        faces.push_back(tmp);*/
    }
    //compue the mean
    avgImg /= DATASET;
    //convert because faces.at(i) is of type CV_8UC1 and subtract only work with same type
//    avgImg.convertTo(avgImg, CV_8UC1);

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
        cv::Mat phi(IMG_H*IMG_W, 1, CV_64F);
        for(int k = 0; k < K; ++k)
        {
            cv::Mat ui = eigenVector.col(k);
//              std::cout << ui << std::endl;
            cv::transpose(ui,ui);
            faces.at(i).getData().convertTo(faces.at(i).getData(), CV_64F);
            cv::Mat wi =  ui*faces.at(i).getData();
            faces.at(i).setWi(k, wi.at<double>(0,0));
            phi += wi.at<double>(0, 0) * eigenVector.col(k);
        }
        faces.at(i).setPhi(phi);
    }

}

double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi)
{
    double res = 0;
    for(int i = 0; i < currImgWi.size(); ++i)
        res += sqrt(pow(datasetImgWi.at(i) - currImgWi.at(i),2));
    return res;
}
