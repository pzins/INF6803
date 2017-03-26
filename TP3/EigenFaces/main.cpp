#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>

#define NB_PERSONS 40
#define NB_IMG_PER_PERSON 9
#define DATASET NB_IMG_PER_PERSON * NB_PERSONS
#define IMG_W 92
#define IMG_H 112
#define K 10
#define DISTANCE_THRESHOLD 2500
#define SHOW_EIGENFACE false //No to activate if K is high



double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi);
void EigenFaces(cv::Mat& eigenVectors, std::vector<std::vector<double>>& dataset_wi, cv::Mat& avgImg, std::map<int, std::string>& mapData);
void computeOmega(cv::Mat& image, std::vector<double>& omega, cv::Mat& eigenVectors);
int identify(std::vector<std::vector<double>>& dataset_wi, std::vector<double>& omega);

int main()
{
    //training
    cv::Mat eigenVectors(IMG_H*IMG_W, K, CV_64F);
    std::vector<std::vector<double>> dataset_wi;
    cv::Mat avgImg;
    std::map<int, std::string> mapData;
    EigenFaces(eigenVectors, dataset_wi, avgImg, mapData);

    //test with images
    float score = 0;
    for(int i = 1; i < NB_PERSONS; ++i)
    {
        std::stringstream ss;
        std::string path = "/home/pierre/Dev/INF6803/TP3/DATA/test/";
        ss << path << "s" << i << "_10.pgm";
        cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
        cv::subtract(image, avgImg, image);
        image = image.reshape(0,IMG_H*IMG_W);
        image.convertTo(image, CV_64F);


        //compute omega [w1 w2 ... wK] for the new image
        std::vector<double> omega;
        computeOmega(image, omega, eigenVectors);

        //identification
        int res = identify(dataset_wi, omega);
        if(res == -1)
        {
            std::cout << "Ce visage est inconnu" << std::endl;
        }
        else
        {
            std::cout << "Visage " << i << " : " << mapData.at(res) << std::endl;
            if(i-1 == (int) res / NB_IMG_PER_PERSON) score++;
        }
    }
    std::cout << "=============================" << std::endl;
    std::cout << "Identifications : " << score << "/" << NB_PERSONS << std::endl;
    std::cout << "Pourcentage : " << score / NB_PERSONS << std::endl;


    return 0;
}

int identify(std::vector<std::vector<double>>& dataset_wi, std::vector<double>& omega)
{
    //check if the face is known and who it is
    double best_dist = std::numeric_limits<int>::max(), best_idx = -1;
    for(int i = 0; i < dataset_wi.size(); ++i)
    {
        double res = dist(omega, dataset_wi.at(i));
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

void EigenFaces(cv::Mat& eigenVector, std::vector<std::vector<double> > &dataset_wi, cv::Mat &avgImg, std::map<int, std::__cxx11::string> &mapData){
    avgImg.create(IMG_H, IMG_W, CV_64F);
    std::map<int, cv::Mat> faces;
    for(int i = 1; i <= NB_PERSONS; ++i)
    {
        for(int j = 1; j <= NB_IMG_PER_PERSON; ++j)
        {
            std::stringstream ss, s;
            std::string path = "/home/pierre/Dev/INF6803/TP3/DATA/training/";
            ss << path << "s" << i << "_" << j << ".pgm";
            cv::Mat img = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
            int index = (i-1) * NB_IMG_PER_PERSON + j - 1;

            //save images of faces
            faces.insert(std::pair<int, cv::Mat>(index, img));

            //sum all faces to compute the mean
            cv::accumulate(img, avgImg);


            //just for diplay purpose
            s << "s" << i << "_" << j << ".pgm";
            mapData.insert(std::pair<int,std::string>(index, s.str()));
        }
    }
    //compue the mean
    avgImg /= DATASET;
    //convert because faces.at(i) is of type CV_8UC1 and subtract only work with same type
    avgImg.convertTo(avgImg, CV_8UC1);

    //compute mean for each face images + reshape + store in a matrix
    cv::Mat matrix(IMG_H*IMG_W, DATASET, CV_64F);
    for(int i = 0; i < faces.size(); ++i)
    {
        cv::subtract(faces.at(i), avgImg, faces.at(i));
        faces.at(i) = faces.at(i).reshape(0,IMG_H*IMG_W);
        faces.at(i).col(0).copyTo(matrix.col(i));

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


    //debug eigenvectors
    /*
    double summ = 0;
    for(int i = 0; i < IMG_H*IMG_W; ++i)
    {
        summ += pow(realEve.at<double>(i,0),2);
    }
    std::cout << std::endl << sqrt(summ) << std::endl;
    return 0;
    */

    //compute wi for all training set images
    for(int i = 0; i < faces.size(); ++i)
    {
        std::vector<double> tmp;
        for(int j = 0; j < K; ++j)
        {
            cv::Mat ui = eigenVector.col(j);
            cv::transpose(ui,ui);
            faces.at(i).convertTo(faces.at(i), CV_64F);
            cv::Mat wi =  ui*faces.at(i);
            tmp.push_back(wi.at<double>(0,0));
        }
        dataset_wi.push_back(tmp);
    }

    /*
    for(int i = 0; i < dataset_wi.size(); ++i)
    {
        for(int j = 0; j < dataset_wi.at(i).size(); ++j)
        {
            std::cout << dataset_wi.at(i).at(j) << " ";
        }
        std::cout << std::endl;
    }
    */
}

double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi)
{
    double res = 0;
    for(int i = 0; i < currImgWi.size(); ++i)
        res += sqrt(pow(datasetImgWi.at(i) - currImgWi.at(i),2));
    return res;
}
