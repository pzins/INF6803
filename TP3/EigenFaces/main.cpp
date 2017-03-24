#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <set>


using	namespace	std;
using	namespace	cv;


struct MyEigen {
    double value;
    int index;
    MyEigen(double _value, int _index) : value(_value), index(_index){}
    bool operator<(const MyEigen &other) const {
        return value < other.value;
    }

//    bool operator <(const MyEigen a, const MyEigen b){
//        return a.value < b.value;
//    }
};

int main()
{
    cv::Mat avgImg;
    avgImg.create(112, 92,CV_32FC1);

    for(int i = 1; i <= 40; ++i)
    {
        for(int j = 1; j <= 10; ++j)
        {
            std::stringstream ss;
            string path = "/home/pierre/Downloads/ALL/";
            ss << path << "s" << i << "_" << j << ".pgm";
            cv::Mat img = cv::imread(ss.str(), IMREAD_GRAYSCALE);
            cv::accumulate(img, avgImg);
        }
    }
    avgImg /= 400;
    avgImg.convertTo(avgImg, CV_8UC1);

    cv::Mat matrix(112*92, 400, CV_32FC1);
    for(int i = 1; i <= 40; ++i)
    {
        for(int j = 1; j <= 10; ++j)
        {
            std::stringstream ss;
            string path = "/home/pierre/Downloads/ALL/";
            ss << path << "s" << i << "_" << j << ".pgm";
            cv::Mat img = cv::imread(ss.str(), IMREAD_GRAYSCALE);

            cv::absdiff(img,avgImg,img);
            img = img.reshape(0,112*92);
            img.col(0).copyTo(matrix.col((i-1)*10+j-1));
        }
    }
    std::cout << matrix.rows << " " << matrix.cols << std::endl;
    matrix.convertTo(matrix, CV_64F);

    cv::Mat matrixT;
    cv::transpose(matrix, matrixT);

    Mat cov, mu;
    cv::calcCovarMatrix(matrixT, cov, mu, CV_COVAR_NORMAL | CV_COVAR_COLS);


//    cov = cov / (matrix.rows - 1);
//    std::cout << cov << std::endl;
    std::cout << "ol" << std::endl;
    cv::Mat eVa, eVe;
    cv::eigen(cov, eVa, eVe);
    //eVe type 64F
    std::cout << eVa.rows << " " << eVa.cols << std::endl;
    std::cout << eVe.rows << " " << eVe.cols << " " << eVe.type() << std::endl;
//    std::cout << eVa << std::endl;
    std::cout << eVa.at<double>(0,0) << std::endl;
    double v = eVa.at<double>(0,0) ;
    cv::Mat res(112*92,400, CV_32FC1);
    std::set<MyEigen> mySet;
    for(int i = 0; i < eVe.cols; ++i){
        res.col(i) = matrix * eVe.col(i);
        cv::normalize(res.col(i),res.col(i));
        mySet.insert(MyEigen(eVa.at<double>(i,0),i));
    }
    for(int i = 0; i < 10; ++i)
    {

    }







    imshow("OL", avgImg);
    cv::waitKey(0);
    namedWindow("OL");
    return 0;
}
