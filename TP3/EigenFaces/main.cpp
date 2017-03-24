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

double dist(const std::vector<double>& currImgWi, const std::vector<double>& datasetImgWi){
    double res = 0;
    for(int i = 0; i < currImgWi.size(); ++i)
    {
        res += pow(datasetImgWi.at(i) - currImgWi.at(i),2);
    }
    return res;
}

int main()
{
    cv::Mat avgImg;
    avgImg.create(112, 92,CV_32FC1);
    std::map<int, string> mapData;
    for(int i = 1; i <= 40; ++i)
    {
        for(int j = 1; j <= 10; ++j)
        {
            std::stringstream ss;
            string path = "/home/pierre/Dev/INF6803/TP3/DATA/";
            ss << path << "s" << i << "_" << j << ".pgm";
            mapData.insert(std::pair<int,string>((i-1)*10+j-1, ss.str()));

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
            string path = "/home/pierre/Dev/INF6803/TP3/DATA/";
            ss << path << "s" << i << "_" << j << ".pgm";
            cv::Mat img = cv::imread(ss.str(), IMREAD_GRAYSCALE);

            cv::subtract(img, avgImg, img);
            img = img.reshape(0,112*92);
            img.col(0).copyTo(matrix.col((i-1)*10+j-1));
        }
    }
    std::cout << matrix.rows << " " << matrix.cols << std::endl;
    matrix.convertTo(matrix, CV_64F);

    cv::Mat matrixT;
    cv::transpose(matrix, matrixT);

    Mat cov, mu;
    cov = matrixT * matrix;
    std::cout << cov.size() << std::endl;
//    cv::calcCovarMatrix(matrixT, cov, mu, CV_COVAR_NORMAL | CV_COVAR_COLS);

//    cov = cov / (matrix.rows - 1);
    std::cout << "ol" << std::endl;
    cv::Mat eVa, eVe;
    cv::eigen(cov, eVa, eVe);
//    std::cout << eVe << std::endl;
//    std::cout << eVa << std::endl;
    std::cout << eVa.rows << " " << eVa.cols << std::endl;
    std::cout << eVe.rows << " " << eVe.cols << " " << eVe.type() << std::endl;
    std::cout << eVa.at<double>(0,0) << std::endl;
    double v = eVa.at<double>(0,0) ;
    cv::Mat res(112*92,400, CV_64F);
    std::set<MyEigen> mySet;
    for(int i = 0; i < eVe.cols; ++i){
        res.col(i) = matrix * eVe.col(i);
        cv::normalize(res.col(i),res.col(i));
        mySet.insert(MyEigen(eVa.at<double>(i,0),i));
    }
    double s = 0;
    for(int i = 0; i < res.rows;++i)
        s += res.at<double>(i,2);
    std::cout << "##  " << s << std::endl;
    int count = 0;
    string path = "/home/pierre/Dev/INF6803/TP3/DATA/";

    cv::Mat image = cv::imread( "/home/pierre/Dev/INF6803/TP3/DATA/l.png", IMREAD_GRAYSCALE);
    std::cout << image.size() << std::endl;
    cv::subtract(image, avgImg, image);
    image = image.reshape(0,112*92);
    image.convertTo(image, CV_64F);

    cv::Mat phi_hat;

    std::vector<double> phi;
    phi_hat.create(112 * 92,1 ,CV_64F);

    for(std::set<MyEigen>::iterator i = mySet.begin(); i != mySet.end(); i++) {
        if(count++ == 9) break;
        cv::Mat tmp = res.col((*i).index);
        cv::Mat tmpT, wi;
        tmp.convertTo(tmp, CV_64F);
        cv::transpose(tmp,tmpT);
        wi = tmpT * image;
        phi.push_back(wi.at<double>(0,0));
        cv::Mat res_ite;
        double wi_fact = wi.at<double>(0,0);
        res_ite = wi_fact * tmp;
        cv::accumulate(res_ite, phi_hat);


    }
    std::vector<std::vector<double>> dataset_wi;
    for(int i = 1; i <= 40; ++i)
    {
        for(int j = 1; j <= 10; ++j)
        {
            std::stringstream ss;
            string path = "/home/pierre/Dev/INF6803/TP3/DATA/";
            ss << path << "s" << i << "_" << j << ".pgm";
            cv::Mat img = cv::imread(ss.str(), IMREAD_GRAYSCALE);
            cv::subtract(img, avgImg, img);
            img = img.reshape(0,112*92);
            img.convertTo(img, CV_64F);
            std::vector<double> tmp;
            int counter = 0;
            for(std::set<MyEigen>::iterator i = mySet.begin(); i != mySet.end(); i++) {
                if(counter++ == 9) break;
                cv::Mat ui = res.col((*i).index);
                cv::transpose(ui,ui);
                ui.convertTo(ui, CV_64F);
                cv::Mat m =  ui*img;
                double res = m.at<double>(0,0);
                tmp.push_back(res);
            }
            dataset_wi.push_back(tmp);
        }
    }
    for(auto i : phi)
        std::cout << i << std::endl;

    double best_dist = 1000000000, best_idx = -1;
    for(int i = 0; i < dataset_wi.size(); ++i)
    {
        double res = dist(phi, dataset_wi.at(i));
        std::cout << res << std::endl;
        if(res < best_dist){
            best_dist = res;
            best_idx = i;
        }
    }

    std::cout << "(" << best_dist << ", " << best_idx << ")" << " => " << mapData.at(best_idx) << std::endl;




    imshow("OL", avgImg);
    cv::waitKey(0);
    namedWindow("OL");
    return 0;
}
