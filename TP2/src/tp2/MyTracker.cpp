#include <tp2/common.hpp>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#define PI 3.14159265
#define NB_PARTICULES 100
#define ANGLE_DIVISION 15 //better if it divides 360
#define BAG_SIZE 100000
#define NB_BEST_PARTICULES_BOX_COO 69 //number of best particules to compute new box coordinate

class Particule
{
private:
    double distance;
    double distanceHOG;
    cv::Rect shape;
public:
    Particule(cv::Rect rec) : shape(rec){}
    cv::Rect& getShape(){return shape;}
};
std::ostream& operator<<(std::ostream& os, Particule& obj)
{
    // write obj to stream
    os << obj.getShape();
    return os;
}

class MyTracker : public Tracker
{
private:
    std::vector<float> histogram;
    std::vector<float>HOGhistogram;
    cv::Rect myBox;
    std::vector<Particule> particules;
public:
    void initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox);
    void apply(const cv::Mat& oCurrFrame, cv::Rect& oOutputBBox);
    void printParticules();
};

std::vector<float> getHOGDescriptor(const cv::Mat& mat){
    cv::HOGDescriptor d(cv::Size(16,16), cv::Size(16,16), cv::Size(16,16), cv::Size(8,8),9);
    std::vector<float> desc;
    d.compute(mat.clone(), desc);
//    std::cout << desc.size() << std::endl;
//    for(auto i : desc)
//        std::cout << i << " ";
//    std::cout << std::endl;
    return desc;
}

std::shared_ptr<Tracker> Tracker::createInstance(){
    return std::shared_ptr<Tracker>(new MyTracker());
}

std::vector<float> getHistogram(const cv::Mat& frame_){
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;

    cv::Mat frame;

//    cv::GaussianBlur(frame_, frame_, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    cvtColor( frame_, frame, CV_BGR2GRAY );
    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    /// Gradient X
    cv::Sobel( frame, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    cv::Sobel( frame, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, frame);

    std::vector<float> histo(360,0);
    double sum_gradient = cv::sum(frame)[0];
    sum_gradient = std::max(0,1); //a vi

    for(int i = 0; i < frame.rows; ++i){
        for(int j = 0; j < frame.cols; ++j){
            float angle = atan2((float)grad_y.at<int16_t>(i,j), (float)grad_x.at<int16_t>(i,j)) * 180 / PI;
            if(angle<0)
                angle = 180 + (180 + angle)-1;
            histo.at(static_cast<int>(angle)) += frame.at<uint8_t>(i,j) / sum_gradient;
        }
    }

    //merge angle directions in ANGLE_DIVISION groups
    std::vector<float> res;
    for(int i = 0; i < ANGLE_DIVISION; ++i)
    {
        double tmp = 0;
        for(int j = 0; j < 360/ANGLE_DIVISION; ++j){
            tmp += histo.at(i * 360/ANGLE_DIVISION + j);
        }
        res.push_back(tmp);
    }
    return res;
}

void MyTracker::printParticules(){
    std::cout << "----------------------------------" << std::endl;
    for(auto i : particules)
        std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "----------------------------------" << std::endl;
}

void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{

    particules.clear();

    //init myBox and the associated histogram
    myBox = oInitBBox;
    histogram = getHistogram(oInitFrame(myBox).clone());
    HOGhistogram = getHOGDescriptor(oInitFrame(myBox));

    //create particules from myBox
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5,0.5);
    for(int i = 0; i < NB_PARTICULES-1; ++i){
        double x = myBox.x+round(2*myBox.width*dis(gen));
        double y = myBox.y+round(2*myBox.height*dis(gen));
        double regionSizeW = myBox.width + 0*round((myBox.width/3)*dis(gen));
        double regionSizeH = myBox.height + 0*round((myBox.height/3)*dis(gen));
        x = std::max(0.0, std::min(x, oInitFrame.cols-regionSizeW));
        y = std::max(0.0, std::min(y, oInitFrame.rows-regionSizeH));
        particules.push_back(Particule(cv::Rect(x,y,regionSizeW, regionSizeH)));
    }
    //also add the current myBox to the particules
    particules.push_back(Particule(myBox));
}


double getDistanceHistogram(const std::vector<float>& refHist, const std::vector<float>& currHist){
    cv::MatND hist(currHist);
    cv::MatND hist2(refHist);
    // get Bhattacharyya distance
    return cv::compareHist(hist, hist2, CV_COMP_BHATTACHARYYA);
}



void MyTracker::apply(const cv::Mat &oCurrFrame, cv::Rect &oOutputBBox)
{


    std::vector<std::tuple<int,float>> particules_index_distance;
    double sum_particule_distance = 0;
    for(int i = 0; i < particules.size(); ++i)
    {
        //print particules (!! affect gradient computation)
//        cv::rectangle(oCurrFrame, cv::Point(particules.at(i).x, particules.at(i).y), cv::Point(particules.at(i).x+particules.at(i).size().width,particules.at(i).y+particules.at(i).size().height), cv::Scalar(0,0,255));

        // compute distance between each particules histogram and the oOutputBBox histogram at the previous frame
        double res = getDistanceHistogram(histogram, getHistogram(oCurrFrame(particules.at(i).getShape()).clone()));
        res = getDistanceHistogram(HOGhistogram, getHOGDescriptor(oCurrFrame(particules.at(i).getShape())));
        // get the total distance
        sum_particule_distance += res;
        // save tuple (particules index, distance)

        particules_index_distance.push_back(std::make_tuple(i, res));
    }

    //simulation tirage avec remise
    //vector with new particules
    std::vector<cv::Rect> newParticules;
    // get normalisation value (low distance == high similarity => more probabilty to be chosen)
    double sum_norm = 0;
    for(auto i : particules_index_distance)
        sum_norm += (1/(std::get<1>(i) / sum_particule_distance));
    std::vector<int> bag;
    // fill bag with particules according to their probability (proportional to the distance)
    for(int i = 0; i < particules_index_distance.size(); ++i)
    {
        float proba = (1/(std::get<1>(particules_index_distance.at(i)) / sum_particule_distance))/sum_norm;
        for(int j = 0; j < proba * BAG_SIZE; ++j)
            bag.push_back(std::get<0>(particules_index_distance.at(i)));
    }

    // shuffle sac
    auto engine = std::default_random_engine{};
    std::shuffle(std::begin(bag), std::end(bag), engine);
    //tirage
    for(int j = 0; j < NB_PARTICULES; ++j)
        newParticules.push_back(particules.at(bag.at(rand() % bag.size())).getShape());

    // sort particules_index_distance according to the distance
    std::sort(
        particules_index_distance.begin(), particules_index_distance.end(),
        [](std::tuple<int, float> a, std::tuple<int, float> b) {
            return std::get<1>(a) < std::get<1>(b);
        }
    );
    //compute coordinate of the new box
    float x_ = 0, y_ = 0, w_ = 0, h_ = 0;
    for(int i = 0; i < NB_BEST_PARTICULES_BOX_COO; ++i)
    {
        x_ += particules.at(std::get<0>(particules_index_distance.at(i))).getShape().x;
        y_ += particules.at(std::get<0>(particules_index_distance.at(i))).getShape().y;
        w_ += particules.at(std::get<0>(particules_index_distance.at(i))).getShape().size().width;
        h_ += particules.at(std::get<0>(particules_index_distance.at(i))).getShape().size().height;
    }
    //mean
    x_ /= NB_BEST_PARTICULES_BOX_COO;
    y_ /= NB_BEST_PARTICULES_BOX_COO;
    w_ /= NB_BEST_PARTICULES_BOX_COO;
    h_ /= NB_BEST_PARTICULES_BOX_COO;
    myBox = cv::Rect(x_, y_, w_, h_);
    oOutputBBox = myBox;

    // update histogram
    histogram=getHistogram(oCurrFrame(myBox).clone());
    HOGhistogram=getHOGDescriptor(oCurrFrame(myBox));

    // set real new particules
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5,0.5);
    for(int i = 0; i < newParticules.size(); ++i)
    {
        double x = newParticules.at(i).x+round(newParticules.at(i).width*dis(gen));
        double y = newParticules.at(i).y+round(newParticules.at(i).height*dis(gen));
        double regionSizeW = std::max(1.0, std::min((double)oCurrFrame.size().width, newParticules.at(i).width + 0*round((newParticules.at(i).width/20)*dis(gen))));
        double regionSizeH = std::max(1.0, std::min((double)oCurrFrame.size().height, newParticules.at(i).height + 0*round((newParticules.at(i).height/20)*dis(gen))));
        x = std::max(0.0, std::min(x, oCurrFrame.cols-regionSizeW));
        y = std::max(0.0, std::min(y, oCurrFrame.rows-regionSizeH));
        particules.at(i) = cv::Rect(x, y, regionSizeW, regionSizeH);
    }


}

