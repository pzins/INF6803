#include <tp2/common.hpp>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <set>

#define PI 3.14159265
#define NB_PARTICULES 20
#define ANGLE_DIVISION 15 //should divide 360
#define NB_BEST_PARTICULES 10 //number of best particules to compute new box coordinate

class Particule
{
private:
    double distance;
    double distanceHOG;
    cv::Rect shape;
public:
    Particule(cv::Rect rec) : shape(rec), distance(0), distanceHOG(0) {}
    const cv::Rect& getShape() const {return shape;}
    void setShape(const cv::Rect& rec){shape = rec;}
    void setHOGDistance(double value){distanceHOG = value;}
    void setDistance(double value){distance = value;}
    double getHOGDistance() const {return distanceHOG;}
    double getDistance() const {return distance;}


    //choose between distance, HOG_distance, distance + HOG_distance
    bool operator< (const Particule& other) const {
        return distance < other.distance;
        return distanceHOG+distance < other.distanceHOG+other.distance;
        return distanceHOG< other.distanceHOG;
    }
    bool operator> (const Particule& other) const {
        return distance > other.distance;
        return distanceHOG+distance > other.distanceHOG+other.distance;
        return distanceHOG > other.distanceHOG;
    }
    double getTotalDistance() const{
        return getHOGDistance()+getDistance();
        return getDistance();
        return getHOGDistance();
    }
};
std::ostream& operator<<(std::ostream& os, Particule& obj)
{
    os << obj.getShape();
    return os;
}


class MyTracker : public Tracker
{
private:
    std::vector<float> histogram;
    std::vector<float> HOGhistogram;

    //histogram initial comme reference pour ajouter des distances basées sur les comparaisons avec ceux là
    std::vector<float> histogram_ref;
    std::vector<float> HOGhistogram_ref;
    cv::Rect myBox;
    std::vector<Particule> particules;
public:
    void initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox);
    void apply(const cv::Mat& oCurrFrame, cv::Rect& oOutputBBox);
    void printParticules();
    void fillNewParticules(const cv::Mat& ocurrFrame, cv::Rect particule);
};



std::shared_ptr<Tracker> Tracker::createInstance(){
    return std::shared_ptr<Tracker>(new MyTracker());
}

void MyTracker::printParticules(){
    std::cout << "----------------------------------" << std::endl;
    for(auto i : particules)
        std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "----------------------------------" << std::endl;
}


std::vector<float> getHOGDescriptor(const cv::Mat& frame_){
    cv::HOGDescriptor d(cv::Size(16,16), cv::Size(16,16), cv::Size(16,16), cv::Size(8,8),9);
    std::vector<float> desc;
    d.compute(frame_.clone(), desc);
//    std::cout << desc.size() << std::endl;
//    for(auto i : desc)
//        std::cout << i << " ";
//    std::cout << std::endl;
    return desc;
}


std::vector<float> getHistogram(const cv::Mat& frame_){
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;

    cv::Mat frame = frame_.clone();

    cvtColor( frame_, frame, CV_BGR2GRAY );

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    // Gradient X
    cv::Sobel( frame, grad_x, ddepth, 1, 0, 1, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    // Gradient Y
    cv::Sobel( frame, grad_y, ddepth, 0, 1, 1, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    // Sum gradient X and Y
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, frame);

    std::vector<float> histo(360,0);

    //sum of gradient of the frame
    double sum_gradient = cv::sum(frame)[0];

    for(int i = 0; i < frame.rows; ++i){
        for(int j = 0; j < frame.cols; ++j){
            //compute the angle
            float angle = atan2((float)grad_y.at<int16_t>(i,j), (float)grad_x.at<int16_t>(i,j)) * 180 / PI;
            //to get gradient direction between 0-360
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


float getDistanceHistogram(const std::vector<float>& refHist, const std::vector<float>& currHist){

    float somme = 0;
    for(int i = 0; i < refHist.size(); ++i)
    {
        somme += sqrt(currHist.at(i) * currHist.at(i));
    }
//    return -log(somme);

//    double num = 0, deno=0;
//    for(int i = 0; i < refHist.size(); ++i){
//        num += pow(refHist.at(i)-currHist.at(i),2);
//        deno += refHist.at(i) + currHist.at(i);
//    }
//    return num / deno;


    cv::MatND hist(currHist);
    cv::MatND hist2(refHist);
    // reutrn Bhattacharyya distance
    // max with very small number to prevent very small value which cause infintie number later
    double ret = cv::compareHist(hist, hist2, CV_COMP_BHATTACHARYYA);
    return ret;
}


void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    particules.clear();

    //init myBox and the associated histogram
    myBox = oInitBBox;
    histogram = getHistogram(oInitFrame(myBox));
    histogram_ref = histogram;
    HOGhistogram = getHOGDescriptor(oInitFrame(myBox));
    HOGhistogram_ref = HOGhistogram;


    //create particules from myBox
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5,0.5);
    for(int i = 0; i < NB_PARTICULES-1; ++i){
        fillNewParticules(oInitFrame, myBox);
    }
    //also add the current myBox to the particules
    particules.push_back(Particule(myBox));
}


void MyTracker::fillNewParticules(const cv::Mat& oCurrFrame, cv::Rect particule)
{
    // set real new particules
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5,0.5);
    double x = particule.x+0.5*round(particule.width*dis(gen));
    double y = particule.y+0.5*round(particule.height*dis(gen));
    double regionSizeW = std::max(1.0, std::min((double)oCurrFrame.size().width, particule.width + 0*round((particule.width/20)*dis(gen))));
    double regionSizeH = std::max(1.0, std::min((double)oCurrFrame.size().height, particule.height + 0*round((particule.height/20)*dis(gen))));
    x = std::max(0.0, std::min(x, oCurrFrame.cols-regionSizeW));
    y = std::max(0.0, std::min(y, oCurrFrame.rows-regionSizeH));
    particules.push_back(cv::Rect(x, y, regionSizeW, regionSizeH));
}

struct mycompare {
    bool operator() (const Particule& p1, const Particule& p2) const{
        return p1.getDistance() < p2.getDistance();
    }
};

void MyTracker::apply(const cv::Mat &oCurrFrame, cv::Rect &oOutputBBox)
{
    double sum_particule_distance = 0;
    double score_min = 100000;
    int nb_best_score = 0;
    std::set<Particule, mycompare> best_particules;
    for(int i = 0; i < particules.size(); ++i)
    {
        //print particules (!!! affect gradient computation)
//        cv::rectangle(oCurrFrame, cv::Point(particules.at(i).getShape().x, particules.at(i).getShape().y),
//                      cv::Point(particules.at(i).getShape().x+particules.at(i).getShape().size().width,
//                                particules.at(i).getShape().y+particules.at(i).getShape().size().height), cv::Scalar(255,255,255));


        // compute distance between each particules histogram and the oOutputBBox histogram at the previous frame
        double res = getDistanceHistogram(histogram, getHistogram(oCurrFrame(particules.at(i).getShape())));
//        res += getDistanceHistogram(histogram_ref, getHistogram(oCurrFrame(particules.at(i).getShape())));
        particules.at(i).setDistance(res);


        //same but with HOG descriptor
        double resHOG = getDistanceHistogram(HOGhistogram, getHOGDescriptor(oCurrFrame(particules.at(i).getShape())));
        resHOG += getDistanceHistogram(HOGhistogram_ref, getHOGDescriptor(oCurrFrame(particules.at(i).getShape())));
        particules.at(i).setHOGDistance(resHOG);

        if(best_particules.begin() == best_particules.end())
        {
            if(best_particules.size() < NB_BEST_PARTICULES || *best_particules.begin() > particules.at(i))
            {
                best_particules.insert(particules.at(i));
            }
        }
        else
        {
            if(best_particules.size() < NB_BEST_PARTICULES || *best_particules.rbegin() > particules.at(i))
            {
                best_particules.insert(particules.at(i));
                if( best_particules.size() > NB_BEST_PARTICULES)
                    best_particules.erase(std::prev(best_particules.end()));
            }
        }


        // get the total distance
        sum_particule_distance += res;
    }


    //compute positions and size of the box (mean from best particules)
    double somme_distance = 0;
    float x_ = 0, y_ = 0, w_ = 0, h_ = 0;
    for(auto i : best_particules)
    {
        somme_distance += i.getDistance();
        x_ += i.getShape().x;
        y_ += i.getShape().y;
        w_ += i.getShape().size().width;
        h_ += i.getShape().size().height;
    }


    double bests_size = best_particules.size();
    //mean
    x_ /= bests_size;
    y_ /= bests_size;
    w_ /= bests_size;
    h_ /= bests_size;
    myBox = cv::Rect(x_, y_, w_, h_);
    oOutputBBox = myBox;

    // update histogram
    histogram=getHistogram(oCurrFrame(myBox));
    HOGhistogram=getHOGDescriptor(oCurrFrame(myBox));

    double norm = 0;
    for(auto i : best_particules)
        norm += somme_distance-i.getDistance();

    double sum_inv_dist = 0;
    for(auto i : best_particules)
    {
        sum_inv_dist += somme_distance-i.getDistance();
    }

    particules.clear();


    int counter = 0;
    for(auto i : best_particules)
    {
        for(int j = 0; j < NB_PARTICULES*(somme_distance-i.getDistance())/norm; ++j)
        {
            fillNewParticules(oCurrFrame, i.getShape());
        }
    }
    std::cout << "SIZE P = " <<particules.size() << std::endl;
}

