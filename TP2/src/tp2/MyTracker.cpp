#include <tp2/common.hpp>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <set>

#define PI 3.14159265
#define NB_PARTICULES 50
#define ANGLE_DIVISION 15 //should divide 360
#define NB_BEST_PARTICULES 5 //number of best particules to compute new box coordinate
#define RANDOM_RANGE 0.5



class Particule
{
private:
    double distanceHOG; //distance with HOG
    cv::Rect shape;

public:
    Particule(cv::Rect rec) : shape(rec), distanceHOG(0) {}
    const cv::Rect& getShape() const {return shape;}
    void setShape(const cv::Rect& rec){shape = rec;}

    void setHOGDistance(double value){distanceHOG = value;}
    double getHOGDistance() const {return distanceHOG;}


    //choose between distance, HOG_distance, distance + HOG_distance
    bool operator< (const Particule& other) const {
        return distanceHOG< other.distanceHOG;
    }
    bool operator> (const Particule& other) const {
        return distanceHOG > other.distanceHOG;
    }


};

//comparaison of Particule for set insert
struct mycompare {
    bool operator() (const Particule& p1, const Particule& p2) const{
        return p1.getHOGDistance() < p2.getHOGDistance();
    }
};



class MyTracker : public Tracker
{
private:
    std::vector<float> HOGhistogram; //histogram HOG

    cv::Rect myBox;
    std::vector<Particule> particules;
public:
    void initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox);
    void apply(const cv::Mat& oCurrFrame, cv::Rect& oOutputBBox);
    //add a new particule based on the particule parameter
    void addParticule(const cv::Mat& ocurrFrame, cv::Rect particule);
};



std::shared_ptr<Tracker> Tracker::createInstance(){
    return std::shared_ptr<Tracker>(new MyTracker());
}


std::vector<float> getHOGDescriptor(const cv::Mat& frame_){

    cv::HOGDescriptor d(frame_.size(), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8),9);
    std::vector<float> desc;
    d.compute(frame_.clone(), desc);
    return desc;
}


float getDistanceHistogram(const std::vector<float>& refHist, const std::vector<float>& currHist)
{
    double re = 0;
    for(int i = 0; i < std::min(refHist.size(), currHist.size()); ++i)
    {
        re += pow(refHist.at(i)-currHist.at(i),2);
    }
    return sqrt(re);


    //L2 distance
    double num = 0, deno=0;
    for(int i = 0; i < std::min(refHist.size(), currHist.size()); ++i)
    {
        num += pow(refHist.at(i)-currHist.at(i),2);
        deno += refHist.at(i) + currHist.at(i);
    }
    return num / deno;

    //Bhattacharyya distance
    cv::MatND hist(currHist);
    cv::MatND hist2(refHist);
    double ret = cv::compareHist(hist, hist2, CV_COMP_BHATTACHARYYA);
    return ret;

    //Bhattacharyya du cours
    float somme = 0;
    for(int i = 0; i < refHist.size(); ++i)
        somme += sqrt(currHist.at(i) * currHist.at(i));
    return -log(somme);


}

void correctRect(cv::Rect& rec){
    int correction = 8;
    if(rec.width % correction != 0){
        rec.width -= rec.width % correction;
    }
    if(rec.height % correction != 0){
        rec.height -= rec.height % correction;
    }
}


void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    srand (time(NULL));

    //clear particules between videos
    particules.clear();
    cv::Rect myoInitBBox(oInitBBox);
    correctRect(myoInitBBox);
    //init myBox and the associated histogram
    myBox = myoInitBBox;
    HOGhistogram = getHOGDescriptor(oInitFrame(myBox));

    //create particules from myBox
    for(int i = 0; i < NB_PARTICULES-1; ++i){
        addParticule(oInitFrame, myBox);
    }
    //also add the current myBox to the particules
    particules.push_back(Particule(myBox));
}


void MyTracker::addParticule(const cv::Mat& oCurrFrame, cv::Rect particule)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-RANDOM_RANGE, RANDOM_RANGE);

    //positions
    //version with +/- a few pixels
//    double x = particule.x+ rand()%11 - 5;
//    double y = particule.y+ rand()%11 - 5;

    //version cours
    double x = particule.x+ 0.5*round(2*particule.width*dis(gen));
    double y = particule.y+ 0.5*round(2*particule.height*dis(gen));

    //size
    //version with +/- a few pixels
//    double regionSizeW = std::max(1, std::min(oCurrFrame.size().width, particule.width + (rand()%11 - 5)));
//    double regionSizeH = std::max(1, std::min(oCurrFrame.size().height, particule.height + (rand()%11 - 5)));

//    version cours
    double regionSizeW = std::max(1, std::min(oCurrFrame.size().width, particule.width));
    double regionSizeH = std::max(1, std::min(oCurrFrame.size().height, particule.height));

    //limit inside the box
    x = std::max(0.0, std::min(x, oCurrFrame.cols-regionSizeW));
    y = std::max(0.0, std::min(y, oCurrFrame.rows-regionSizeH));
    cv::Rect res(x, y, regionSizeW, regionSizeH);
    correctRect(res);
    particules.push_back(res);
}


void MyTracker::apply(const cv::Mat &oCurrFrame, cv::Rect &oOutputBBox)
{
    //set with only NB_BEST_PARTICULES particules
    std::set<Particule, mycompare> best_particules;

    for(int i = 0; i < particules.size(); ++i)
    {
        //print particules (!!! affect gradient computation)
//        cv::rectangle(oCurrFrame, cv::Point(particules.at(i).getShape().x, particules.at(i).getShape().y),
//                      cv::Point(particules.at(i).getShape().x+particules.at(i).getShape().size().width,
//                                particules.at(i).getShape().y+particules.at(i).getShape().size().height), cv::Scalar(0,0,255));


        //same but with HOG descriptor
        double resHOG = getDistanceHistogram(HOGhistogram, getHOGDescriptor(oCurrFrame(particules.at(i).getShape())));
        particules.at(i).setHOGDistance(resHOG);

        //add only if better score than the worst in particules, or if size of particules < NB_BEST_PARTICULES
        if(best_particules.begin() == best_particules.end())
        {
            best_particules.insert(particules.at(i));
        }
        else
        {
            if(best_particules.size() < NB_BEST_PARTICULES || *best_particules.rbegin() > particules.at(i))
            {
                best_particules.insert(particules.at(i));
                //remove the worst particule
                if( best_particules.size() > NB_BEST_PARTICULES)
                    best_particules.erase(std::prev(best_particules.end()));
            }
        }
    }
    //compute position and size of the box (mean from best particules)
    double somme_distance = 0;

    float x_ = 0, y_ = 0, w_ = 0, h_ = 0;
    for(auto i : best_particules)
    {
        somme_distance += i.getHOGDistance(); //get the sum of all distance
        x_ += i.getShape().x;
        y_ += i.getShape().y;
        w_ += i.getShape().size().width;
        h_ += i.getShape().size().height;
    }
    double bests_size = best_particules.size();
    //mean
    myBox = cv::Rect(x_/bests_size, y_/bests_size, w_/bests_size, h_/bests_size);


    correctRect(myBox);
    oOutputBBox = myBox;

    // update histogram
    HOGhistogram=getHOGDescriptor(oCurrFrame(myBox));

    //clear old particules
    particules.clear();

    //generate new particules from best particules
    //compute a normalisaiton
    double norm = 0;
    for(auto i : best_particules)
        norm += somme_distance-i.getHOGDistance();

    int counter = 0;
    for(auto i : best_particules)
//        particules with short distance generate more new particules
    {
        for(int j = 0; j < NB_PARTICULES*(somme_distance-i.getHOGDistance())/norm; ++j)
            addParticule(oCurrFrame, i.getShape());
    }
}

