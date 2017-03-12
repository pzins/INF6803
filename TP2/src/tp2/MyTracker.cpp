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
#define NB_BEST_PARTICULES 2 //number of best particules to compute new box coordinate
#define RANDOM_RANGE 0.5

#define CELL_SIZE 8

enum DISTANCE_VERSION {DISTANCE_1, L2, BHATTACHARYYA_COURS};

DISTANCE_VERSION DV = DISTANCE_1; //choose which distance between histograms to use




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


std::vector<float> getHistogram(const cv::Mat& frame_){
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;

    cv::Mat frame = frame_.clone();
    //convert to grayscale
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

    cv::Mat directions(frame.clone());
    for(int i = 0; i < frame.rows; ++i){
        for(int j = 0; j < frame.cols; ++j){
            //compute the angle
            float angle = atan2((float)grad_y.at<int16_t>(i,j), (float)grad_x.at<int16_t>(i,j)) * 180 / PI;
            //to get gradient direction between 0-360
            if(angle<0)
                angle = 180 + angle;
            directions.at<uint8_t>(i,j) = angle;
        }
    }


    int nb_cell_width = frame_.size().width / CELL_SIZE;
    int nb_cell_height = frame_.size().height / CELL_SIZE;

    std::vector<std::vector<std::vector<float>>> matHisto;
    std::vector<std::vector<float>> matMag;

    std::vector<int> bins = {10,30,50,70,90,110,130,150,170};
    for(int i = 0; i < nb_cell_height; ++i)
    {
        std::vector<std::vector<float>> tmp;
        std::vector<float> tmp2;

        for(int j = 0; j < nb_cell_width; ++j)
        {
            cv::Mat cellGrad(frame, cv::Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE));
            cv::Mat cellDir(directions, cv::Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE));
            std::vector<float> histoCell(9, 0);
            float sumMag = 0;
            for(int x = 0; x < 8; ++x)
            {
                for(int y = 0; y < 8; ++y)
                {
                    double value = cellDir.at<uint8_t>(x,y);
                    sumMag += cellGrad.at<uint8_t>(x,y);

                    int idx = 0;
                    for(int a = 0; a < bins.size(); ++a)
                    {
                        if(value <= bins.at(a)+10 && value >= bins.at(a)-10){
                            idx = a;
                            break;
                        }
                    }

                    if(value>=170 || value <= 10){
                        histoCell.at(idx) += cellGrad.at<uint8_t>(x,y);
                    }
                    else{
                        if(idx == 8){
                            histoCell.at(idx) += cellGrad.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx)))/20;
                            histoCell.at(idx-1) += cellGrad.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx-1)))/20;
                        } else if(idx == 0){
                            histoCell.at(idx) += cellGrad.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx)))/20;
                            histoCell.at(idx+1) += cellGrad.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx+1)))/20;
                        } else {
                            histoCell.at(idx) += cellGrad.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx)))/20;
                            histoCell.at(idx+1) += cellGrad.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx+1)))/20;
                        }

                    }
                }
            }
            tmp.push_back(histoCell);
            tmp2.push_back(sumMag);
        }
        matMag.push_back(tmp2);
        matHisto.push_back(tmp);
    }


    std::vector<float> finalRes;

    for(int i = 0; i < nb_cell_height-1; ++i)
    {
        for(int j = 0; j < nb_cell_width - 1; ++j)
        {
            std::vector<float> res = matHisto.at(i).at(j);
            res.insert(res.end(), matHisto.at(i).at(j+1).begin(), matHisto.at(i).at(j+1).end());
            res.insert(res.end(), matHisto.at(i+1).at(j).begin(), matHisto.at(i+1).at(j).end());
            res.insert(res.end(), matHisto.at(i+1).at(j+1).begin(), matHisto.at(i+1).at(j+1).end());
            double norm = matMag.at(i).at(j)+matMag.at(i+1).at(j)+matMag.at(i).at(j+1)+matMag.at(i+1).at(j+1);
            double norm2 = 0;
            for(int k = 0; k < res.size(); ++k)
                norm2 += pow(res.at(i),2) + 0.5*0.5;
            for(int k = 0; k < res.size(); ++k)
                res.at(k) /= sqrt(norm2);
            finalRes.insert(finalRes.end(), res.begin(), res.end());
        }
    }

    return finalRes;
}


float getDistanceHistogram(const std::vector<float>& refHist, const std::vector<float>& currHist)
{
    if(DV == DISTANCE_1)
    {
        //L2 distance
        double num = 0, deno=0;
        for(int i = 0; i < std::min(refHist.size(), currHist.size()); ++i)
        {
            num += pow(refHist.at(i)-currHist.at(i),2);
            deno += refHist.at(i) + currHist.at(i);
        }
        return num / deno;
    }
    else if(DV == L2)
    {
        double res = 0;
        for(int i = 0; i < std::min(refHist.size(), currHist.size()); ++i)
            res += pow(refHist.at(i)-currHist.at(i),2);
        return sqrt(res);

    }
    else if(DV == BHATTACHARYYA_COURS)
    {
        //Bhattacharyya du cours
        float res = 0;
        for(int i = 0; i < std::min(refHist.size(), currHist.size()); ++i)
            res += sqrt(currHist.at(i) * currHist.at(i));
        return -log(res);
    }
}


void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    srand (time(NULL));

    //clear particules between videos
    particules.clear();

    //init myBox and the associated histogram
    myBox = oInitBBox;
    HOGhistogram = getHistogram(oInitFrame(myBox));

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
    double x = particule.x+ round(2*particule.width*dis(gen));
    double y = particule.y+ round(2*particule.height*dis(gen));

    //size
    //version with +/- a few pixels
//    double regionSizeW = std::max(1, std::min(oCurrFrame.size().width, particule.width + (rand()%11 - 5)));
//    double regionSizeH = std::max(1, std::min(oCurrFrame.size().height, particule.height + (rand()%11 - 5)));

//    version cours
    double regionSizeW = std::max(1.0, std::min((double)oCurrFrame.size().width, particule.width + round((particule.width/10)*dis(gen))));
    double regionSizeH = std::max(1.0, std::min((double)oCurrFrame.size().height, particule.height + round((particule.height/10)*dis(gen))));

    //limit inside the box
    x = std::max(0.0, std::min(x, oCurrFrame.cols-regionSizeW));
    y = std::max(0.0, std::min(y, oCurrFrame.rows-regionSizeH));

    particules.push_back(cv::Rect(x, y, regionSizeW, regionSizeH));
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


        // compute distance between each particules histogram and the oOutputBBox histogram at the previous frame
        double res = getDistanceHistogram(HOGhistogram, getHistogram(oCurrFrame(particules.at(i).getShape())));
        particules.at(i).setHOGDistance(res);


        //add only if better score than the worst in particules, or if size of particules < NB_BEST_PARTICULES
        if(best_particules.begin() == best_particules.end())
            best_particules.insert(particules.at(i));
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
    oOutputBBox = myBox;

    // update histogram
    HOGhistogram=getHistogram(oCurrFrame(myBox));

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
        for(int j = 0; j < NB_PARTICULES*(somme_distance-i.getHOGDistance())/norm; ++j)
            addParticule(oCurrFrame, i.getShape());
}

