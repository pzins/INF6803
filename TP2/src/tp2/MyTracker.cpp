#include <tp2/common.hpp>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <set>

#define PI 3.14159265
#define NB_PARTICULES 100

#define FACTOR_CHANGE_PARTICULE_POSITION 1
#define FACTOR_CHANGE_PARTICULE_SIZE 10
#define NB_BEST_PARTICULES_BOX 3 //number of best particules to compute new box coordinate
#define NB_BEST_PARTICULES_PART 3 //number of best particules to compute new box coordinate

//version baseline
#define ANGLE_SIGNED true
#define ANGLE_DIVISION 9 //should divide 360 if angle are signed or 180 if angle are unsigned

//version myHOG
#define CELL_SIZE 8

#define USE_REFERENCE_BOX false


enum DISTANCE_VERSION {CHI2, L2, BHATTACHARYYA}; //BHATTACHARYYA doesn't work if histogram have difference size (HOG_OPENCV, _MY_HOG)
DISTANCE_VERSION DV = CHI2; //choose which distance between histograms to use

enum VERSION {BASELINE, HOG_OPENCV, MY_HOG};
VERSION V = HOG_OPENCV;


class Particule
{
private:
    double distance; //distance with suivi baseline
    cv::Rect shape;

public:
    Particule(cv::Rect rec) : shape(rec), distance(0) {}
    const cv::Rect& getShape() const {return shape;}
    void setShape(const cv::Rect& rec){shape = rec;}

    void setDistance(double value){distance = value;}
    double getDistance() const {return distance;}

    bool operator< (const Particule& other) const {
        return distance < other.distance;
    }
    bool operator> (const Particule& other) const {
        return distance > other.distance;
    }
};

//comparaison of Particule for set insert
struct mycompare {
    bool operator() (const Particule& p1, const Particule& p2) const{
        return p1.getDistance() < p2.getDistance();
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
    std::vector<float> histogram; //histogram suivi baseline
    std::vector<float> refHistogram;

    cv::Rect myBox;
    std::vector<Particule> particules;
    cv::Size initSizeBox;
public:
    void initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox);
    void apply(const cv::Mat& oCurrFrame, cv::Rect& oOutputBBox);
    //add a new particule based on the particule parameter
    void addParticule(const cv::Mat& ocurrFrame, cv::Rect particule);
    void printParticules();
};

void MyTracker::printParticules(){
    std::cout << "----------------------------------" << std::endl;
    for(auto i : particules)
        std::cout << i << std::endl;
    std::cout << "----------------------------------" << std::endl;
}


std::shared_ptr<Tracker> Tracker::createInstance(){
    return std::shared_ptr<Tracker>(new MyTracker());
}


std::vector<float> getBaselineHistogram(const cv::Mat& frame_){
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


    int histo_size;
    //set histo size (180 if angles are unsigned, 360 if angles are unsigned)
    if(ANGLE_SIGNED == true) histo_size = 360;
    else histo_size = 180;
    std::vector<float> histo(histo_size, 0);

    //sum of gradient of the frame
    double sum_gradient = cv::sum(frame)[0];

    for(int i = 0; i < frame.rows; ++i){
        for(int j = 0; j < frame.cols; ++j){
            //compute the angle (-180 180)
            float angle = atan2((float)grad_y.at<int16_t>(i,j), (float)grad_x.at<int16_t>(i,j)) * 180 / PI;
            if(ANGLE_SIGNED==true)
            {
                if(angle<0) angle = 180 + (180 + angle); //to get angle between 0-360
                if(angle==360) angle--;
            }
            else {
                if(angle<0) angle = 180 + angle; //to get angle between 0-180
                if(angle == 180) angle--;
            }
            histo.at(static_cast<int>(angle)) += frame.at<uint8_t>(i,j) / sum_gradient;
        }
    }

    //merge angle directions in ANGLE_DIVISION groups
    std::vector<float> res;
    for(int i = 0; i < ANGLE_DIVISION; ++i)
    {
        double tmp = 0;
        for(int j = 0; j < histo_size/ANGLE_DIVISION; ++j){
            tmp += histo.at(i * histo_size/ANGLE_DIVISION + j);
        }
        res.push_back(tmp);
    }
    return res;
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


std::vector<float> getHOGDescriptor(const cv::Mat& frame_){
    cv::HOGDescriptor d(frame_.size(), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8),9);
    std::vector<float> desc;
    d.compute(frame_.clone(), desc);
    return desc;
}

std::vector<float> getMyHOGDescriptor(const cv::Mat& frame_){
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
            //to get gradient direction between 0-180
            if(angle<0)
                angle = 180 + angle;
            directions.at<uint8_t>(i,j) = angle;
        }
    }

    //max number of cell (in width and in height) in the box
    int nb_cell_width = frame_.size().width / CELL_SIZE;
    int nb_cell_height = frame_.size().height / CELL_SIZE;

    //matrix with cell histogramm
    std::vector<std::vector<std::vector<float>>> matHisto;

    //bins angles
    std::vector<int> bins = {10,30,50,70,90,110,130,150,170};
    //iterate over each cell
    for(int i = 0; i < nb_cell_height; ++i)
    {
        //vector of cell histogram for one row
        std::vector<std::vector<float>> tmp;

        for(int j = 0; j < nb_cell_width; ++j)
        {
            //get cell gradient direction and magnitude
            cv::Mat cellMag(frame, cv::Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE));
            cv::Mat cellDir(directions, cv::Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE));

            //create the cell histogram
            std::vector<float> histoCell(9, 0);

            //iterate over pixels in a cell
            for(int x = 0; x < 8; ++x)
            {
                for(int y = 0; y < 8; ++y)
                {
                    double value = cellDir.at<uint8_t>(x,y);

                    //find the correct bin (go through bins and stop when value is smaller)
                    int idx = -1;
                    for(int a = 0; a < bins.size(); ++a)
                    {
                        if(value <= bins.at(a)){
                            idx = a;
                            break;
                        }
                    }
                    // update histogram : two bins are increased if the angle is between
                    //special case : only on bin
                    if(idx == 0){
                        histoCell.at(idx) += cellMag.at<uint8_t>(x,y);
                    } else if(idx == -1)
                    {
                        histoCell.at(histoCell.size()-1) += cellMag.at<uint8_t>(x,y);
                    }
                    else{
                        histoCell.at(idx) += cellMag.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx)))/20;
                        histoCell.at(idx-1) += cellMag.at<uint8_t>(x,y)*(20-abs(value-bins.at(idx-1)))/20;
                    }
                }
            }
            tmp.push_back(histoCell);
        }
        matHisto.push_back(tmp);
    }


    std::vector<float> finalRes;

    //iterate over blocks (1 block = 2x2 cells)
    for(int i = 0; i < nb_cell_height-1; ++i)
    {
        for(int j = 0; j < nb_cell_width - 1; ++j)
        {
            //concatenate 4 histograms
            std::vector<float> res = matHisto.at(i).at(j);
            res.insert(res.end(), matHisto.at(i).at(j+1).begin(), matHisto.at(i).at(j+1).end());
            res.insert(res.end(), matHisto.at(i+1).at(j).begin(), matHisto.at(i+1).at(j).end());
            res.insert(res.end(), matHisto.at(i+1).at(j+1).begin(), matHisto.at(i+1).at(j+1).end());

            double norm = 0;
            //compute normalisation
            for(int k = 0; k < res.size(); ++k)
                norm += pow(res.at(i),2) + 0.5*0.5;

            for(int k = 0; k < res.size(); ++k)
                res.at(k) /= sqrt(norm);
            finalRes.insert(finalRes.end(), res.begin(), res.end());
        }
    }
    return finalRes;
}

std::vector<float> getHistogram(const cv::Mat& frame_){
    switch (V) {
    case BASELINE:
        return getBaselineHistogram(frame_);
    case HOG_OPENCV:
        return getHOGDescriptor(frame_);
    case MY_HOG:
        return getMyHOGDescriptor(frame_);
    }
}


float getDistanceHistogram(const std::vector<float>& refHist, const std::vector<float>& currHist)
{
    if(DV == CHI2)
    {
        //L2 distance
        double res = 0;
        for(int i = 0; i < std::min(refHist.size(), currHist.size()); ++i)
        {
            if(refHist.at(i) + currHist.at(i) == 0) continue; //if both histogram have 0, continue to the next index
            res += pow(refHist.at(i)-currHist.at(i),2) / (refHist.at(i) + currHist.at(i));
        }
        return res;
    }
    else if(DV == L2)
    {
        double res = 0;
        for(int i = 0; i < std::min(refHist.size(), currHist.size()); ++i)
            res += pow(refHist.at(i)-currHist.at(i),2);
        return sqrt(res);

    }
    else if(DV == BHATTACHARYYA)
    {
        //Bhattacharyya distance
        cv::MatND hist(currHist);
        cv::MatND hist2(refHist);
        double res = cv::compareHist(hist, hist2, CV_COMP_BHATTACHARYYA);
        return res;
    }
}

void MyTracker::addParticule(const cv::Mat& oCurrFrame, cv::Rect particule)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    //positions
    double x = particule.x+ FACTOR_CHANGE_PARTICULE_POSITION * round(particule.width*dis(gen));
    double y = particule.y+ FACTOR_CHANGE_PARTICULE_POSITION * round(particule.height*dis(gen));

    //size
    double regionSizeW = std::max(1, std::min(oCurrFrame.size().width, particule.width));
    double regionSizeH = std::max(1, std::min(oCurrFrame.size().height, particule.height));
    //if baseline we can change particule size
    if(V == BASELINE)
    {
        regionSizeW = std::max(1.0, std::min((double)oCurrFrame.size().width, particule.width + round((particule.width/FACTOR_CHANGE_PARTICULE_SIZE)*dis(gen))));
        regionSizeH = std::max(1.0, std::min((double)oCurrFrame.size().height, particule.height + round((particule.height/FACTOR_CHANGE_PARTICULE_SIZE)*dis(gen))));
    }
    if(regionSizeW < initSizeBox.width && regionSizeH < initSizeBox.height)
    {
        regionSizeH = particule.size().height;
        regionSizeW = particule.size().width;
    }
    //limit inside the box
    x = std::max(0.0, std::min(x, oCurrFrame.cols-regionSizeW));
    y = std::max(0.0, std::min(y, oCurrFrame.rows-regionSizeH));
    cv::Rect res(x, y, regionSizeW, regionSizeH);
    if(V == HOG_OPENCV) correctRect(res);
    particules.push_back(res);
}


void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    srand (time(NULL));

    //save initial size to prevent box to become too small
    initSizeBox = oInitBBox.size();

    //clear particules between videos
    particules.clear();

    //init myBox and the associated histogram
    myBox = oInitBBox;
    if(V==HOG_OPENCV) correctRect(myBox);
    histogram = getHistogram(oInitFrame(myBox));
    refHistogram = getHistogram(oInitFrame(myBox));

    //create particules from myBox
    for(int i = 0; i < NB_PARTICULES-1; ++i){
        addParticule(oInitFrame, myBox);
    }
    //also add the current myBox to the particules
    particules.push_back(Particule(myBox));

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
        std::vector<float> histo_parti = getHistogram(oCurrFrame(particules.at(i).getShape()));
        double res = getDistanceHistogram(histogram, histo_parti);
        if(USE_REFERENCE_BOX==true)
            res += getDistanceHistogram(refHistogram, histo_parti);

        particules.at(i).setDistance(res);

        //add only if better score than the worst in particules, or if size of particules < NB_BEST_PARTICULES
        if(best_particules.begin() == best_particules.end())
            best_particules.insert(particules.at(i));
        else
        {
            if(best_particules.size() < NB_BEST_PARTICULES_PART || *best_particules.rbegin() > particules.at(i))
            {
                best_particules.insert(particules.at(i));
                //remove the worst particule
                if( best_particules.size() > NB_BEST_PARTICULES_PART)
                    best_particules.erase(std::prev(best_particules.end()));
            }
        }
    }

    //compute position and size of the box (mean from NB_BEST_PARTICULES_BOX of the best particules)
    int counter = 0;
    double somme_distance = 0;
    float x_ = 0, y_ = 0, w_ = 0, h_ = 0;
    for(auto i : best_particules)
    {
        if(counter++ < NB_BEST_PARTICULES_BOX) //compute the new position of the box from only NB_BEST_PARTICULES_BOX particules (not the entire best_particules)
        {
            x_ += i.getShape().x;
            y_ += i.getShape().y;
            w_ += i.getShape().size().width;
            h_ += i.getShape().size().height;
        }
        somme_distance += i.getDistance(); //compute sum of the distance for all best particules
    }
    //mean
    myBox = cv::Rect(x_/NB_BEST_PARTICULES_BOX, y_/NB_BEST_PARTICULES_BOX, w_/NB_BEST_PARTICULES_BOX, h_/NB_BEST_PARTICULES_BOX);
    if(V == HOG_OPENCV) correctRect(myBox);
    oOutputBBox = myBox;

    // update histogram
    histogram=getHistogram(oCurrFrame(myBox));

    //clear old particules
    particules.clear();

    //generate new particules from best particules
    //compute a normalisaiton
    double norm = 0;
    for(auto i : best_particules)
        norm += somme_distance-i.getDistance();

    counter = 0;
    for(auto i : best_particules)
//        particules with short distance generate more new particules
        for(int j = 0; j < NB_PARTICULES*(somme_distance-i.getDistance())/norm; ++j)
            addParticule(oCurrFrame, i.getShape());

}
