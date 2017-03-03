#include <tp2/common.hpp>
#include <unistd.h>
#define PI 3.14159265


class MyTracker : public Tracker
{
private:
    std::vector<float> histogram;
    cv::Rect myBox;
    std::vector<cv::Rect> boxes;
public:
    void initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox);
    void apply(const cv::Mat& oCurrFrame, cv::Rect& oOutputBBox);
};

std::shared_ptr<Tracker> Tracker::createInstance(){
    return std::shared_ptr<Tracker>(new MyTracker());
}

std::vector<float> getHistogram(const cv::Mat& frame_){

    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;

    cv::Mat frame;

    cv::GaussianBlur(frame_, frame_, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
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
//    std::cout << grad_y << std::endl;

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, frame);

    std::vector<float> histo(360,0);
    double sum_gradient = cv::sum(frame)[0];

    for(int i = 0; i < frame.rows; ++i){
        for(int j = 0; j < frame.cols; ++j){
            float angle = atan2((float)grad_y.at<int>(i,j), (float)grad_x.at<int>(i,j)) * 180 / PI;
            if(angle<0)
            {
                angle = 180 + (180 + angle)-1;
            }
            histo.at(static_cast<int>(angle)) += frame.at<uint8_t>(i,j) / sum_gradient;
        }
    }
//    cv::waitKey();
//    usleep(100000);
    //quantif gradient
    std::vector<float> res;
    for(int i = 0; i < 15; ++i)
    {
        double tmp = 0;
        for(int j = 0; j < 24; ++j){
            tmp += histo.at(i * 24 + j);
        }
        res.push_back(tmp);
    }
//    for(auto i : res)
//        std::cout << i << " ";
//   std::cout<<std::endl;
    return res;
}

void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    myBox = oInitBBox; //mybox my rect tracker
    cv::Mat myBoxFrame = oInitFrame(myBox).clone();
    histogram = getHistogram(myBoxFrame);
}


float getDistanceHistogram(const std::vector<float>& refHist, const std::vector<float>& currHist){
    cv::MatND hist(currHist);
    cv::MatND hist2(refHist);
    double dist_b = cv::compareHist(hist, hist2, CV_COMP_BHATTACHARYYA);
//    std::cout << "BATTTA = " << dist_b << std::endl;
    return dist_b;
    double res = 0;
    double denominator = 0;
    double somme = 0;

//    for(int i = 0; i < refHist.size(); ++i)
//        somme += fabs(refHist.at(i)-currHist.at(i));
//    return somme;

    for(int i = 0; i < refHist.size(); ++i)
    {
        somme += refHist.at(i)*currHist.at(i);
    }
    std::cout << -log(somme) << std::endl;
    return -log(somme);

    for(int i = 0; i < refHist.size(); ++i)
        denominator += refHist.at(i) * currHist.at(i);
    for(int i = 0; i < refHist.size(); ++i)
        res += sqrt(refHist.at(i)*currHist.at(i)) / sqrt(denominator);

    return sqrt(1-res);
}


void MyTracker::apply(const cv::Mat &oCurrFrame, cv::Rect &oOutputBBox)
{
    int nbParticules = 100;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5,0.5);
    for(int i = 0; i < nbParticules-1; ++i){
        double x = myBox.x+round(2*myBox.width*dis(gen));
        double y = myBox.y+round(2*myBox.height*dis(gen));

        double regionSizeW = myBox.width + round((myBox.width/3)*dis(gen));
        double regionSizeH = myBox.height + round((myBox.height/3)*dis(gen));
        x = std::max(0.0, std::min(x, oCurrFrame.cols-regionSizeW));
        y = std::max(0.0, std::min(y, oCurrFrame.rows-regionSizeH));

        boxes.push_back(cv::Rect(x,y,regionSizeW, regionSizeH));
    }
    boxes.push_back(myBox);

    float mini_diff = 1000, mini_idx = 0;
    std::vector<float> tmp;
    for(int i = 0; i < boxes.size(); ++i)
    {
//        cv::rectangle(oCurrFrame, cv::Point(boxes.at(i).x, boxes.at(i).y), cv::Point(boxes.at(i).x+boxes.at(i).size().width, boxes.at(i).y+boxes.at(i).size().height), cv::Scalar(0,0,255));
        std::vector<float> curr_histo = getHistogram(oCurrFrame(boxes.at(i)).clone());

        float res = getDistanceHistogram(histogram, curr_histo);
//        std::cout << "res = " << res << std::endl;
        if( res < mini_diff)
        {
            mini_idx = i;
            mini_diff = res;
            tmp=curr_histo;
        }

    }
//    std::cout << "mini = " << mini_idx << std::endl;
    myBox = boxes.at(mini_idx);
    boxes.clear();
    oOutputBBox = myBox;

    //get frame from ref rect + compute histo
//    cv::Mat myBoxFrame = oCurrFrame(myBox).clone();
//    std::vector<int> ref_histo = getHistogram(myBoxFrame);
    histogram=tmp;

//    cv::waitKey(0);
//    usleep(100);
}

//voir function normalize()
