#include <tp2/common.hpp>



class MyTracker : public Tracker
{
private:
    std::vector<int> histogram;
    cv::Rect myBox;
    std::vector<cv::Rect> boxes;
public:
    void initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox);
    void apply(const cv::Mat& oCurrFrame, cv::Rect& oOutputBBox);
};

std::shared_ptr<Tracker> Tracker::createInstance(){
    return std::shared_ptr<Tracker>(new MyTracker());
}

std::vector<int> getHistogram(const cv::Mat& frame){
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;

//    cv::GaussianBlur( frame, frame, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    /// Convert it to gray
//    cvtColor( frame, frame, CV_BGR2GRAY );

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
    std::vector<int> histo(360,0);
    for(int i = 0; i < frame.rows; ++i){
        for(int j = 0; j < frame.cols; ++j){
            float angle = cv::fastAtan2(grad_x.at<uint8_t>(i,j), grad_y.at<uchar>(i,j));
            histo.at(static_cast<int>(angle)) += (int)frame.at<uint8_t>(i,j);
        }
    }
    return histo;
}

void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    myBox = oInitBBox; //mybox my rect tracker
}


int getDistanceHistogram(const std::vector<int>& refHist, const std::vector<int>& currHist){
    float somme = 0;
    for(int i = 0; i < refHist.size(); ++i)
    {
        somme += sqrt(refHist.at(i) * currHist.at(i));
    }
    return -log(somme);
}


void MyTracker::apply(const cv::Mat &oCurrFrame, cv::Rect &oOutputBBox)
{
    int nbParticules = 10;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5,0.5);

    for(int i = 0; i < 10; ++i){
        float x = myBox.x+round(2*myBox.width*dis(gen));
        float y = myBox.y+round(2*myBox.height*dis(gen));
        int regionSizeW =  myBox.width + round((myBox.width/3)*dis(gen));
        int regionSizeH =  myBox.height+ round((myBox.height/3)*dis(gen));
        boxes.push_back(cv::Rect(std::max(int(x),0), std::max(int(y),0), std::min(regionSizeW, oCurrFrame.rows), std::min(regionSizeH, oCurrFrame.cols)));
    }
    //get frame from ref rect + compute histo
    cv::Mat myBoxFrame = oCurrFrame(myBox);
    std::vector<int> ref_histo = getHistogram(myBoxFrame);
    int mini_diff = 10000, mini_idx = -1;
    for(int i = 0; i < boxes.size(); ++i)
    {
        cv::rectangle(oCurrFrame, cv::Point(boxes.at(i).x, boxes.at(i).y), cv::Point(boxes.at(i).x+boxes.at(i).size().width, boxes.at(i).y+boxes.at(i).size().height), cv::Scalar(255,0,0));
//        std::cout << boxes.at(i).x << " "  << boxes.at(i).y << " " << boxes.at(i).height << " "  << boxes.at(i).y+boxes.at(i).width << std::endl;
        std::vector<int> curr_histo = getHistogram(oCurrFrame(boxes.at(i)));
        if(abs(getDistanceHistogram(ref_histo, curr_histo)) <= mini_diff)
        {
            mini_idx = i;
        }
        std::cout << "ol" << std::endl;
    }
    myBox = boxes.at(mini_idx);
    boxes.clear();

}

//voir function normalize()
