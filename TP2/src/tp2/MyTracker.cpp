#include <tp2/common.hpp>
#include <unistd.h>


class MyTracker : public Tracker
{
private:
    std::vector<int> histogram;
    cv::Rect myBox;
    std::vector<cv::Rect> boxes;
    std::vector<int> pred_histo;
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

    cv::GaussianBlur( frame, frame, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

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

//    std::cout << grad_x.size() << " " << grad_y.size() << std::endl;
//    cv::phase(grad_x, grad_y, grad_x, true);
//    std::cout << "ANGLES" << std::endl;
//    std::cout << angles << std::endl;

//    cv::Mat angles(grad_x), mag(grad_x);
//    cv::cartToPolar(grad_x, grad_y, angles, mag, true);

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
    //quantif gradient
    std::vector<int> res;
    for(int i = 0; i < 60; ++i)
    {
        double tmp = 0;
        for(int j = 0; j < 6; ++j){
            tmp += histo.at(i * 6 + j);
        }
        res.push_back(tmp);
    }
    return res;
}

void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    myBox = oInitBBox; //mybox my rect tracker
    cv::Mat myBoxFrame = oInitFrame(myBox).clone();
    pred_histo = getHistogram(myBoxFrame);
    std::cout << "MYBOX : " << myBox.x << " " << myBox.y << " " << myBox.width << " " << myBox.y << std::endl;
}


int getDistanceHistogram(const std::vector<int>& refHist, const std::vector<int>& currHist){

    double somme = 0;
    double somme2 = 0;
    for(int i = 0; i < refHist.size(); ++i)
    {
        somme += sqrt(refHist.at(i) * currHist.at(i));
        somme2 += abs(refHist.at(i) - currHist.at(i));
    }
    return somme2;
//    std::cout << "SOMME  =" << somme << std::endl;
    //max to avoid log(0)
    return -log(std::max(1.0,somme));
}


void MyTracker::apply(const cv::Mat &oCurrFrame, cv::Rect &oOutputBBox)
{
    int nbParticules = 15;
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
//    std::cout << "PARTICULES : " << std::endl;
//    for(auto i : boxes)
//        std::cout << i.x << " " << i.y << " " << i.width << " " << i.height << std::endl;



//    std::cout << "Histo" << std::endl;
//    for(auto i : ref_histo)
//        std::cout << i << " ";
//    std::cout << std::endl;
//    cv::waitKey();

    int mini_diff = 1000000000, mini_idx = 0;

    for(int i = 0; i < boxes.size(); ++i)
    {
//        cv::rectangle(oCurrFrame, cv::Point(boxes.at(i).x, boxes.at(i).y), cv::Point(boxes.at(i).x+boxes.at(i).size().width, boxes.at(i).y+boxes.at(i).size().height), cv::Scalar(0,0,255));
        std::vector<int> curr_histo = getHistogram(oCurrFrame(boxes.at(i)).clone());

//        std::cout << std::endl << "other histo" << std::endl;
//        for(auto i : curr_histo)
//            std::cout << i << " ";
//        std::cout << std::endl;
        int res = getDistanceHistogram(pred_histo, curr_histo);
//        std::cout << "RESULT :  " << i << " => " << res << std::endl;
        if( res < mini_diff)
        {
            mini_idx = i;
            mini_diff = res;
        }

    }
    std::cout << "MINI = " << mini_idx << std::endl;
    myBox = boxes.at(mini_idx);
    boxes.clear();
    oOutputBBox = myBox;

    //get frame from ref rect + compute histo
    cv::Mat myBoxFrame = oCurrFrame(myBox).clone();
    std::vector<int> ref_histo = getHistogram(myBoxFrame);
//    usleep(100000);
}

//voir function normalize()
