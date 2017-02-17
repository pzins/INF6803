#include <tp2/common.hpp>



class MyTracker : public Tracker
{
private:
    cv::Mat grad;
    cv::Mat angles;
public:
    void initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox);
    void apply(const cv::Mat& oCurrFrame, cv::Rect& oOutputBBox);

};

std::shared_ptr<Tracker> Tracker::createInstance(){
    return std::shared_ptr<Tracker>(new MyTracker());
}

void MyTracker::initialize(const cv::Mat& oInitFrame, const cv::Rect& oInitBBox)
{
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;

    cv::Mat tmp;
    cv::GaussianBlur( oInitFrame, tmp, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    /// Convert it to gray
    cvtColor( tmp, tmp, CV_BGR2GRAY );

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    cv::Sobel( tmp, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    cv::Sobel( tmp, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    for(int i = 0; i < grad.rows; ++i){
        for(int j = 0; j < grad.cols; ++j){
            angles.at<uchar>(i,j) = cv::fastAtan2(grad_x.at<uchar>(i,j), grad_y.at<uchar>(i,j));
        }
    }
}


void MyTracker::apply(const cv::Mat &oCurrFrame, cv::Rect &oOutputBBox)
{

}
