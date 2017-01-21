
#include "tp1/common.hpp"

// local implementation for ViBe segmentation algorithm
struct ViBe_impl : ViBe {
    ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma);
    virtual void initialize(const cv::Mat& oInitFrame);
    virtual void apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask);
    const size_t m_N; //< internal ViBe parameter; number of samples to keep in each pixel model
    const size_t m_R; //< internal ViBe parameter; maximum color distance allowed between RGB samples for matching
    const size_t m_nMin; //< internal ViBe parameter; required number of matches for background classification
    const size_t m_nSigma; //< internal ViBe parameter; model update rate

    // @@@@ ADD ALL REQUIRED DATA MEMBERS FOR BACKGROUND MODEL HERE
    std::vector<std::vector<cv::Vec3b>> background; //background model

};

std::shared_ptr<ViBe> ViBe::createInstance(size_t N, size_t R, size_t nMin, size_t nSigma) {
    return std::shared_ptr<ViBe>(new ViBe_impl(N,R,nMin,nSigma));
}

ViBe_impl::ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma) :
    m_N(N),
    m_R(R),
    m_nMin(nMin),
    m_nSigma(nSigma) {}


void ViBe_impl::initialize(const cv::Mat& oInitFrame) {
    CV_Assert(!oInitFrame.empty() && oInitFrame.isContinuous() && oInitFrame.type()==CV_8UC3);

    // hint: we work with RGB images, so the type of one pixel is a "cv::Vec3b"! (i.e. three uint8_t's are stored per pixel)
    //loop over the initial frame (except the outer border)
    background.clear();
    for(int i = 1; i < oInitFrame.rows-1; i++)
    {
        for(int j = 1; j < oInitFrame.cols-1; ++j)
        {
            std::vector<cv::Vec3b> tmp; //contain samples for 1 pixel
            //neighbours pixels
            cv::Vec3b neighbours[] = {oInitFrame.at<cv::Vec3b>(i-1,j-1), oInitFrame.at<cv::Vec3b>(i-1,j), oInitFrame.at<cv::Vec3b>(i-1,j+1), oInitFrame.at<cv::Vec3b>(i,j-1),
                                  oInitFrame.at<cv::Vec3b>(i,j+1), oInitFrame.at<cv::Vec3b>(i+1,j-1), oInitFrame.at<cv::Vec3b>(i+1,j), oInitFrame.at<cv::Vec3b>(i+1,j+1)};
            //loop to put 20 random neighbours in the tmp vector
            for(int k = 0; k < 20; ++k)
                tmp.push_back(neighbours[rand() % 8]);
            //add the sample vector to the background model
            background.push_back(tmp);
        }
    }
}

//function which checks wether two pixel are close
bool isInSphere(const cv::Vec3b& pix, const cv::Vec3b& samples, int seuil){
    return (sqrt(pow(pix.val[0]-samples.val[0],2) + pow(pix.val[1]-samples.val[1],2) + pow(pix.val[2]-samples.val[2],2)) <= seuil);
}

void ViBe_impl::apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask) {
    CV_Assert(!oCurrFrame.empty() && oCurrFrame.isContinuous() && oCurrFrame.type()==CV_8UC3);
    oOutputMask.create(oCurrFrame.size(),CV_8UC1); // output is binary, but always stored in a byte (so output values are either '0' or '255')

    int coo = 0;
    //loop over the current frame
    for(int i = 1; i < oCurrFrame.rows-1; i++)
    {
        for(int j = 1; j < oCurrFrame.cols-1; ++j)
        {
            int nbOk = 0;   //number of samples pixels which are close to the current pixel
            int counter = 0;
            coo = (i-1)*(oCurrFrame.cols-2)+j-1;    //compute coordinate (i,j) for the vector background model (1 dimension)
            //loop to check if a pixel is background or foreground
            while (nbOk < 2 && counter < 20){
                if(isInSphere(background.at(coo).at(counter++), oCurrFrame.at<cv::Vec3b>(i,j), 20)){
                    nbOk++;
                }
            }
            //pixel is background
            if(nbOk == 2)
            {
                oOutputMask.at<uchar>(i,j) = 0;
                cv::Vec3b newValue = oCurrFrame.at<cv::Vec3b>(i,j);  //the current pixel value
                //update background model
                //add the new sample with 1/16 probability
                if(!(rand() % 16))
                    background.at(coo).at(rand() % 20) = newValue;

                //update neighbours
                if(i != 1 && i != oCurrFrame.rows -2 && j != 1 && j != oCurrFrame.cols -2)
                {
                    //only with a 1/16 probability
                    if(!(rand()%16)){
                        int neighbours = rand() % 8; //get 1 random neighbours to be updated
                        if(neighbours == 0)
                            background.at((i-2)*(oCurrFrame.cols-2)+j-2).at(rand()%20) = newValue;
                        else if (neighbours == 1)
                            background.at((i-2)*(oCurrFrame.cols-2)+j-1).at(rand()%20) = newValue;
                        else if (neighbours == 2)
                            background.at((i-2)*(oCurrFrame.cols-2)+j).at(rand()%20) = newValue;
                        else if (neighbours == 3)
                            background.at((i-1)*(oCurrFrame.cols-2)+j).at(rand()%20) = newValue;
                        else if (neighbours == 4)
                            background.at((i-1)*(oCurrFrame.cols-2)+j-2).at(rand()%20) = newValue;
                        else if (neighbours == 5)
                            background.at((i)*(oCurrFrame.cols-2)+j-2).at(rand()%20) = newValue;
                        else if (neighbours == 6)
                            background.at((i)*(oCurrFrame.cols-2)+j-1).at(rand()%20) = newValue;
                        else if (neighbours == 7)
                            background.at((i)*(oCurrFrame.cols-2)+j).at(rand()%20) = newValue;
                    }
                }
            }
            //pixel is foreground
            else {
                oOutputMask.at<uchar>(i,j) = 255;
            }
        }
    }

    //improvment with morphological operations
//    int erosion_size = 2;
//    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
//                          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
//                          cv::Point(erosion_size, erosion_size) );

//    cv::erode(oOutputMask, oOutputMask, element);
//    cv::dilate(oOutputMask, oOutputMask, element);

    // hint: we work with RGB images, so the type of one pixel is a "cv::Vec3b"! (i.e. three uint8_t's are stored per pixel)

}
