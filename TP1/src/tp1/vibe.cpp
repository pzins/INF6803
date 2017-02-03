
#include "tp1/common.hpp"
int computeLBP(const cv::Mat& area);
int distanceLBP(cv::Vec3b pix, cv::Vec3b neighbour);

// local implementation for ViBe segmentation algorithm
struct ViBe_impl : ViBe {
    ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma);
    virtual void initialize(const cv::Mat& oInitFrame);
    virtual void apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask);
    const size_t m_N; //< internal ViBe parameter; number of samples to keep in each pixel model
    const size_t m_R; //< internal ViBe parameter; maximum color distance allowed between RGB samples for matching
    const size_t m_nMin; //< internal ViBe parameter; required number of matches for background classification
    const size_t m_nSigma; //< internal ViBe parameter; model update rate

    bool checkDescriptor(const cv::Mat &currentArea, int coo);
    bool isSimilar(const cv::Vec3b& pix, const cv::Vec3b& samples);
    void ViBe_impl::applyMorpho(cv::Mat& oOutputMask);

    // @@@@ ADD ALL REQUIRED DATA MEMBERS FOR BACKGROUND MODEL HERE
    std::vector<std::vector<cv::Vec3b>> background; //background model
    std::vector<std::vector<int>> descriptors; //descriptors samples for each pixels
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
    descriptors.clear();
    for(int i = 1; i < oInitFrame.rows-1; i++)
    {
        for(int j = 1; j < oInitFrame.cols-1; ++j)
        {
            std::vector<cv::Vec3b> tmp; //contain samples for 1 pixel
            //neighbours pixels
            cv::Vec3b neighbours[] = {oInitFrame.at<cv::Vec3b>(i-1,j-1), oInitFrame.at<cv::Vec3b>(i-1,j), oInitFrame.at<cv::Vec3b>(i-1,j+1), oInitFrame.at<cv::Vec3b>(i,j-1),
                                  oInitFrame.at<cv::Vec3b>(i,j+1), oInitFrame.at<cv::Vec3b>(i+1,j-1), oInitFrame.at<cv::Vec3b>(i+1,j), oInitFrame.at<cv::Vec3b>(i+1,j+1)};
            //loop to put 20 random neighbours in the tmp vector
            for(int k = 0; k < m_N; ++k)
                tmp.push_back(neighbours[rand() % 8]);
            //add the sample vector to the background model
            background.push_back(tmp);
            cv::Vec3b curPixel = oInitFrame.at<cv::Vec3b>(i,j);
            cv::Mat roi;
            cv::Rect rect;
            rect = cv::Rect(j-1, i-1, 3, 3);
            roi= oInitFrame(rect);
            int LBPvalue= computeLBP(roi);
            std::vector<int> tmp2; //contain samples for 1 pixel
            for(int k = 0; k < m_N; ++k)
                tmp2.push_back(LBPvalue);
            descriptors.push_back(tmp2);
        }
    }
}

bool ViBe_impl::isSimilar(const cv::Vec3b& pix, const cv::Vec3b& samples){
//    L2 distance
    return (sqrt(pow(pix.val[0]-samples.val[0],2) + pow(pix.val[1]-samples.val[1],2) + pow(pix.val[2]-samples.val[2],2)) <= m_R);

//    L1 distance
//    return (abs(pix.val[0]-samples.val[0])+abs(pix.val[1]-samples.val[1])+abs(pix.val[2]-samples.val[2]) <= m_R);

//    HSV color model
//    cv::Mat input_pix(1,1,CV_8UC3);
//    input_pix.at<cv::Vec3b>(0,0) = pix;
//    cv::Mat input_samples(1,1, CV_8UC3);
//    input_samples.at<cv::Vec3b>(0,0) = samples;
//    cv::Mat res_pix(input_pix), res_samples(input_samples);
//    cv::cvtColor(input_pix, res_pix, CV_BGR2HLS);
//    cv::cvtColor(input_samples, res_samples, CV_BGR2HLS);
//    return (sqrt(pow(res_pix.at<cv::Vec3b>(0).val[0]-res_samples.at<cv::Vec3b>(0).val[0],2) +
//                 pow(res_pix.at<cv::Vec3b>(0).val[1]-res_samples.at<cv::Vec3b>(0).val[1],2) +
//                 pow(res_pix.at<cv::Vec3b>(0).val[2]-res_samples.at<cv::Vec3b>(0).val[2],2)) <= m_R);


//    image distorsion
//    int xt = pow(pix.val[0],2)+pow(pix.val[1],2)+pow(pix.val[2],2);
//    int vi = pow(samples.val[0],2)+pow(samples.val[1],2)+pow(samples.val[2],2);
//    int xtvi = pix.val[0]*samples.val[0]*pix.val[1]*samples.val[1]*pix.val[2]*samples.val[2];
//    float p2 =    xtvi / vi;
//    return sqrt(xt-p2) <= m_R;
}


bool ViBe_impl::checkDescriptor(const cv::Mat& currentArea, int coo){
    int res = computeLBP(currentArea);
    int counter = 0;
    int k = 0;
    while(counter < m_nMin && k < m_N){
        if(res == descriptors.at(coo).at(k++))
            counter++;
    }
    if (counter == m_nMin){
        int randomIndex = rand() % m_N;
        descriptors.at(coo).at(randomIndex) = res;
        return true;
    }
    return false;
}

void ViBe_impl::applyMorpho(cv::Mat& oOutputMask){
    //improvment with morphological operations
    int erosion_size = 2;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                              cv::Size( erosion_size + 1,  erosion_size + 1),
                          cv::Point(erosion_size, erosion_size));

    cv::erode(oOutputMask, oOutputMask, element);
    erosion_size = 4;
    element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                              cv::Size( erosion_size + 1,  erosion_size + 1),
                          cv::Point(erosion_size, erosion_size));
    cv::dilate(oOutputMask, oOutputMask, element);
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
            cv::Vec3b curPix = oCurrFrame.at<cv::Vec3b>(i,j);

            cv::Mat roi;
            cv::Rect rect;
            rect = cv::Rect(j-1, i-1, 3, 3);
            roi= oCurrFrame(rect);

            if(checkDescriptor(roi, coo))
            {
                nbOk = m_nMin; //we quit directly
            } else{
                counter = m_N;
            }
            //loop to check if a pixel is background or foreground
            while (nbOk < m_nMin && counter < m_N){
                if(isSimilar(background.at(coo).at(counter++), curPix)){
                    nbOk++;
                }
            }
            //pixel is background
            if(nbOk == m_nMin)
            {
                oOutputMask.at<uchar>(i,j) = 0;
                cv::Vec3b newValue = oCurrFrame.at<cv::Vec3b>(i,j);  //the current pixel value
                //update background model
                //add the new sample with m_nSigma probability
                if(!(rand() % m_nSigma))
                    background.at(coo).at(rand() % m_N) = newValue;

                //update neighbours
                if(i != 1 && i != oCurrFrame.rows -2 && j != 1 && j != oCurrFrame.cols -2)
                {
                    //only with a 1/16 probability
                    if(!(rand()%m_nSigma)){
                        int neighbours = rand() % 8; //get 1 random neighbours to be updated
                        if(neighbours == 0)
                            background.at((i-2)*(oCurrFrame.cols-2)+j-2).at(rand()%m_N) = newValue;
                        else if (neighbours == 1)
                            background.at((i-2)*(oCurrFrame.cols-2)+j-1).at(rand()%m_N) = newValue;
                        else if (neighbours == 2)
                            background.at((i-2)*(oCurrFrame.cols-2)+j).at(rand()%m_N) = newValue;
                        else if (neighbours == 3)
                            background.at((i-1)*(oCurrFrame.cols-2)+j).at(rand()%m_N) = newValue;
                        else if (neighbours == 4)
                            background.at((i-1)*(oCurrFrame.cols-2)+j-2).at(rand()%m_N) = newValue;
                        else if (neighbours == 5)
                            background.at((i)*(oCurrFrame.cols-2)+j-2).at(rand()%m_N) = newValue;
                        else if (neighbours == 6)
                            background.at((i)*(oCurrFrame.cols-2)+j-1).at(rand()%m_N) = newValue;
                        else if (neighbours == 7)
                            background.at((i)*(oCurrFrame.cols-2)+j).at(rand()%m_N) = newValue;
                    }
                }
            }
            //pixel is foreground
            else {
                oOutputMask.at<uchar>(i,j) = 255;
            }
        }
    }

    // hint: we work with RGB images, so the type of one pixel is a "cv::Vec3b"! (i.e. three uint8_t's are stored per pixel)

}



int distanceLBP(cv::Vec3b pix, cv::Vec3b neighbour){
    if(abs(pix.val[0]+pix.val[1]+pix.val[2] - neighbour.val[0]+neighbour.val[1]+neighbour.val[2]) <= 0.365*0.299*pix.val[2]+0.587*pix.val[1]*0.114*pix.val[0])
        return 1;
    else
        return 0;
}

int computeLBP(const cv::Mat& area){
    CV_Assert(area.cols == 3 && area.rows == 3);
    cv::Vec3b centerPixel = area.at<cv::Vec3b>(1,1);
    int counter = 0;
    int res = 0;
    for(int i = 0; i < area.rows; ++i){
        for(int j = 0; j < area.cols; ++j){
            if(!(i==1&&j==1))
                res += distanceLBP(centerPixel, area.at<cv::Vec3b>(i,j)) * pow(counter++,2);
        }
    }
    return res;
}

