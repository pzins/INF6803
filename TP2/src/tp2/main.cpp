#include "tp2/common.hpp"
#include <unistd.h>

#define NB_FRAME_BEFORE_DECROCHAGE 10 //number of frames with OR = 0  accepted before decrochage

float computeCLE(const cv::Rect& ref, cv::Rect& myRect)
{
    float centerX1 = ref.x + ref.width / 2;
    float centerY1 = ref.y + ref.height / 2;
    float centerX2 = myRect.x + myRect.width / 2;
    float centerY2 = myRect.y + myRect.height / 2;
    return sqrt(pow(centerX1 - centerX2, 2)+pow(centerY1 - centerY2, 2));
}

double computeOR(const cv::Rect& ref, cv::Rect& myRect)
{
    //check if there is no overlap
    int x_min = std::min(ref.x, myRect.x);
    if(x_min == ref.x)
    {
        if(myRect.x - ref.x > ref.width) return 0;
    }
    else {
        if(ref.x - myRect.x > myRect.width) return 0;
    }

    int y_min = std::min(ref.y, myRect.y);
    if(y_min == ref.y){
        if(myRect.y-ref.y > ref.height) return 0;
    }
    else
    {
        if(ref.y - myRect.y > myRect.height) return 0;
    }

    cv::Point hg(std::max(ref.x, myRect.x), std::max(ref.y, myRect.y));
    cv::Point bd(std::min(ref.x+ref.width, myRect.x+myRect.width), std::min(ref.y+ref.height, myRect.y+myRect.height));
    double res = abs(bd.x-hg.x)*abs(bd.y-hg.y);
    return res/(ref.width*ref.height+myRect.width*myRect.height-res);
}


int main(int /*argc*/, char** /*argv*/) {
    try {

         std::shared_ptr<Tracker> pAlgo = Tracker::createInstance();

        const std::string sBaseDataPath(DATA_ROOT_PATH "/tp2/");
        const std::vector<std::string> vsSequenceNames = {"dog","face","woman"};
        const std::vector<size_t> vnSequenceSizes = {988,892,597};
        for(size_t nSeqIdx=1; nSeqIdx<vsSequenceNames.size(); ++nSeqIdx) {
            std::cout << "\nProcessing sequence '" << vsSequenceNames[nSeqIdx] << "'..." << std::endl;
            const std::string sInitFramePath = sBaseDataPath+vsSequenceNames[nSeqIdx]+"/img/0001.jpg";
            const cv::Mat oInitFrame = cv::imread(sInitFramePath);
            CV_Assert(!oInitFrame.empty() && oInitFrame.type()==CV_8UC3);
            std::ifstream oGTFile(sBaseDataPath+vsSequenceNames[nSeqIdx]+"/groundtruth_rect.txt");
            CV_Assert(oGTFile.is_open());
            std::string sCurrGTLine;
            std::getline(oGTFile,sCurrGTLine);
            CV_Assert(!sCurrGTLine.empty());
            std::cout << "Parsing input bounding box..." << std::endl;
            const cv::Rect oInitBBox = convertToRect(sCurrGTLine);
            std::cout << "Parsing input bounding box... done --- " << oInitBBox << std::endl;

            int nbFrames = 0, nbFrameOK = 0;
            double meanCLE = 0, meanOR = 0;
            int decrochage = 0;
            int nb_frame_decroche = 0;


            pAlgo->initialize(oInitFrame, oInitBBox);

            for(size_t nFrameIdx=2; nFrameIdx<=vnSequenceSizes[nSeqIdx]; ++nFrameIdx) {
//                std::cout << "\tProcessing input # " << nFrameIdx << " / " << vnSequenceSizes[nSeqIdx] << "..." << std::endl;
                const std::string sCurrFramePath = putf((sBaseDataPath+vsSequenceNames[nSeqIdx]+"/img/%04d.jpg").c_str(),(int)(nFrameIdx));
                const cv::Mat oCurrFrame = cv::imread(sCurrFramePath);
                CV_Assert(!oCurrFrame.empty() && oInitFrame.size()==oCurrFrame.size() && oCurrFrame.type()==CV_8UC3);
                cv::Rect oOutputBBox;

                pAlgo->apply(oCurrFrame,oOutputBBox);

                std::getline(oGTFile,sCurrGTLine);
                CV_Assert(!sCurrGTLine.empty());
                const cv::Rect oGTBBox = convertToRect(sCurrGTLine);

                // @@@@@ TODO : compute CLE/OR with oGTBBox and oOutputBBox here*
                ++nbFrames;
                if(decrochage == 0)
                {
                    double tmp = computeOR(oGTBBox, oOutputBBox);
                    if(tmp != 0)
                    {
                        meanOR += tmp;
                        meanCLE += computeCLE(oGTBBox, oOutputBBox);
                        nb_frame_decroche = 0; //reset
                        ++nbFrameOK;
                    } else {
                        nb_frame_decroche++;
                        if(nb_frame_decroche == NB_FRAME_BEFORE_DECROCHAGE)
                            decrochage = nbFrames;
                    }
                }
                // for display purposes only
                cv::Mat oDisplayFrame = oCurrFrame.clone();
                cv::rectangle(oDisplayFrame,oOutputBBox.tl(),oOutputBBox.br(),cv::Scalar_<uchar>(255,0,0),2); // output box = blue
                cv::rectangle(oDisplayFrame,oGTBBox.tl(),oGTBBox.br(),cv::Scalar_<uchar>(0,255,0),2); // target box = green
                cv::imshow("display",oDisplayFrame);
                cv::waitKey(1);
//                cv::waitKey(0);

            }
            if(decrochage == 0) decrochage = nbFrames; //no decrochage
            std::cout << "Mean CLE : " << meanCLE / nbFrameOK << std::endl;
            std::cout << "Mean OR : " << meanOR / nbFrameOK << std::endl;
            std::cout << "Suivi Algo : " << decrochage << "/" << nbFrames << std::endl;
            // @@@@ TODO : compute average CLE/OR measure for current sequence here

        }
        std::cout << "\nAll done." << std::endl;
    }
    catch(const cv::Exception& e) {
        std::cerr << "Caught cv::Exceptions: " << e.what() << std::endl;
    }
    catch(const std::runtime_error& e) {
        std::cerr << "Caught std::runtime_error: " << e.what() << std::endl;
    }
    catch(...) {
        std::cerr << "Caught unhandled exception." << std::endl;
    }
#ifdef _MSC_VER
    system("pause");
#endif //def(_MSC_VER)
    return 0;
}
