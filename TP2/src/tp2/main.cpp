#include "tp2/common.hpp"
#include <unistd.h>

int main(int /*argc*/, char** /*argv*/) {
    try {

         std::shared_ptr<Tracker> pAlgo = Tracker::createInstance();

        const std::string sBaseDataPath(DATA_ROOT_PATH "/tp2/");
        const std::vector<std::string> vsSequenceNames = {"dog","face","woman"};
        const std::vector<size_t> vnSequenceSizes = {988,892,597};
        for(size_t nSeqIdx=0; nSeqIdx<vsSequenceNames.size(); ++nSeqIdx) {
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

                // @@@@@ TODO : compute CLE/OR with oGTBBox and oOutputBBox here

                // for display purposes only
                cv::Mat oDisplayFrame = oCurrFrame.clone();
                cv::rectangle(oDisplayFrame,oOutputBBox.tl(),oOutputBBox.br(),cv::Scalar_<uchar>(255,0,0),2); // output box = blue
                cv::rectangle(oDisplayFrame,oGTBBox.tl(),oGTBBox.br(),cv::Scalar_<uchar>(0,255,0),2); // target box = green
                cv::imshow("display",oDisplayFrame);
                cv::waitKey(1);
//                int a;std::cin>>a;
//                usleep(100œ000);
//                usleep(5000000);

            }

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
