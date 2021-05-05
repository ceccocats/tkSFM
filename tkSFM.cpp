/**
 * BA Example 
 * Author: Xiang Gao
 * Date: 2016.3
 * Email: gaoxiang12@mails.tsinghua.edu.cn
 * 
 * In this program, we read two images and perform feature matching. Then according to the matching features, the camera motion and the location of the feature points are calculated. This is a typical Bundle Adjustment, we use g2o for optimization.
*/

// for std
#include <iostream>
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace std;

// Camera internal parameters
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

std::string getKittiName(int i) {
    std::string s;
    s.resize(10);
    sprintf(&s[0], "%010d", i);
    return s;
}

class Frame {
public:
    cv::Mat img;
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;

    cv::Mat viz() {
        cv::Mat viz;
        cv::cvtColor(img, viz,cv::COLOR_GRAY2RGB);
        for(unsigned int i=0; i<kp.size(); i++) {
            cv::circle(viz, kp[i].pt, 3, cv::Scalar(0,0,255), -1);
        }
        return viz;
    }
};

typedef cv::AKAZE FEATURE;

class FeatureManager {
public:
    cv::Ptr<FEATURE> detector;
    cv::Ptr<cv::DescriptorMatcher>  matcher;

    std::vector<Frame> keys;

    void init() {
        detector = FEATURE::create();
        matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");
    }

    void update(cv::Mat aFrame) {
        
        Frame f;
        f.img = aFrame.clone();
        detector->detectAndCompute(f.img, cv::Mat(), f.kp, f.desc);
        std::cout<<"features: "<<f.kp.size()<<"  desc: "<<f.desc.size()<<"\n";
        
        if(keys.size() > 0) {
            Frame &f1 = keys.back();
            Frame &f2 = f; 

            double knn_match_ratio=0.8;
            vector< vector<cv::DMatch> > matches_knn;
            matcher->knnMatch( f1.desc, f2.desc, matches_knn, 2 );
            vector< cv::DMatch > matches;
            for ( size_t i=0; i<matches_knn.size(); i++ ) {
                if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
                    matches.push_back( matches_knn[i][0] );
            }
            
            std::vector<cv::KeyPoint> kp1, kp2;
            cv::Mat desc1 = f1.desc.clone(); 
            cv::Mat desc2 = f2.desc.clone(); 
            desc1.resize(matches.size(), 32);
            desc2.resize(matches.size(), 32);
            for (unsigned int i=0; i<matches.size(); i++) {
                f1.desc.row(matches[i].queryIdx).copyTo(desc1.row(i));
                f2.desc.row(matches[i].trainIdx).copyTo(desc2.row(i));
                kp1.push_back(f1.kp[matches[i].queryIdx]);
                kp2.push_back(f2.kp[matches[i].trainIdx]);
            }
            f1.kp = kp1;
            f1.desc = desc1;
            f2.kp = kp2;
            f2.desc = desc2;
        } 
        keys.push_back(f);
    }


};


int main( int argc, char** argv )
{
    // Call format: command dataset (kitty raw format)
    if (argc != 2)
    {
        cout<<"Usage: tkSFM kitty_dir"<<endl;
        exit(1);
    }
    std::string data_path = std::string(argv[1]);
    std::string img_path = data_path + "/image_00/data/";
    int S = 0;
    int E = 50;

    FeatureManager fm; fm.init();

    for(int i=S; i<E; i++) {
        // read image
        std::string f_path = img_path + getKittiName(i) + ".png";
        std::cout<<"Load: "<<f_path<<"\n";
        cv::Mat raw = cv::imread(f_path, cv::IMREAD_GRAYSCALE); 
        
        fm.update(raw);

        cv::imshow("img",fm.keys.back().viz());
        cv::waitKey(0);
    }
    

    return 0;
}

