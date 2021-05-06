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
double cx = 6.900000e+02;
double cy = 2.331966e+02;
double fx = 9.842439e+02;
double fy = 9.808141e+02;

std::string getKittiName(int i) {
    std::string s;
    s.resize(10);
    sprintf(&s[0], "%010d", i);
    return s;
}

class Feature {
public:
    cv::KeyPoint kp;
    cv::Mat desc;
    int idx;
    int age;

    Feature(cv::KeyPoint akp, cv::Mat adesc) {
        static int I = 1000;
        idx = I++;
        kp = akp;
        desc = adesc.clone();
        age = 0;
    }
};

class Frame {
public:
    cv::Mat img;
    std::vector<Feature> fts;

    cv::Mat viz() {
        cv::Mat viz;
        cv::cvtColor(img, viz,cv::COLOR_GRAY2RGB);
        for(unsigned int i=0; i<fts.size(); i++) {
            int agec = fts[i].age * 70;
            if(agec > 255) agec = 255;
            cv::circle(viz, fts[i].kp.pt, 3, cv::Scalar(agec,0,255-agec), -1);
        }
        return viz;
    }

    void addFeatures(std::vector<cv::KeyPoint> &kp, cv::Mat desc) {
        for(int i=0; i<kp.size(); i++) {
            fts.push_back(Feature(kp[i], desc.row(i)));
        }
    }

    cv::Mat getDesc() {
        cv::Mat d = fts[0].desc.clone();
        d.resize(fts.size(), d.cols);
        for(int i=0; i<fts.size(); i++)
            fts[i].desc.row(0).copyTo(d.row(i));
        return d;
    }
};

typedef cv::ORB FEATURE;
//typedef cv::AKAZE FEATURE;
//typedef cv::xfeatures2d::SURF FEATURE;


class FeatureManager {
public:
    cv::Ptr<FEATURE> detector;
    cv::Ptr<cv::DescriptorMatcher>  matcher;

    std::vector<Frame> keys;

    void init() {
        detector = FEATURE::create(2000);
        matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");
    }

    void update(cv::Mat aFrame) {
        
        Frame f;
        std::vector<cv::KeyPoint> kp;
        cv::Mat desc; 
        f.img = aFrame.clone();
        detector->detectAndCompute(f.img, cv::Mat(), kp, desc);
        std::cout<<"features: "<<kp.size()<<"  desc: "<<desc.size()<<"\n";
        f.addFeatures(kp, desc);

        if(keys.size() > 0) {
            Frame &f1 = keys.back();
            Frame &f2 = f; 

            double knn_match_ratio=0.8;
            vector< vector<cv::DMatch> > matches_knn;
            matcher->knnMatch( f1.getDesc(), f2.getDesc(), matches_knn, 2 );
            vector< cv::DMatch > matches;
            for ( size_t i=0; i<matches_knn.size(); i++ ) {
                if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
                    matches.push_back( matches_knn[i][0] );
            }
            
            // update feature indecies
            std::vector<Feature> ff1, ff2; 
            for (unsigned int i=0; i<matches.size(); i++) {
                f2.fts[matches[i].trainIdx].idx = f1.fts[matches[i].queryIdx].idx;
                f2.fts[matches[i].trainIdx].age = f1.fts[matches[i].queryIdx].age +1;
            
                ff1.push_back(f1.fts[matches[i].queryIdx]);
                ff2.push_back(f2.fts[matches[i].trainIdx]);
            }
            f1.fts = ff1;
            f2.fts = ff2;
        } 
        keys.push_back(f);
    }


};


class BundleAdjustment {
public:
    g2o::SparseOptimizer    optimizer;
    typedef  g2o::BlockSolver_6_3 BlockSolverType;
    typedef  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* algorithm;

    std::vector<g2o::VertexSE3Expmap*> poses;
    std::vector<g2o::VertexSBAPointXYZ*> points;
    std::vector<g2o::EdgeProjectXYZ2UV*> edges;

    void init() {
        algorithm = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        optimizer.setAlgorithm( algorithm );
        optimizer.setVerbose( false );

        // Prepare camera parameters
        g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
        camera->setId(0);
        optimizer.addParameter( camera );
    }

    void addFrame(Frame &f) {
        // add frame pose
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(poses.size());
        if (v->id() == 0)
            v->setFixed( true ); //The first point is fixed to zero
        //The default value is the unit Pose, because we don’t know any information
        v->setEstimate( g2o::SE3Quat() );
        optimizer.addVertex( v );
        poses.push_back(v);

        // add new vertex of frame
        for ( size_t i=0; i<f.fts.size(); i++ ) {
            if(f.fts[i].age == 0) {
                g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
                v->setId(f.fts[i].idx);

                // Since the depth is unknown, I can only set the depth to 1
                double z = 1;
                double x = ( f.fts[i].kp.pt.x - cx ) * z / fx; 
                double y = ( f.fts[i].kp.pt.y - cy ) * z / fy; 
                v->setMarginalized(true);
                v->setEstimate( Eigen::Vector3d(x,y,z) );
                optimizer.addVertex( v );
                points.push_back(v);
            }
        }

        // add edges
        for ( size_t i=0; i<f.fts.size(); i++ ) {
            g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(f.fts[i].idx)) );
            edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>  (optimizer.vertex(poses.size()-1)) );
            edge->setMeasurement( Eigen::Vector2d(f.fts[i].kp.pt.x, f.fts[i].kp.pt.y ) );
            edge->setInformation( Eigen::Matrix2d::Identity() );
            edge->setParameterId(0, 0);
            // Kernel function
            edge->setRobustKernel( new g2o::RobustKernelHuber() );
            optimizer.addEdge( edge );
            edges.push_back(edge);
        }
    }

    void optimize() {
        cout<< " Start optimization " <<endl;
        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        optimizer.optimize(10000);
        cout<< " Optimization complete " <<endl;

        optimizer.save("ba.g2o");

        for(int i=0; i<poses.size(); i++) {
            g2o::VertexSE3Expmap* v = poses[i];
            Eigen::Isometry3d pose = v->estimate();
            cout<<"Pose="<<endl<<pose.matrix()<<endl;
        }

        std::vector<Eigen::Vector3d> pts;
        for ( size_t i=0; i<points.size(); i++ ) {
            g2o::VertexSBAPointXYZ* v = points[i];
            Eigen::Vector3d pos = v->estimate();
            if(v->edges().size()>1) {
                pts.push_back(pos);
            }
        }

        // save cloud
        std::ofstream out("pts.pcd");
        out<<"# .PCD v.5 - Point Cloud Data file format\n"
        <<"VERSION .5\n"
        <<"FIELDS x y z\n"
        <<"SIZE 4 4 4\n"
        <<"TYPE F F F\n"
        <<"COUNT 1 1 1\n"
        <<"WIDTH "<<pts.size()<<"\n"
        <<"HEIGHT 1\n"
        <<"POINTS "<<pts.size()<<"\n"
        <<"DATA ascii\n";
        for(int i=0; i<pts.size(); i++)
            out<<pts[i](0)<<" "<<pts[i](1)<<" "<<pts[i](2)<<endl;
    }

};



int main( int argc, char** argv )
{
    // Call format: command dataset (kitty raw format)
    if (argc < 2)
    {
        cout<<"Usage: tkSFM kitty_dir"<<endl;
        exit(1);
    }
    std::string data_path = std::string(argv[1]);
    std::string img_path = data_path + "/image_00/data/";

    int S = 0;
    int E = 10;
    if(argc >=3) S =std::atoi(argv[2]);
    if(argc >=4) E =std::atoi(argv[3]);


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

    BundleAdjustment ba;
    ba.init();
    for(int i=0; i<fm.keys.size(); i++) {
        ba.addFrame(fm.keys[i]);
    }
    ba.optimize();
    return 0;
}

