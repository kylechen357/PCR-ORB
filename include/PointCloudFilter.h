#ifndef POINTCLOUDFILTER_H
#define POINTCLOUDFILTER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <cuda_runtime.h>
#include "Frame.h"
#include "MapPoint.h"
#include <mutex>
#include <deque>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <future>
#include <atomic>
#include "Frame.h"

namespace ORB_SLAM3 {

class PointCloudFilter {
friend class Tracking;
public:
    PointCloudFilter();
    ~PointCloudFilter() = default;

    void FilterFrame(Frame &frame);
    void ClearCache();
    cv::Mat GetLatestSegMask() const;
    bool IsInitialized() const { return mbInitialized; }
    std::chrono::high_resolution_clock::time_point mLastFilterTime;
    float mAverageFilterTime;
    int mFilterCount;
    std::atomic<bool> mProcessingFrame{false};
    std::future<void> mFilterFuture;
    struct FilterStats {
        int totalPoints = 0;
        int dynamicFiltered = 0;
        int groundFiltered = 0;
        int skyFiltered = 0;
        int edgeFiltered = 0;
        int temporalFiltered = 0;  // Added for temporal/motion-based filtering
        
        // For confusion matrix
        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;
    };
    void ResetFilterStats();
    FilterStats GetFilterStats() const;
    void EvaluateFiltering(const std::vector<bool>& originalOutliers, 
                        const std::vector<bool>& filteredOutliers,
                        const Frame& frame);
    void SaveConfusionMatrix(const std::string& filename) const;

private:
    bool LoadModel();
    void PreprocessImage(const cv::Mat& input, torch::Tensor& output);
    cv::Mat ProcessYOLOOutput(const torch::Tensor& output, const cv::Size& imgSize);
    float ScorePoint(const cv::KeyPoint& kp, const cv::Mat& segMask, const cv::Mat& motionMask, 
                    float imageWidth, float imageHeight);
    void EstimateGroundPlane(const Frame& frame, std::vector<bool>& groundMask);
    bool IsGroundPoint(const cv::Point3f& pt, const cv::Mat& plane, float threshold);
    
    // New function for motion detection
    cv::Mat GenerateMotionMask(const cv::Mat& prevImg, const cv::Mat& currImg);

    // Model and device
    torch::jit::script::Module mSegModel;    
    bool mbInitialized;                      
    torch::Device mDevice;
    bool mCudaEnabled;

    // Model path and settings
    const std::string MODEL_PATH = "yolov8s-seg.torchscript";
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    
    // Modified filtering thresholds for better results
    const float DYNAMIC_MIN_THRESHOLD = 0.35f;  // Increased from 0.30f
    const float DYNAMIC_MAX_THRESHOLD = 0.75f;  // Increased from 0.70f
    const float GROUND_HEIGHT_RATIO = 0.80f;    // Increased from 0.8f
    const float SKY_THRESHOLD = 0.10f;          // Reduced from 0.15f
    const float EDGE_THRESHOLD = 25.0f;         // Reduced from 50.0f
    const float GROUND_PLANE_THRESHOLD = 0.12f; // Increased from 0.1f
    const int RANSAC_ITERATIONS = 200;          // Increased from 100
    
    // Parameters for clustering
    const float CLUSTER_DISTANCE = 25.0f;       // Increased from 20.0f
    const int MIN_CLUSTER_SIZE = 2;             // Reduced from 3
    const float MOTION_THRESHOLD = 0.25f;       // Reduced from 0.3f
    
    // Motion tracking
    cv::Mat mPrevImage;
    std::deque<cv::Mat> mMotionHistory;
    
    // Segmentation cache
    std::map<long unsigned int, cv::Mat> mSegCache;
    mutable std::mutex mMutexFilter;
    FilterStats mFilterStats;
    
    // Additional CUDA acceleration methods
    void WarmupCudaDevice();
    bool CheckCudaCompatibility();

    std::map<long unsigned int, int> mPointDynamicCount;  // Track dynamic detections by MapPoint ID
    const int CONSISTENCY_THRESHOLD = 3;  // Number of frames to consider a point truly dynamic
    const float DYNAMIC_SCORE_THRESHOLD = 0.3f;  // Score below which a point is considered potentially dynamic
};

} //namespace ORB_SLAM3

#endif // POINTCLOUDFILTER_H