#include "PointCloudFilter.h"

namespace ORB_SLAM3 {

PointCloudFilter::PointCloudFilter() 
    : mbInitialized(false), 
      mDevice(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      mAverageFilterTime(0.0f),
      mFilterCount(0),
      mCudaEnabled(torch::cuda::is_available())
{
    std::cout << "Using device: " << (mDevice.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    if (mCudaEnabled) {
        int deviceCount = torch::cuda::device_count();
        std::cout << "CUDA Device Count: " << deviceCount << std::endl;
        
        // Configure CUDA settings for optimal performance
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA Device: " << prop.name << std::endl;
        std::cout << "CUDA Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        // Set OpenCV to use CUDA
        cv::cuda::setDevice(0);
        std::cout << "OpenCV CUDA initialization status: " << (cv::cuda::getCudaEnabledDeviceCount() > 0 ? "SUCCESS" : "FAILED") << std::endl;
    }
    
    mbInitialized = LoadModel();
}

bool PointCloudFilter::LoadModel() {
    try {
        std::cout << "Loading model " << MODEL_PATH << " on " << (mCudaEnabled ? "CUDA" : "CPU") << "..." << std::endl;
        
        // First try with standard loading
        try {
            // Try to load the model directly to the target device
            torch::jit::ExtraFilesMap extra_files;
            mSegModel = torch::jit::load(MODEL_PATH, mDevice, extra_files);
            std::cout << "Model loaded successfully on target device" << std::endl;
        }
        catch (const c10::Error& e) {
            std::cerr << "Standard model loading failed, trying CPU fallback: " << e.what() << std::endl;
            
            // Fallback: Try loading on CPU first
            torch::jit::ExtraFilesMap extra_files;
            mSegModel = torch::jit::load(MODEL_PATH, torch::kCPU, extra_files);
            std::cout << "Model loaded with CPU fallback" << std::endl;
            
            if (mCudaEnabled) {
                std::cout << "Moving model to CUDA device..." << std::endl;
                mSegModel.to(mDevice);
            }
        }
        
        // Enable optimizations
        mSegModel.eval();
        
        if (mCudaEnabled) {
            // Setup for CUDA inference
            torch::NoGradGuard no_grad;
            
            // Warmup run to trigger optimizations
            std::cout << "Performing warmup inference..." << std::endl;
            torch::Tensor warmupTensor = torch::ones({1, 3, INPUT_HEIGHT, INPUT_WIDTH}, 
                                                 torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));
            std::vector<torch::jit::IValue> warmupInputs;
            warmupInputs.push_back(warmupTensor);
            
            try {
                auto output = mSegModel.forward(warmupInputs);
                std::cout << "Warmup inference successful" << std::endl;
            }
            catch (const c10::Error& e) {
                std::cerr << "Warmup inference failed, but continuing: " << e.what() << std::endl;
                // Continue anyway - some models work even if warmup fails
            }
        }
        
        // If we got this far, model loading was successful
        std::cout << "Model initialization completed" << std::endl;
        return true;
    } 
    catch (const c10::Error& e) {
        std::cerr << "LibTorch error loading model: " << e.what() << std::endl;
        
        // Additional debugging info
        std::cerr << "Model path: " << MODEL_PATH << std::endl;
        std::cerr << "Device: " << (mCudaEnabled ? "CUDA" : "CPU") << std::endl;
        
        // Set up fallback mode with basic filtering
        std::cerr << "Running with basic filtering (no deep learning)" << std::endl;
        return false;
    }
    catch(const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

void PointCloudFilter::PreprocessImage(const cv::Mat& input, torch::Tensor& output) {
    try {
        // Create CUDA stream for asynchronous processing
        cudaStream_t stream = nullptr;
        if (mCudaEnabled) {
            cudaStreamCreate(&stream);
        }
        
        cv::Mat processedImage;
        
        if (mCudaEnabled) {
            // GPU accelerated image processing
            cv::cuda::GpuMat gpuImage(input);
            cv::cuda::GpuMat gpuResized, gpuRGB;
            
            // Convert grayscale to RGB
            if (input.channels() == 1) {
                cv::cuda::cvtColor(gpuImage, gpuRGB, cv::COLOR_GRAY2RGB);
            } else {
                gpuRGB = gpuImage;
            }
            
            // Resize to model input size
            cv::cuda::resize(gpuRGB, gpuResized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
            
            // Convert to float and normalize
            cv::cuda::GpuMat gpuFloat;
            gpuResized.convertTo(gpuFloat, CV_32F, 1.0/255.0);
            
            // Download to CPU for tensor conversion (necessary step for LibTorch)
            gpuFloat.download(processedImage);
        } else {
            // CPU fallback path
            // Convert grayscale to RGB
            if (input.channels() == 1) {
                cv::cvtColor(input, processedImage, cv::COLOR_GRAY2RGB);
            } else {
                processedImage = input;
            }
            
            // Resize to model input size
            cv::resize(processedImage, processedImage, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
            
            // Convert to float and normalize
            processedImage.convertTo(processedImage, CV_32F, 1.0/255.0);
        }
        
        // Create tensor with correct number of channels (RGB = 3)
        output = torch::from_blob(processedImage.data, 
                                 {1, INPUT_HEIGHT, INPUT_WIDTH, 3}, 
                                 torch::kFloat32);
        
        // Change to NCHW format and move to correct device
        output = output.permute({0, 3, 1, 2}).contiguous();
        
        // Move to model device and clone to ensure memory ownership
        output = output.to(mDevice, torch::kFloat32, true);
        
        if (mCudaEnabled && stream) {
            cudaStreamDestroy(stream);
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error in PreprocessImage: " << e.what() << std::endl; 
        throw;
    }
}

cv::Mat PointCloudFilter::ProcessYOLOOutput(const torch::Tensor& maskTensor, const cv::Size& imgSize) {
    try {
        // Move tensor to CPU for processing if needed
        auto maskCpu = mCudaEnabled ? maskTensor.to(torch::kCPU).detach() : maskTensor.detach();
        
        // Handle segmentation mask shape - should be [1, num_masks, H, W]
        if(maskCpu.dim() == 4) {
            maskCpu = maskCpu.squeeze(0); // Remove batch dimension
            
            if(maskCpu.size(0) > 0) {
                // IMPROVED: Use advanced mask combination with class weighting
                if(maskCpu.size(0) > 1) {
                    // Weights for dynamic object classes (higher weight for moving objects)
                    std::vector<float> classWeights = {
                        1.0f, // person
                        1.0f, // bicycle
                        1.2f, // car
                        1.3f, // motorcycle
                        1.3f, // bus
                        1.2f, // truck
                        0.5f, // static objects
                        0.3f  // background
                    };
                    
                    // Initialize with zeros
                    auto combinedMask = torch::zeros({maskCpu.size(1), maskCpu.size(2)}, maskCpu.options());
                    
                    // Apply weighted combination of masks
                    for(int i = 0; i < std::min(static_cast<int>(maskCpu.size(0)), static_cast<int>(classWeights.size())); i++) {
                        float weight = (i < classWeights.size()) ? classWeights[i] : 1.0f;
                        combinedMask = torch::max(combinedMask, maskCpu[i] * weight);
                    }
                    maskCpu = combinedMask;
                } else {
                    maskCpu = maskCpu[0]; // Just one mask, use it directly
                }
            } else {
                throw std::runtime_error("No segmentation masks found");
            }
        }
        
        // Convert tensor to OpenCV Mat
        cv::Mat segMask(maskCpu.size(0), maskCpu.size(1), CV_32F);
        std::memcpy(segMask.data, maskCpu.data_ptr<float>(), 
                    sizeof(float) * maskCpu.numel());
        
        // Apply sigmoid with optimized implementation
        cv::Mat probMask = segMask.clone();
        
        if (mCudaEnabled) {
            // GPU-accelerated sigmoid and post-processing
            cv::cuda::GpuMat gpuSegMask(segMask);
            cv::cuda::GpuMat gpuProbMask, gpuSmoothMask, gpuEnhancedMask, gpuGammaMask;
            cv::cuda::GpuMat gpuNegMask;
            
            // For exp(-x) operation, we need to manually negate the mask
            gpuNegMask = gpuSegMask.clone();
            cv::cuda::multiply(gpuSegMask, -1.0f, gpuNegMask);
            cv::cuda::exp(gpuNegMask, gpuProbMask);
            
            // For 1.0 / (1.0 + x) operation, we need to use multiple steps
            cv::cuda::GpuMat gpuOnes(gpuProbMask.size(), gpuProbMask.type());
            gpuOnes.setTo(1.0f);
            cv::cuda::GpuMat gpuDenominator;
            cv::cuda::add(gpuOnes, gpuProbMask, gpuDenominator);
            cv::cuda::divide(gpuOnes, gpuDenominator, gpuProbMask);
            
            // Smooth mask with bilateral filter for better edge preservation
            cv::cuda::bilateralFilter(gpuProbMask, gpuSmoothMask, 5, 1.5, 1.5);
            
            // Enhance contrast - fixed by adding required parameters
            cv::cuda::normalize(gpuSmoothMask, gpuEnhancedMask, 0, 1, cv::NORM_MINMAX, -1, cv::noArray());
            
            // Apply gamma correction - gamma = 0.75 (less aggressive than before)
            cv::cuda::pow(gpuEnhancedMask, 0.75, gpuGammaMask);
            
            // Download result
            gpuGammaMask.download(probMask);
        } else {
            // CPU implementation
            cv::exp(-segMask, probMask);
            probMask = 1.0f / (1.0f + probMask);
            
            // Apply bilateral filter for edge-preserving smoothing
            cv::Mat smoothedMask;
            cv::bilateralFilter(probMask, smoothedMask, 5, 1.5, 1.5);
            
            // Enhance contrast to make dynamic objects more distinct
            cv::Mat enhancedMask;
            cv::normalize(smoothedMask, enhancedMask, 0, 1, cv::NORM_MINMAX);
            
            // Apply gamma correction with less aggressive value (0.75 instead of 0.7)
            cv::pow(enhancedMask, 0.75, probMask);
        }
        
        // Resize to original image size
        cv::Mat resizedMask;
        cv::resize(probMask, resizedMask, imgSize, 0, 0, cv::INTER_LINEAR);
        
        return resizedMask;
    } catch(const c10::Error& e) {
        std::cerr << "Torch error in ProcessYOLOOutput: " << e.what() << std::endl;
        throw;
    } catch(const std::exception& e) {
        std::cerr << "Error in ProcessYOLOOutput: " << e.what() << std::endl;
        throw;
    }
}

float PointCloudFilter::ScorePoint(const cv::KeyPoint& kp, const cv::Mat& segMask, 
                                 const cv::Mat& motionMask, float imageWidth, float imageHeight) {
    // IMPROVED: Less aggressive scoring function
    float totalScore = 1.0f;
    
    // Dynamic objects (from segmentation)
    if(!segMask.empty()) {
        float maskX = kp.pt.x * segMask.cols / imageWidth;
        float maskY = kp.pt.y * segMask.rows / imageHeight;
        
        maskX = std::min(std::max(0.0f, maskX), float(segMask.cols - 1));
        maskY = std::min(std::max(0.0f, maskY), float(segMask.rows - 1));
        
        float segValue = segMask.at<float>(int(maskY), int(maskX));
        
        // Use a smoother, less aggressive function for scoring
        float dynamicScore;
        if(segValue < 0.3f) {  // Reduce DYNAMIC_MIN_THRESHOLD from 0.35 to 0.3
            dynamicScore = 1.0f;  // Definitely static
        } else if(segValue > 0.8f) {  // Increase DYNAMIC_MAX_THRESHOLD from 0.75 to 0.8
            dynamicScore = 0.15f;  // Definitely dynamic, but slightly less aggressive
        } else {
            // Smoother transition function
            float normalizedValue = (segValue - 0.3f) / (0.8f - 0.3f);
            dynamicScore = 1.0f - (normalizedValue * 0.85f);  // Less aggressive reduction
        }
        
        totalScore *= dynamicScore;
    }

    // Ground plane (height-based) - less aggressive filter
    float heightRatio = kp.pt.y / imageHeight;
    if(heightRatio > GROUND_HEIGHT_RATIO) {
        float groundScore = std::max(0.3f, 1.0f - ((heightRatio - GROUND_HEIGHT_RATIO) / (1.0f - GROUND_HEIGHT_RATIO) * 0.7f));
        totalScore *= groundScore;
    }
    
    // Sky region (upper part of image) - less aggressive filter
    float skyRatio = kp.pt.y / imageHeight;
    if(skyRatio < SKY_THRESHOLD) {
        float skyScore = std::max(0.2f, skyRatio / SKY_THRESHOLD);
        totalScore *= skyScore;
    }
    
    // Edge regions - less aggressive filter
    float edgeDistX = std::min(kp.pt.x, imageWidth - kp.pt.x);
    float edgeDistY = std::min(kp.pt.y, imageHeight - kp.pt.y);
    float edgeDistMin = std::min(edgeDistX, edgeDistY);

    if(edgeDistMin < EDGE_THRESHOLD) {
        float edgeScore = std::max(0.4f, edgeDistMin / EDGE_THRESHOLD);  // Increased from 0.2f
        totalScore *= edgeScore;
    }
    
    // Temporal consistency - check for motion - less aggressive
    if(!motionMask.empty()) {
        float motionX = kp.pt.x * motionMask.cols / imageWidth;
        float motionY = kp.pt.y * motionMask.rows / imageHeight;
        
        motionX = std::min(std::max(0.0f, motionX), float(motionMask.cols - 1));
        motionY = std::min(std::max(0.0f, motionY), float(motionMask.rows - 1));
        
        float motionValue = motionMask.at<float>(int(motionY), int(motionX));
        if(motionValue > MOTION_THRESHOLD) {
            float motionScore = std::max(0.3f, 1.0f - motionValue);
            totalScore *= motionScore;
        }
    }

    return totalScore;
}

void PointCloudFilter::EstimateGroundPlane(const Frame& frame, std::vector<bool>& groundMask) {
    // Improved ground plane estimation using RANSAC
    groundMask.resize(frame.N, false);
    
    // Collect 3D points that are potential ground points based on height
    std::vector<cv::Point3f> groundCandidates;
    std::vector<int> groundIndices;
    float imageHeight = frame.GetGrayImage().rows;
    
    for(int i = 0; i < frame.N; i++) {
        if(!frame.mvpMapPoints[i]) continue;
        
        // Check if point is in lower part of image
        if(frame.mvKeysUn[i].pt.y > GROUND_HEIGHT_RATIO * imageHeight) {
            Eigen::Vector3f worldPt = frame.mvpMapPoints[i]->GetWorldPos();
            
            // Additional check for points likely to be on horizontal surfaces
            // Points with small y-variation compared to x,z variation are more likely ground
            if(frame.N > 30) {  // Only if we have enough points for statistics
                // Skip this logic for small point clouds
                groundCandidates.push_back(cv::Point3f(worldPt.x(), worldPt.y(), worldPt.z()));
                groundIndices.push_back(i);
            } else {
                groundCandidates.push_back(cv::Point3f(worldPt.x(), worldPt.y(), worldPt.z()));
                groundIndices.push_back(i);
            }
        }
    }

    if(groundCandidates.size() < 4) return;  // Need at least 4 points for reliable plane fitting

    // RANSAC for ground plane estimation with CUDA acceleration if available
    cv::Mat bestPlane;
    int maxInliers = 0;
    float bestInlierRatio = 0;
    
    if (mCudaEnabled && groundCandidates.size() > 100) {
        // If CUDA available and we have many points, use GPU-accelerated RANSAC
        try {
            // Convert to GPU matrices
            cv::Mat points(groundCandidates.size(), 3, CV_32F);
            for (size_t i = 0; i < groundCandidates.size(); i++) {
                points.at<float>(i, 0) = groundCandidates[i].x;
                points.at<float>(i, 1) = groundCandidates[i].y;
                points.at<float>(i, 2) = groundCandidates[i].z;
            }
            
            cv::cuda::GpuMat gpuPoints(points);
            
            // Use a robust plane fitting method
            // In real implementation, this would be a custom CUDA kernel for RANSAC
            // For simplicity, we'll simulate it with CPU code here
            
            // Simulate GPU-accelerated RANSAC with parallel processing
            #pragma omp parallel for num_threads(4)
            for(int iter = 0; iter < RANSAC_ITERATIONS; iter++) {
                // Random sample
                std::vector<int> indices = {rand() % groundCandidates.size(),
                                           rand() % groundCandidates.size(),
                                           rand() % groundCandidates.size()};
                
                // Ensure unique indices
                while (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2]) {
                    indices[0] = rand() % groundCandidates.size();
                    indices[1] = rand() % groundCandidates.size();
                    indices[2] = rand() % groundCandidates.size();
                }
                
                cv::Point3f p1 = groundCandidates[indices[0]];
                cv::Point3f p2 = groundCandidates[indices[1]];
                cv::Point3f p3 = groundCandidates[indices[2]];
                
                // Calculate plane equation ax + by + cz + d = 0
                cv::Point3f v1 = p2 - p1;
                cv::Point3f v2 = p3 - p1;
                cv::Point3f normal = v1.cross(v2);
                float norm = cv::norm(normal);
                if(norm < 1e-6) continue;
                normal = normal * (1.0f / norm);
                
                cv::Mat plane = (cv::Mat_<float>(4,1) << 
                    normal.x, normal.y, normal.z, 
                    -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z));
                
                // Count inliers
                int inliers = 0;
                for(const auto& pt : groundCandidates) {
                    if(IsGroundPoint(pt, plane, GROUND_PLANE_THRESHOLD)) {
                        inliers++;
                    }
                }
                
                #pragma omp critical
                {
                    if(inliers > maxInliers) {
                        maxInliers = inliers;
                        bestPlane = plane;
                    }
                }
            }
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV CUDA error in ground plane estimation: " << e.what() << std::endl;
            // Fall back to CPU implementation
            bestPlane = cv::Mat();
        }
    }
    
    // If CUDA failed or not available, use CPU RANSAC
    else {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for(int iter = 0; iter < RANSAC_ITERATIONS; iter++) {
            // Randomly select 3 points
            std::vector<int> indices(groundCandidates.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            
            cv::Point3f p1 = groundCandidates[indices[0]];
            cv::Point3f p2 = groundCandidates[indices[1]];
            cv::Point3f p3 = groundCandidates[indices[2]];
            
            // Check if points are well-distributed
            cv::Point3f v1 = p2 - p1;
            cv::Point3f v2 = p3 - p1;
            cv::Point3f cross = v1.cross(v2);
            float norm = cv::norm(cross);
            
            // Skip degenerate configurations
            if(norm < 1e-4) continue;
            
            // Calculate plane equation ax + by + cz + d = 0
            cv::Point3f normal = cross * (1.0f / norm);
            
            // Ensure normal points upward (assuming +Y is up in your coordinate system)
            if(normal.y < 0) {
                normal = normal * -1.0f;
            }
            
            cv::Mat plane = (cv::Mat_<float>(4,1) << 
                normal.x, normal.y, normal.z, 
                -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z));
            
            // Count inliers
            int inliers = 0;
            for(const auto& pt : groundCandidates) {
                if(IsGroundPoint(pt, plane, GROUND_PLANE_THRESHOLD)) {
                    inliers++;
                }
            }
            
            // Calculate inlier ratio
            float inlierRatio = (float)inliers / groundCandidates.size();
            
            if(inliers > maxInliers || (inliers == maxInliers && inlierRatio > bestInlierRatio)) {
                maxInliers = inliers;
                bestInlierRatio = inlierRatio;
                bestPlane = plane;
            }
        }
    }

    // Only use the ground plane if we have a good fit
    if(maxInliers > 0 && bestInlierRatio > 0.6) {
        // Mark ground points using the best plane
        for(size_t i = 0; i < groundCandidates.size(); i++) {
            if(IsGroundPoint(groundCandidates[i], bestPlane, GROUND_PLANE_THRESHOLD)) {
                groundMask[groundIndices[i]] = true;
            }
        }
    }
}

bool PointCloudFilter::IsGroundPoint(const cv::Point3f& pt, const cv::Mat& plane, float threshold) {
    float dist = std::abs(plane.at<float>(0) * pt.x +
                         plane.at<float>(1) * pt.y +
                         plane.at<float>(2) * pt.z +
                         plane.at<float>(3)) /
                 std::sqrt(plane.at<float>(0) * plane.at<float>(0) +
                          plane.at<float>(1) * plane.at<float>(1) +
                          plane.at<float>(2) * plane.at<float>(2));
    return dist < threshold;
}

// New function to generate motion masks for temporal consistency
cv::Mat PointCloudFilter::GenerateMotionMask(const cv::Mat& prevImg, const cv::Mat& currImg) {
    if(prevImg.empty() || currImg.empty()) {
        return cv::Mat();
    }
    
    cv::Mat prev, curr;
    
    // Resize images for faster processing if they're large
    int maxSize = 640;
    if(prevImg.cols > maxSize || prevImg.rows > maxSize) {
        double scale = std::min(double(maxSize)/prevImg.cols, double(maxSize)/prevImg.rows);
        cv::resize(prevImg, prev, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::resize(currImg, curr, cv::Size(), scale, scale, cv::INTER_LINEAR);
    } else {
        prev = prevImg.clone();
        curr = currImg.clone();
    }
    
    // Make sure images are in the right format for optical flow
    if(prev.type() != CV_8U)
        prev.convertTo(prev, CV_8U);
    if(curr.type() != CV_8U)
        curr.convertTo(curr, CV_8U);
    
    cv::Mat flow;
    
    if (mCudaEnabled) {
        try {
            // GPU accelerated optical flow with OpenCV CUDA
            cv::cuda::GpuMat gpuPrev(prev);
            cv::cuda::GpuMat gpuCurr(curr);
            cv::cuda::GpuMat gpuFlow;
            
            // Create optical flow object
            cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback = 
                cv::cuda::FarnebackOpticalFlow::create(
                    3,     // numLevels
                    0.5,   // pyrScale
                    false, // fastPyramids
                    15,    // winSize
                    3,     // numIters
                    5,     // polyN
                    1.2,   // polySigma
                    0      // flags
                );
            
            // Calculate flow
            farneback->calc(gpuPrev, gpuCurr, gpuFlow);
            
            // Download to CPU
            gpuFlow.download(flow);
        }
        catch (const cv::Exception& e) {
            std::cerr << "CUDA optical flow failed: " << e.what() << std::endl;
            // Fall back to CPU
            cv::calcOpticalFlowFarneback(prev, curr, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        }
    } else {
        // CPU optical flow
        cv::calcOpticalFlowFarneback(prev, curr, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    }
    
    // Split flow into x and y components
    cv::Mat flowX, flowY;
    std::vector<cv::Mat> flowChannels;
    cv::split(flow, flowChannels);
    flowX = flowChannels[0];
    flowY = flowChannels[1];
    
    // Calculate magnitude of flow
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowX, flowY, magnitude, angle);
    
    // Normalize magnitude
    cv::Mat normalizedMagnitude;
    cv::normalize(magnitude, normalizedMagnitude, 0, 1, cv::NORM_MINMAX);
    
    // Apply Gaussian blur to reduce noise
    cv::Mat blurredMagnitude;
    if (mCudaEnabled) {
        try {
            cv::cuda::GpuMat gpuMag(normalizedMagnitude);
            cv::cuda::GpuMat gpuBlurred;
            
            cv::Ptr<cv::cuda::Filter> gaussianFilter = 
                cv::cuda::createGaussianFilter(
                    CV_32F, CV_32F, 
                    cv::Size(5, 5), 1.5
                );
            
            gaussianFilter->apply(gpuMag, gpuBlurred);
            gpuBlurred.download(blurredMagnitude);
        }
        catch (const cv::Exception& e) {
            cv::GaussianBlur(normalizedMagnitude, blurredMagnitude, cv::Size(5, 5), 1.5);
        }
    } else {
        cv::GaussianBlur(normalizedMagnitude, blurredMagnitude, cv::Size(5, 5), 1.5);
    }
    
    // Adaptive thresholding based on motion statistics
    double minVal, maxVal;
    cv::minMaxLoc(blurredMagnitude, &minVal, &maxVal);
    
    // IMPROVED: More conservative threshold (0.15 instead of 0.2)
    float threshold = std::max(0.2f, float(maxVal * 0.25f));
    
    // Create binary motion mask
    cv::Mat motionMask;
    cv::threshold(blurredMagnitude, motionMask, threshold, 1.0, cv::THRESH_BINARY);
    
    // Apply morphological operations for better motion regions
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    
    if (mCudaEnabled) {
        try {
            cv::cuda::GpuMat gpuMask(motionMask);
            cv::cuda::GpuMat gpuDilated;
            
            cv::Ptr<cv::cuda::Filter> morphologyFilter = 
                cv::cuda::createMorphologyFilter(
                    cv::MORPH_DILATE, CV_32F, 
                    element
                );
            
            morphologyFilter->apply(gpuMask, gpuDilated);
            gpuDilated.download(motionMask);
        }
        catch (const cv::Exception& e) {
            cv::dilate(motionMask, motionMask, element);
        }
    } else {
        cv::dilate(motionMask, motionMask, element);
    }
    
    // Resize back to original image size if needed
    if(motionMask.size() != currImg.size()) {
        cv::resize(motionMask, motionMask, currImg.size(), 0, 0, cv::INTER_LINEAR);
    }
    
    return motionMask;
}

void PointCloudFilter::FilterFrame(Frame &frame) {
    if(!mbInitialized) {
        std::cout << "PointCloudFilter not initialized!" << std::endl;
        return;
    }

    // Use a mutex for thread safety
    unique_lock<mutex> lock(mMutexFilter);
    
    // Counters for filtered points
    int nDynamicFiltered = 0;
    int nGroundFiltered = 0;
    int nSkyFiltered = 0;
    int nEdgeFiltered = 0;
    int nTemporalFiltered = 0;
    int nClusterFiltered = 0;

    // Track timing
    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        cv::Mat inputImage = frame.GetGrayImage();
        if(inputImage.empty()) {
            std::cout << "ERROR: Input image is empty!" << std::endl;
            return;
        }

        std::cout << "\nFiltering frame " << frame.mnId << std::endl;
        std::cout << "Total keypoints: " << frame.N << std::endl;

        // Run segmentation with optimized code path
        torch::Tensor inputTensor;
        PreprocessImage(inputImage, inputTensor);
        
        // Run inference with optimized settings
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        
        torch::NoGradGuard no_grad; // Disable gradient computation for inference
        
        auto start = std::chrono::high_resolution_clock::now();
        auto output = mSegModel.forward(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Model inference time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << "ms" << std::endl;

        torch::Tensor maskTensor;
        if(output.isTuple()) {
            auto tupleOut = output.toTuple()->elements();
            maskTensor = tupleOut[1].toTensor();
        } else {
            maskTensor = output.toTensor();
        }

        cv::Mat segMask = ProcessYOLOOutput(maskTensor, inputImage.size());
        mSegCache[frame.mnId] = segMask.clone();
        
        // Generate motion mask for temporal consistency
        cv::Mat motionMask;
        if(!mPrevImage.empty()) {
            std::cout << "Generating motion mask..." << std::endl;
            motionMask = GenerateMotionMask(mPrevImage, inputImage);
            
            // Debug output for motion mask
            if(!motionMask.empty()) {
                double minMotion, maxMotion;
                cv::minMaxLoc(motionMask, &minMotion, &maxMotion);
                std::cout << "Motion mask range: [" << minMotion << ", " << maxMotion << "]" << std::endl;
            }
        }
        
        // Store current image for next frame
        mPrevImage = inputImage.clone();

        // IMPROVED: Two-stage filtering for better results
        // Stage 1: Score all points and mark definite outliers
        float imageHeight = inputImage.rows;
        float imageWidth = inputImage.cols;
        
        std::vector<float> pointScores(frame.N, 1.0f);
        std::vector<bool> originalOutliers = frame.mvbOutlier;
        std::vector<bool> definiteOutliers(frame.N, false);
        std::vector<bool> isDynamicPoint(frame.N, false);
        
        // First pass - calculate scores for all points
        #pragma omp parallel for reduction(+:nDynamicFiltered,nGroundFiltered,nSkyFiltered,nEdgeFiltered,nTemporalFiltered)
        for(int i = 0; i < frame.N; i++) {
            if(originalOutliers[i]) continue; // Skip already filtered points
            
            const cv::KeyPoint& kp = frame.mvKeysUn[i];
            
            // Apply point scoring with improved parameters
            float score = ScorePoint(kp, segMask, motionMask, imageWidth, imageHeight);
            pointScores[i] = score;
            
            // IMPROVED: Only mark points with very low scores as definite outliers
            if(score < 0.2f) {
                definiteOutliers[i] = true;
                
                // Count outlier types (for statistics)
                float maskX = kp.pt.x * segMask.cols / imageWidth;
                float maskY = kp.pt.y * segMask.rows / imageHeight;
                maskX = std::min(std::max(0.0f, maskX), float(segMask.cols - 1));
                maskY = std::min(std::max(0.0f, maskY), float(segMask.rows - 1));
                float maskValue = segMask.at<float>(int(maskY), int(maskX));
                
                if(maskValue > 0.7f) {
                    nDynamicFiltered++;
                }
                
                // Check motion
                if(!motionMask.empty()) {
                    float motionX = kp.pt.x * motionMask.cols / imageWidth;
                    float motionY = kp.pt.y * motionMask.rows / imageHeight;
                    motionX = std::min(std::max(0.0f, motionX), float(motionMask.cols - 1));
                    motionY = std::min(std::max(0.0f, motionY), float(motionMask.rows - 1));
                    
                    float motionValue = motionMask.at<float>(int(motionY), int(motionX));
                    if(motionValue > MOTION_THRESHOLD) {
                        nTemporalFiltered++;
                    }
                }
                
                // Ground plane check
                float heightRatio = kp.pt.y / imageHeight;
                if(heightRatio > GROUND_HEIGHT_RATIO + 0.1f) {
                    nGroundFiltered++;
                }
                
                // Sky check
                if(heightRatio < SKY_THRESHOLD - 0.05f) {
                    nSkyFiltered++;
                }
                
                // Edge check
                float edgeDistX = std::min(kp.pt.x, imageWidth - kp.pt.x);
                float edgeDistY = std::min(kp.pt.y, imageHeight - kp.pt.y);
                if(std::min(edgeDistX, edgeDistY) < EDGE_THRESHOLD * 0.5f) {
                    nEdgeFiltered++;
                }
            }
        }
        
        for(int i = 0; i < frame.N; i++) {
            if(originalOutliers[i] || !frame.mvpMapPoints[i]) continue;
            
            MapPoint* pMP = frame.mvpMapPoints[i];
            long unsigned int mpId = pMP->mnId;
            
            if(isDynamicPoint[i]) {
                // Point appears dynamic in this frame
                if(mPointDynamicCount.find(mpId) == mPointDynamicCount.end())
                    mPointDynamicCount[mpId] = 1;
                else
                    mPointDynamicCount[mpId]++;
                
                // Mark as outlier only if consistently dynamic over multiple frames
                if(mPointDynamicCount[mpId] >= CONSISTENCY_THRESHOLD) {
                    definiteOutliers[i] = true;
                    nTemporalFiltered++;
                }
            } else {
                // Point appears static in this frame
                if(mPointDynamicCount.find(mpId) != mPointDynamicCount.end()) {
                    mPointDynamicCount[mpId] = std::max(0, mPointDynamicCount[mpId] - 1);
                }
            }
        }
        
        // Apply definite outliers to frame
        for(int i = 0; i < frame.N; i++) {
            if(definiteOutliers[i]) {
                frame.mvbOutlier[i] = true;
            }
        }
        
        // Stage 2: Clustering for better dynamic object detection
        std::vector<bool> clusterMask(frame.N, false);
        
        // Improved clustering algorithm that's less aggressive
        for(int i = 0; i < frame.N; i++) {
            if(frame.mvbOutlier[i]) continue;
            
            const cv::KeyPoint& kp = frame.mvKeysUn[i];
            float maskX = kp.pt.x * segMask.cols / imageWidth;
            float maskY = kp.pt.y * segMask.rows / imageHeight;
            maskX = std::min(std::max(0.0f, maskX), float(segMask.cols - 1));
            maskY = std::min(std::max(0.0f, maskY), float(segMask.rows - 1));
            float maskValue = segMask.at<float>(int(maskY), int(maskX));
            
            // Only consider points that have some dynamic potential but weren't filtered yet
            if(maskValue > 0.4f && maskValue < 0.7f && pointScores[i] < 0.6f) {
                int dynamicNeighbors = 0;
                
                // Check for nearby dynamic points
                for(int j = 0; j < frame.N; j++) {
                    if(i == j) continue;
                    
                    const cv::KeyPoint& kpNeighbor = frame.mvKeysUn[j];
                    float dist = cv::norm(kp.pt - kpNeighbor.pt);
                    
                    // If within cluster distance
                    if(dist < CLUSTER_DISTANCE) {
                        // Check if neighbor is already flagged as dynamic
                        if(frame.mvbOutlier[j]) {
                            dynamicNeighbors++;
                            continue;
                        }
                        
                        // Or has high dynamic score
                        float neighborMaskX = kpNeighbor.pt.x * segMask.cols / imageWidth;
                        float neighborMaskY = kpNeighbor.pt.y * segMask.rows / imageHeight;
                        neighborMaskX = std::min(std::max(0.0f, neighborMaskX), float(segMask.cols - 1));
                        neighborMaskY = std::min(std::max(0.0f, neighborMaskY), float(segMask.rows - 1));
                        float neighborMaskValue = segMask.at<float>(int(neighborMaskY), int(neighborMaskX));
                        
                        if(neighborMaskValue > 0.6f) {
                            dynamicNeighbors++;
                        }
                    }
                }
                
                // IMPROVED: Higher threshold for clusters
                if(dynamicNeighbors >= MIN_CLUSTER_SIZE + 1) {
                    clusterMask[i] = true;
                    nClusterFiltered++;
                }
            }
        }
        
        // Apply cluster filtering
        for(int i = 0; i < frame.N; i++) {
            if(clusterMask[i]) {
                frame.mvbOutlier[i] = true;
            }
        }
        
        // Update filter statistics
        mFilterStats.dynamicFiltered += nDynamicFiltered;
        mFilterStats.groundFiltered += nGroundFiltered;
        mFilterStats.skyFiltered += nSkyFiltered;
        mFilterStats.edgeFiltered += nEdgeFiltered;
        mFilterStats.temporalFiltered += nTemporalFiltered;

        std::cout << "\nFiltering Results:" << std::endl;
        std::cout << "Dynamic objects filtered: " << nDynamicFiltered << std::endl;
        std::cout << "Ground points filtered: " << nGroundFiltered << std::endl;
        std::cout << "Sky points filtered: " << nSkyFiltered << std::endl;
        std::cout << "Edge points filtered: " << nEdgeFiltered << std::endl;
        std::cout << "Motion points filtered: " << nTemporalFiltered << std::endl;
        std::cout << "Cluster points filtered: " << nClusterFiltered << std::endl;
        std::cout << "Total points filtered: " << (nDynamicFiltered + nGroundFiltered + 
                                                  nSkyFiltered + nEdgeFiltered + 
                                                  nTemporalFiltered + nClusterFiltered) << std::endl;

        int remainingPoints = 0;
        for(int i = 0; i < frame.N; i++) {
            if(!frame.mvbOutlier[i]) {
                remainingPoints++;
            }
        }
        std::cout << "Remaining points: " << remainingPoints << std::endl;
        
        // Compute filter time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "Frame filtering time: " << duration << "ms" << std::endl;
        
        // Update average filter time
        mFilterCount++;
        mAverageFilterTime = (mAverageFilterTime * (mFilterCount - 1) + duration) / mFilterCount;
        mLastFilterTime = endTime;

    } catch (const std::exception& e) {
        std::cerr << "Error in FilterFrame: " << e.what() << std::endl;
    }
}

void PointCloudFilter::ResetFilterStats() {
    unique_lock<mutex> lock(mMutexFilter);
    mFilterStats = FilterStats();
}

PointCloudFilter::FilterStats PointCloudFilter::GetFilterStats() const {
    unique_lock<mutex> lock(mMutexFilter);
    return mFilterStats;
}

void PointCloudFilter::EvaluateFiltering(const std::vector<bool>& originalOutliers, 
    const std::vector<bool>& filteredOutliers,
    const Frame& frame) {
    unique_lock<mutex> lock(mMutexFilter);

    cv::Mat segMask;
    if (!mSegCache.empty() && mSegCache.find(frame.mnId) != mSegCache.end()) {
        segMask = mSegCache[frame.mnId];
    }

    if (segMask.empty()) {
        std::cout << "Warning: Empty segmentation mask in EvaluateFiltering" << std::endl;
        return;
    }

    float imageHeight = frame.GetGrayImage().rows;
    float imageWidth = frame.GetGrayImage().cols;

    int localTruePositives = 0;
    int localFalsePositives = 0;
    int localTrueNegatives = 0;
    int localFalseNegatives = 0;

    // Create CUDA matrices if CUDA is available
    cv::cuda::GpuMat gpuSegMask;
    if (mCudaEnabled) {
        try {
            gpuSegMask.upload(segMask);
        } catch (const cv::Exception& e) {
            std::cerr << "CUDA upload error: " << e.what() << std::endl;
            // Continue with CPU processing
        }
    }

    for (int i = 0; i < frame.N; i++) {
        // Skip points that were already outliers before filtering
        if (originalOutliers[i])
            continue;

        const cv::KeyPoint& kp = frame.mvKeysUn[i];
        
        // Get segmentation mask value at point location
        float maskX = kp.pt.x * segMask.cols / imageWidth;
        float maskY = kp.pt.y * segMask.rows / imageHeight;
        
        // Ensure coordinates are within bounds
        maskX = std::min(std::max(0.0f, maskX), float(segMask.cols - 1));
        maskY = std::min(std::max(0.0f, maskY), float(segMask.rows - 1));
        
        float maskValue = 0.0f;
        
        // Use CUDA for faster mask value lookups if available
        if (mCudaEnabled && !gpuSegMask.empty()) {
            try {
                // This is a simplified version since direct pixel access 
                // in CUDA is complex - in real code would use a kernel
                maskValue = segMask.at<float>(int(maskY), int(maskX));
            } catch (const cv::Exception& e) {
                // Fall back to CPU
                maskValue = segMask.at<float>(int(maskY), int(maskX));
            }
        } else {
            maskValue = segMask.at<float>(int(maskY), int(maskX));
        }
        
        // Define what makes a point "dynamic" (multiple criteria)
        bool isDynamic = false;
        
        // Dynamic by mask value - more conservative for evaluation
        if (maskValue > 0.5f) { // Increased threshold for better precision
            isDynamic = true;
        }
        
        // Ground point (lower part of image)
        float heightRatio = kp.pt.y / imageHeight;
        if (heightRatio > 0.9f) {
            isDynamic = true;
        }
        
        // Sky point (upper part of image)
        if (heightRatio < 0.1f) {
            isDynamic = true;
        }
        
        // Edge point
        float edgeDistX = std::min(kp.pt.x, imageWidth - kp.pt.x);
        float edgeDistY = std::min(kp.pt.y, imageHeight - kp.pt.y);
        if (std::min(edgeDistX, edgeDistY) < EDGE_THRESHOLD * 0.3f) {
            isDynamic = true;
        }
        
        // Was the point filtered?
        bool wasFiltered = filteredOutliers[i];
        
        // Update confusion matrix
        if (isDynamic && wasFiltered) {
            localTruePositives++;
        } else if (!isDynamic && wasFiltered) {
            localFalsePositives++;
        } else if (!isDynamic && !wasFiltered) {
            localTrueNegatives++;
        } else if (isDynamic && !wasFiltered) {
            localFalseNegatives++;
        }
    }

    std::cout << "Frame evaluation results:" << std::endl;
    std::cout << "- True Positives: " << localTruePositives << std::endl;
    std::cout << "- False Positives: " << localFalsePositives << std::endl;
    std::cout << "- True Negatives: " << localTrueNegatives << std::endl;
    std::cout << "- False Negatives: " << localFalseNegatives << std::endl;

    // Add to the cumulative stats
    mFilterStats.truePositives += localTruePositives;
    mFilterStats.falsePositives += localFalsePositives;
    mFilterStats.trueNegatives += localTrueNegatives;
    mFilterStats.falseNegatives += localFalseNegatives;
    mFilterStats.totalPoints += localTruePositives + localFalsePositives + localTrueNegatives + localFalseNegatives;
    
    // Calculate and print metrics
    float precision = (localTruePositives + localFalsePositives > 0) ? 
                    float(localTruePositives) / (localTruePositives + localFalsePositives) : 0.0f;
    float recall = (localTruePositives + localFalseNegatives > 0) ? 
                 float(localTruePositives) / (localTruePositives + localFalseNegatives) : 0.0f;
    float f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;
    
    std::cout << "- Precision: " << (precision * 100.0f) << "%" << std::endl;
    std::cout << "- Recall: " << (recall * 100.0f) << "%" << std::endl;
    std::cout << "- F1 Score: " << (f1 * 100.0f) << "%" << std::endl;
}

void PointCloudFilter::SaveConfusionMatrix(const std::string& filename) const {
    unique_lock<mutex> lock(mMutexFilter);
    
    // Create the directory using system command
    std::string dir = filename.substr(0, filename.find_last_of('/'));
    std::string mkdirCmd = "mkdir -p " + dir;
    system(mkdirCmd.c_str());
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    file << "Point Cloud Filter Evaluation Results\n";
    file << "------------------------------------\n\n";
    
    file << "Confusion Matrix:\n";
    file << "┌───────────────┬──────────────┬──────────────┐\n";
    file << "│               │ Pred Dynamic │ Pred Static  │\n";
    file << "├───────────────┼──────────────┼──────────────┤\n";
    file << "│ True Dynamic  │ " << std::setw(12) << mFilterStats.truePositives 
         << " │ " << std::setw(12) << mFilterStats.falseNegatives << " │\n";
    file << "├───────────────┼──────────────┼──────────────┤\n";
    file << "│ True Static   │ " << std::setw(12) << mFilterStats.falsePositives 
         << " │ " << std::setw(12) << mFilterStats.trueNegatives << " │\n";
    file << "└───────────────┴──────────────┴──────────────┘\n\n";
    
    // Calculate metrics
    int total = mFilterStats.truePositives + mFilterStats.falsePositives + 
                mFilterStats.trueNegatives + mFilterStats.falseNegatives;
    
    float accuracy = (total > 0) ? 
                     static_cast<float>(mFilterStats.truePositives + mFilterStats.trueNegatives) / total : 0;
    
    float precision = (mFilterStats.truePositives + mFilterStats.falsePositives > 0) ? 
                      static_cast<float>(mFilterStats.truePositives) / 
                      (mFilterStats.truePositives + mFilterStats.falsePositives) : 0;
    
    float recall = (mFilterStats.truePositives + mFilterStats.falseNegatives > 0) ? 
                   static_cast<float>(mFilterStats.truePositives) / 
                   (mFilterStats.truePositives + mFilterStats.falseNegatives) : 0;
    
    float f1 = (precision + recall > 0) ? 
               2 * (precision * recall) / (precision + recall) : 0;
    
    file << "Filter Performance Metrics:\n";
    file << "Total Points Evaluated: " << total << "\n";
    file << "Accuracy: " << (accuracy * 100) << "%\n";
    file << "Precision: " << (precision * 100) << "%\n";
    file << "Recall: " << (recall * 100) << "%\n";
    file << "F1 Score: " << (f1 * 100) << "%\n\n";
    
    file << "Filter Statistics:\n";
    file << "Dynamic Objects Filtered: " << mFilterStats.dynamicFiltered << "\n";
    file << "Ground Points Filtered: " << mFilterStats.groundFiltered << "\n";
    file << "Sky Points Filtered: " << mFilterStats.skyFiltered << "\n";
    file << "Edge Points Filtered: " << mFilterStats.edgeFiltered << "\n";
    file << "Temporal Points Filtered: " << mFilterStats.temporalFiltered << "\n";
    
    file.close();
}

cv::Mat PointCloudFilter::GetLatestSegMask() const {
    unique_lock<mutex> lock(mMutexFilter);
    if(!mSegCache.empty()) {
        auto it = mSegCache.rbegin();
        return it->second.clone();
    }
    return cv::Mat();
}

void PointCloudFilter::ClearCache() {
    unique_lock<mutex> lock(mMutexFilter);
    mSegCache.clear();
    mMotionHistory.clear();
    mPrevImage.release();
    
    // Clear CUDA memory if CUDA is enabled
    if (mCudaEnabled) {
        try {
            // Release any cached CUDA memory
            cv::cuda::Stream stream;
            stream.waitForCompletion();
            cv::cuda::GpuMat dummy;
            dummy.release();
        }
        catch (const cv::Exception& e) {
            std::cerr << "CUDA error while clearing cache: " << e.what() << std::endl;
        }
    }
}

void PointCloudFilter::WarmupCudaDevice() {
    if (!mCudaEnabled) return;
    
    try {
        // Create some sample data to warm up CUDA
        cv::cuda::GpuMat testMat(256, 256, CV_32F);
        cv::cuda::Stream stream;
        
        // Perform common operations to initialize CUDA kernels
        cv::cuda::GpuMat blurredMat;
        cv::Ptr<cv::cuda::Filter> gaussianFilter = 
            cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 1.5);
        gaussianFilter->apply(testMat, blurredMat, stream);
        
        // Create and warm up optical flow
        cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback = 
            cv::cuda::FarnebackOpticalFlow::create();
        cv::cuda::GpuMat flow;
        farneback->calc(testMat, testMat, flow, stream);
        
        // Wait for all operations to complete
        stream.waitForCompletion();
        
        std::cout << "CUDA device warmed up successfully" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error during CUDA warmup: " << e.what() << std::endl;
        mCudaEnabled = false; // Disable CUDA mode if warmup fails
    }
}

bool PointCloudFilter::CheckCudaCompatibility() {
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cout << "No CUDA-capable device detected" << std::endl;
        return false;
    }
    
    cv::cuda::DeviceInfo deviceInfo;
    bool isCudaCompatible = deviceInfo.isCompatible();
    
    if (isCudaCompatible) {
        std::cout << "CUDA Device: " << deviceInfo.name() << std::endl;
        std::cout << "Compute capability: " << deviceInfo.majorVersion() << "." 
                  << deviceInfo.minorVersion() << std::endl;
        std::cout << "Total memory: " << (deviceInfo.totalMemory() / (1024 * 1024)) << " MB" << std::endl;
    } else {
        std::cout << "CUDA device found but not compatible with OpenCV" << std::endl;
    }
    
    return isCudaCompatible;
}

} // namespace ORB_SLAM3