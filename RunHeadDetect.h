#ifndef _RUN_FACE_DETECT_H_
#define _RUN_FACE_DETECT_H_

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>

using namespace cv;
using namespace std;
using namespace cv::dnn;

#define MAX_NUM 64
#define MAX_LIST_NUM 256

const float pnet_stride = 2;
const float pnet_cell_size = 12;



typedef struct RunFaceRectInfo {
    cv::Rect m_rect;    // 目标坐标
    float confidence;   // 置信度
//    float quality;   // 质量分数
    float k_score; //人脸框因子
    float qa_score;   // 质量分数
    float pose_score;   // 质量分数
    int match;
} RunFaceRectInfo;

typedef struct RunFaceTrackInfo {
    int ID;
    cv::Rect m_RectList[MAX_LIST_NUM];// 目标轨迹链
    cv::Rect m_CurRect; // 当前目标框位置
    cv::Rect m_SmarllRect;
    cv::Rect m_cropRect;
    float maxpeak;
    float confidence;
//    float quality;   // 质量分数
    float k_score; //人脸框因子
    float qa_score;   // 质量分数
    float pose_score;   // 质量分数
    int objectClass;
    int listnum;
    int matchlable;
    int lostnum;
    int m_delete;
} RunFaceTrackInfo;

typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    cv::Rect m_rect;
    float confidence;    // 置信度
//    float quality;   // 置信度
    float k_score; //人脸框因子
    float qa_score;   // 质量分数
    float pose_score;   // 质量分数
    int match;
} FaceBox;

typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark_reg[10];
    float landmark[10];
    FaceBox bbox;
} FaceInfo;


class RunFaceDetect {
public:
    RunFaceDetect();

    ~RunFaceDetect();

public:
    void LoadModel(string pnet_prototxt, string pnet_model,
                   string rnet_prototxt, string rnet_model,
                   string onet_prototxt, string onet_model,
                   string PosePrototxt, string PoseModel,
                   string mobileQAprototxt, string mobileQAmodel,
                   int _minface, float _factor
    );

    void Detect(cv::Mat Src);

    void ProcessPnet(cv::Mat Src);

    void RNetDetect(const cv::Mat &img, vector <FaceInfo> &PNetResult);

    void ONetDetect(const cv::Mat &img, vector <FaceInfo> &RNetResult);

    vector<float> Head_Pose(const cv::Mat &img, vector <FaceInfo> &ResultBox);

    vector<float> QA(const cv::Mat &img, vector <FaceInfo> &ResultBox);


    void GenerateBBox(float *confidence, float *reg_box, int ws, int hs, float scale, float thresh);

    std::vector<FaceInfo> NMS(std::vector<FaceInfo> &bboxes, float thresh, char methodType);

//    void LoadModel(string prototxt, string model, string QAprototxt, string QAmodel);
//
//    void FaceDetect(cv::Mat Src);

    void Process(cv::Mat Src);

private:
    int VideoTrack(cv::Mat frame, cv::Mat orgframe);

public:
    int minface;
    float factor;
    vector<float> pose_score;
    vector<float> qa_score;
    std::vector<FaceInfo> ResultBox;
    vector<RunFaceRectInfo> m_FaceResult;
    RunFaceTrackInfo m_TrackResultInfo[MAX_NUM];

    cv::Mat QASrc[MAX_NUM];
    int m_TrackResultNum;

private:
    std::vector<FaceInfo> res_boxes;
    std::vector<FaceInfo> rnet_boxes;
    std::vector<FaceInfo> candidate_boxes_;
    dnn::Net net;
    dnn::Net rnet;
    dnn::Net onet;
    //dnn::Net posenet;
   // dnn::Net mobileqanet;

    std::vector<cv::String> outBlobNames_;
    std::vector<cv::Mat> outputBlobs_;
    float Scale_x;
    float Scale_y;
//    dnn::Net net;
//    dnn::Net net2;

};


#endif