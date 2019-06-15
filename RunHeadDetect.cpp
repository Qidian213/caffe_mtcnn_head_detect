#include "RunHeadDetect.h"

#define  WORK_WIDTH 512
#define  WORK_HEIGHT 512


static inline float calc_overlap(cv::Rect &r1, cv::Rect &r2) {
    int x1 = max(r1.x, r2.x);
    int y1 = max(r1.y, r2.y);
    int x2 = min(r1.x + r1.width, r2.x + r2.width);
    int y2 = min(r1.y + r1.height, r2.y + r2.height);
    if (x1 > x2 || y1 > y2) {
        return 0;
    }
    float inter = (x2 - x1) * (y2 - y1);
    float join = r1.width * r1.height + r2.width * r2.height - inter;
    return inter / join;
}


RunFaceDetect::RunFaceDetect() {
    m_TrackResultNum = 0;
}

RunFaceDetect::~RunFaceDetect() {

}


void RunFaceDetect::LoadModel(string pnet_prototxt, string pnet_model,
                           string rnet_prototxt, string rnet_model,
                           string onet_prototxt, string onet_model,
                           string PosePrototxt, string PoseModel,
                           string mobileQAprototxt, string mobileQAmodel,
                           int _minface, float _factor
) {
    net = readNetFromCaffe(pnet_prototxt, pnet_model);
    rnet = readNetFromCaffe(rnet_prototxt, rnet_model);
    onet = readNetFromCaffe(onet_prototxt, onet_model);

	net.setPreferableTarget(DNN_TARGET_OPENCL);
	rnet.setPreferableTarget(DNN_TARGET_OPENCL);
	onet.setPreferableTarget(DNN_TARGET_OPENCL);

  //  posenet = readNetFromCaffe(PosePrototxt, PoseModel);
  //  mobileqanet = readNetFromCaffe(mobileQAprototxt, mobileQAmodel);

//	net.setPreferableTarget(DNN_TARGET_OPENCL);
//	rnet.setPreferableTarget(DNN_TARGET_OPENCL);
//	onet.setPreferableTarget(DNN_TARGET_OPENCL);

    minface = _minface;
    factor = _factor;
}

void RunFaceDetect::Detect(cv::Mat Src) {

	
    ProcessPnet(Src);
    RNetDetect(Src, res_boxes);
   ONetDetect(Src, rnet_boxes);
//
//    Head_Pose(Src, ResultBox);
//    QA(Src, ResultBox);
//
//    m_FaceResult.clear();
//    float w = Src.cols;
//    float h = Src.rows;
//    RunFaceRectInfo mm_V_RectInfo;
//    for (int i = 0; i < this->ResultBox.size(); i++) {
//        mm_V_RectInfo.m_rect.x = (int) this->ResultBox[i].bbox.xmin;
//        mm_V_RectInfo.m_rect.y = (int) this->ResultBox[i].bbox.ymin;
//        mm_V_RectInfo.m_rect.width = (int) (this->ResultBox[i].bbox.xmax - this->ResultBox[i].bbox.xmin);
//        mm_V_RectInfo.m_rect.height = (int) (this->ResultBox[i].bbox.ymax - this->ResultBox[i].bbox.ymin);
//        mm_V_RectInfo.confidence = this->ResultBox[i].bbox.confidence;
//        float k_score = mm_V_RectInfo.m_rect.width * mm_V_RectInfo.m_rect.height;
//        mm_V_RectInfo.qa_score = (qa_score[i] * k_score) * ((112 * 112) / (w * h)) ;
////        mm_V_RectInfo.qa_score = qa_score[i] ;
//        mm_V_RectInfo.pose_score = pose_score[i];
//        mm_V_RectInfo.k_score = k_score * ((112 * 112) / (w * h));
//        m_FaceResult.push_back(mm_V_RectInfo);
//    }
}

void RunFaceDetect::GenerateBBox(float *confidence, float *reg_box, int ws, int hs, float scale, float thresh) {
    int feature_map_h_ = std::ceil((hs - pnet_cell_size) * 1.0 / pnet_stride) + 1;
    int feature_map_w_ = std::ceil((ws - pnet_cell_size) * 1.0 / pnet_stride) + 1;

    int spatical_size = feature_map_w_ * feature_map_h_;
    const float *confidence_data = (float *) confidence + spatical_size;
    const float *reg_data = (float *) reg_box;
    candidate_boxes_.clear();
    for (int i = 0; i < spatical_size; i++) {
        if (confidence_data[i] >= thresh) {

            int y = i / feature_map_w_;
            int x = i - feature_map_w_ * y;
            FaceInfo faceInfo;
            FaceBox &faceBox = faceInfo.bbox;

            faceBox.xmin = (float) (x * pnet_stride) / scale;
            faceBox.ymin = (float) (y * pnet_stride) / scale;
            faceBox.xmax = (float) (x * pnet_stride + pnet_cell_size - 1.f) / scale;
            faceBox.ymax = (float) (y * pnet_stride + pnet_cell_size - 1.f) / scale;
            faceInfo.bbox_reg[0] = reg_data[i];
            faceInfo.bbox_reg[1] = reg_data[i + spatical_size];
            faceInfo.bbox_reg[2] = reg_data[i + 2 * spatical_size];
            faceInfo.bbox_reg[3] = reg_data[i + 3 * spatical_size];

            faceBox.confidence = confidence_data[i];
            candidate_boxes_.push_back(faceInfo);
        }
    }
}

bool CompareBBox(const FaceInfo &a, const FaceInfo &b) {
    return a.bbox.confidence > b.bbox.confidence;
}

std::vector<FaceInfo> RunFaceDetect::NMS(std::vector<FaceInfo> &bboxes, float thresh, char methodType) {
    std::vector<FaceInfo> bboxes_nms;
    if (bboxes.size() == 0) {
        return bboxes_nms;
    }
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int select_idx = 0;
    int num_bbox = static_cast<int>(bboxes.size());
    std::vector<int> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        FaceBox select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) *
                                         (select_bbox.ymax - select_bbox.ymin + 1));
        float x1 = static_cast<float>(select_bbox.xmin);
        float y1 = static_cast<float>(select_bbox.ymin);
        float x2 = static_cast<float>(select_bbox.xmax);
        float y2 = static_cast<float>(select_bbox.ymax);

        select_idx++;
#pragma omp parallel for num_threads(threads_num)
        for (int i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            FaceBox &bbox_i = bboxes[i].bbox;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
            float area_intersect = w * h;

            switch (methodType) {
                case 'u':
                    if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                        mask_merged[i] = 1;
                    break;
                case 'm':
                    if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
                        mask_merged[i] = 1;
                    break;
                default:
                    break;
            }
        }
    }
    return bboxes_nms;
}

void BBoxRegression(vector<FaceInfo> &bboxes) {
#pragma omp parallel for num_threads(threads_num)
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        float *bbox_reg = bboxes[i].bbox_reg;
        float w = bbox.xmax - bbox.xmin + 1;
        float h = bbox.ymax - bbox.ymin + 1;
        bbox.xmin += bbox_reg[0] * w;
        bbox.ymin += bbox_reg[1] * h;
        bbox.xmax += bbox_reg[2] * w;
        bbox.ymax += bbox_reg[3] * h;
    }
}

void BBoxPad(vector<FaceInfo> &bboxes, int width, int height) {
#pragma omp parallel for num_threads(threads_num)
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        bbox.xmin = round(max(bbox.xmin, 0.f));
        bbox.ymin = round(max(bbox.ymin, 0.f));
        bbox.xmax = round(min(bbox.xmax, width - 1.f));
        bbox.ymax = round(min(bbox.ymax, height - 1.f));
    }
}

void BBoxPadSquare(vector<FaceInfo> &bboxes, int width, int height) {
#pragma omp parallel for num_threads(threads_num)
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        float w = bbox.xmax - bbox.xmin + 1;
        float h = bbox.ymax - bbox.ymin + 1;
        float side = h > w ? h : w;
        bbox.xmin = round(max(bbox.xmin + (w - side) * 0.5f, 0.f));

        bbox.ymin = round(max(bbox.ymin + (h - side) * 0.5f, 0.f));
        bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
        bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
    }
}

void RunFaceDetect::ONetDetect(const cv::Mat &img, vector<FaceInfo> &RNetResult) {
    int pnetnum = RNetResult.size();

	ResultBox.clear();
	if (pnetnum <= 0)
	{
		return;
	}

    vector<cv::Mat> vec_img;
    ResultBox.clear();

    for (int i = 0; i < pnetnum; i++) {
        cv::Mat patch;
        cv::Rect face_rect;
        face_rect.x = RNetResult[i].bbox.xmin;
        face_rect.y = RNetResult[i].bbox.ymin;
        face_rect.width = (int) (RNetResult[i].bbox.xmax - RNetResult[i].bbox.xmin + 1);
        face_rect.height = (int) (RNetResult[i].bbox.ymax - RNetResult[i].bbox.ymin + 1);
        cv::resize(img(face_rect), patch, cv::Size(48, 48), cv::INTER_AREA);
        vec_img.push_back(patch);
    }

  //  cv::Mat inPutBlob = cv::dnn::blobFromImages(vec_img, 0.0078, cv::Size(48, 48), cv::Scalar(127.5, 127.5, 127.5), 0);
	cv::Mat inPutBlob = cv::dnn::blobFromImages(vec_img, 0.0039, cv::Size(48, 48), cv::Scalar(104, 117, 123), 0);
    onet.setInput(inPutBlob, "data");

    std::vector<cv::Mat> outputBlobs;
    outputBlobs.clear();
    std::vector<cv::String> outBlobNames;
    outBlobNames.push_back("fc3");
  //  outBlobNames.push_back("conv6-3");
    outBlobNames.push_back("prob");

    onet.forward(outputBlobs, outBlobNames);

    const float *reg_data = (float *) outputBlobs[0].ptr<float>();
 //   const float *landmark_data = (float *) outputBlobs[1].ptr<float>();
    const float *confidence_data = (float *) outputBlobs[1].ptr<float>();


    for (int i = 0; i < pnetnum; i++) {
        if (*(confidence_data + 1) > 0.8) {
            FaceInfo info;
            info.bbox.confidence = *(confidence_data + 1);
            info.bbox.xmin = RNetResult[i].bbox.xmin;
            info.bbox.ymin = RNetResult[i].bbox.ymin;
            info.bbox.xmax = RNetResult[i].bbox.xmax;
            info.bbox.ymax = RNetResult[i].bbox.ymax;
            for (int n = 0; n < 4; ++n) {
                info.bbox_reg[n] = reg_data[n];
            }
            float w = info.bbox.xmax - info.bbox.xmin + 1.f;
            float h = info.bbox.ymax - info.bbox.ymin + 1.f;
           /* for (int n = 0; n < 5; ++n) {
                info.landmark[2 * n] = landmark_data[2 * n] * w + info.bbox.xmin;
                info.landmark[2 * n + 1] = landmark_data[2 * n + 1] * h + info.bbox.ymin;
            }*/

            ResultBox.push_back(info);
        }
        confidence_data += 2;
        reg_data += 4;
       /* landmark_data += 10;*/
    }
    BBoxRegression(ResultBox);
    ResultBox = NMS(ResultBox, 0.5f, 'm');
    BBoxPad(ResultBox, img.cols, img.rows);

    for (int i = 0; i < ResultBox.size(); i++) {
        float onet_width = ResultBox[i].bbox.xmax - ResultBox[i].bbox.xmin;
        float onet_height = ResultBox[i].bbox.ymax - ResultBox[i].bbox.ymin;
        float maxgrap = onet_width * 0.50 + onet_height * 0.5;
        ResultBox[i].bbox.xmin += (onet_width - maxgrap) * 0.5;
        ResultBox[i].bbox.ymin += (onet_height - maxgrap) * 0.5;
        ResultBox[i].bbox.xmin = std::max(ResultBox[i].bbox.xmin, 0.f);
        ResultBox[i].bbox.ymin = std::max(ResultBox[i].bbox.ymin, 0.f);
        ResultBox[i].bbox.ymin += maxgrap / 10;
        maxgrap = std::min(img.cols - 1 - ResultBox[i].bbox.xmin,
                           std::min(img.rows - ResultBox[i].bbox.ymin - 1, maxgrap));
        ResultBox[i].bbox.xmax = ResultBox[i].bbox.xmin + maxgrap;
        ResultBox[i].bbox.ymax = ResultBox[i].bbox.ymin + maxgrap;

    }
}

void RunFaceDetect::RNetDetect(const cv::Mat &img, vector<FaceInfo> &PNetResult) {
    int pnetnum = PNetResult.size();
	rnet_boxes.clear();
	if (pnetnum <= 0)
	{
		return;
	}
    vector<cv::Mat> vec_img;

    for (int i = 0; i < pnetnum; i++) {
        cv::Mat patch;
        cv::Rect face_rect;
        face_rect.x = PNetResult[i].bbox.xmin;
        face_rect.y = PNetResult[i].bbox.ymin;
        face_rect.width = (int) (PNetResult[i].bbox.xmax - PNetResult[i].bbox.xmin + 1);
        face_rect.height = (int) (PNetResult[i].bbox.ymax - PNetResult[i].bbox.ymin + 1);
        cv::resize(img(face_rect), patch, cv::Size(24, 24), cv::INTER_AREA);
        vec_img.push_back(patch);
    }

//    cv::Mat inPutBlob = cv::dnn::blobFromImages(vec_img, 0.0078, cv::Size(24, 24), cv::Scalar(127.5, 127.5, 127.5), 0);
	cv::Mat inPutBlob = cv::dnn::blobFromImages(vec_img, 0.0039, cv::Size(24, 24), cv::Scalar(104, 117, 123), 0);
    rnet.setInput(inPutBlob, "data");

    std::vector<cv::Mat> outputBlobs;
    outputBlobs.clear();
    std::vector<cv::String> outBlobNames;
    outBlobNames.push_back("fc3");
    outBlobNames.push_back("prob");
    rnet.forward(outputBlobs, outBlobNames);

    rnet_boxes.clear();
    const float *reg_data = (float *) outputBlobs[0].ptr<float>();
    const float *confidence_data = (float *) outputBlobs[1].ptr<float>();


    for (int i = 0; i < pnetnum; i++) {
        if (*(confidence_data + 1) > 0.7) {
            FaceInfo info;
            info.bbox.confidence = *(confidence_data + 1);
            info.bbox.xmin = PNetResult[i].bbox.xmin;
            info.bbox.ymin = PNetResult[i].bbox.ymin;
            info.bbox.xmax = PNetResult[i].bbox.xmax;
            info.bbox.ymax = PNetResult[i].bbox.ymax;
            for (int n = 0; n < 4; ++n) {
                info.bbox_reg[n] = reg_data[n];
            }

            rnet_boxes.push_back(info);
        }
        confidence_data += 2;
        reg_data += 4;
    }

    rnet_boxes = NMS(rnet_boxes, 0.5f, 'u');
    BBoxRegression(rnet_boxes);
    BBoxPadSquare(rnet_boxes, img.cols, img.rows);


}

void RunFaceDetect::ProcessPnet(cv::Mat Src) {
    cv::Mat resized;
    int width = Src.cols;
    int height = Src.rows;
    float scale = 12.f / minface;
    float minWH = std::min(height, width) * scale;
    std::vector<float> scales;
    while (minWH >= 12) {
        scales.push_back(scale);
        minWH *= factor;
        scale *= factor;
    }


    std::vector<FaceInfo> total_boxes_;
    total_boxes_.clear();

    for (int i = 0; i < scales.size(); i++)
        //for (int i = 0; i < 1; i++)
    {
        int ws = (int) std::ceil(width * scales[i]);
        int hs = (int) std::ceil(height * scales[i]);

      //  Mat inputBlob = blobFromImage(Src, 0.0078, Size(ws, hs), Scalar(127.5, 127.5, 127.5), false, false);
		Mat inputBlob = blobFromImage(Src, 0.0039, Size(ws, hs), Scalar(104, 117, 123), false, false);
        net.setInput(inputBlob, "data");

        vector<String> outBlobNames;
        outBlobNames.push_back("conv4_2");
        outBlobNames.push_back("cls_prob");
        vector<Mat> outputBlobs;
        net.forward(outputBlobs, outBlobNames);

        float *confidence = (float *) outputBlobs[1].data;
        float *reg = (float *) outputBlobs[0].data;


        GenerateBBox(confidence, reg, ws, hs, scales[i], 0.7);
        std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.4, 'u');

        if (bboxes_nms.size() > 0) {
            total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
        }
    }

    res_boxes.clear();

    int num_box = (int) total_boxes_.size();
    if (num_box != 0) {
        res_boxes = NMS(total_boxes_, 0.4f, 'u');
        BBoxRegression(res_boxes);
        BBoxPadSquare(res_boxes, width, height);
    }

}
