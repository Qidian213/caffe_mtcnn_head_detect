// caffe

#include "mtcnnHead.h"
#include <caffe/caffe.hpp>
//#include <caffe/layers/memory_data_layer.hpp>

// c++
#include <string>
#include <vector>
#include <fstream>
// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <Common/LogEx.h>
#include <mutex>
#include<algorithm>
#define CPU_ONLY
#define INTER_FAST
using namespace caffe;
class MTCNNHead;

MTCNNHead *  caffe_InitHeadObj();
void caffe_headSetObjNotUse(MTCNNHead * obj);

 int caffe_detectHeadObj(MTCNNHead * obj, unsigned char * input, int w, int h, float* faceRect);
typedef struct FaceRect {
	float x1;
	float y1;
	float x2;
	float y2;
	float score; /**< Larger score should mean higher confidence. */
} FaceRect;

typedef struct FacePts {
	float x[5], y[5];
} FacePts;

typedef struct FaceInfo {
	FaceRect bbox;
	cv::Vec4f regression;
	FacePts facePts;
	double roll;
	double pitch;
	double yaw;
} FaceInfo;

class MTCNNHead {
public:
	MTCNNHead();
	void Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo, int minSize, double* threshold, double factor);
	void load(const string& proto_model_dir);
	void SetBusing(bool bUsing){ mbUsing = bUsing; }
	bool Busing(){ return mbUsing; }
	static std::string mModelPath;
private:
	void WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, caffe::shared_ptr<Net<float> >& net);
	bool CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum);
	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels, Blob<float>* input_layer,
		const int height, const int width);
	void SetMean();
	void GenerateBoundingBox(Blob<float>* confidence, Blob<float>* reg,
		float scale, float thresh, int image_width, int image_height);
	void ClassifyFace(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,
		caffe::shared_ptr<Net<float> >& net, double thresh, char netName);
	void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,
		caffe::shared_ptr<Net<float> >& net, double thresh, char netName);
	std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
	void Bbox2Square(std::vector<FaceInfo>& bboxes);
	void Padding(int img_w, int img_h);
	std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);
	void RegressPoint(const std::vector<FaceInfo>& faceInfo);

private:
	bool mbUsing;
	bool mbLoad;
	caffe::shared_ptr<Net<float> > PNet_;
	caffe::shared_ptr<Net<float> > RNet_;
	caffe::shared_ptr<Net<float> > ONet_;

	// x1,y1,x2,t2 and score
	std::vector<FaceInfo> condidate_rects_;
	std::vector<FaceInfo> total_boxes_;
	std::vector<FaceInfo> regressed_rects_;
	std::vector<FaceInfo> regressed_pading_;

	std::vector<cv::Mat> crop_img_;
	int curr_feature_map_w_;
	int curr_feature_map_h_;
	int num_channels_;
};
std::string MTCNNHead::mModelPath = "/bhps/rzbd/apache-tomcat-8.5.34/webapps/webroot/scanFile_upload";
// compare score
bool CompareBBoxMy(const FaceInfo & a, const FaceInfo & b) {
	return a.bbox.score > b.bbox.score;
}

// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Maximum)
std::vector<FaceInfo> MTCNNHead::NonMaximumSuppression(std::vector<FaceInfo>& bboxes,
	float thresh, char methodType){
	std::vector<FaceInfo> bboxes_nms;
	std::sort(bboxes.begin(), bboxes.end(), CompareBBoxMy);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
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

		FaceRect select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
		float x1 = static_cast<float>(select_bbox.x1);
		float y1 = static_cast<float>(select_bbox.y1);
		float x2 = static_cast<float>(select_bbox.x2);
		float y2 = static_cast<float>(select_bbox.y2);

		select_idx++;
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			FaceRect& bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
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



void MTCNNHead::Bbox2Square(std::vector<FaceInfo>& bboxes){
	for (int i = 0; i<bboxes.size(); i++){
		float h = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
		float w = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
		float side = h>w ? h : w;
		bboxes[i].bbox.x1 += (h - side)*0.5;
		bboxes[i].bbox.y1 += (w - side)*0.5;

		bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
		bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
		bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
		bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

	}
}

std::vector<FaceInfo> MTCNNHead::BoxRegress(std::vector<FaceInfo>& faceInfo, int stage){
	std::vector<FaceInfo> bboxes;
	for (int bboxId = 0; bboxId<faceInfo.size(); bboxId++){
		FaceRect faceRect;
		FaceInfo tempFaceInfo;
		float regh = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1;
		regh += (stage == 1) ? 0 : 1;
		float regw = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1;
		regw += (stage == 1) ? 0 : 1;
		faceRect.x1 = faceInfo[bboxId].bbox.x1 + regh * faceInfo[bboxId].regression[0];
		faceRect.y1 = faceInfo[bboxId].bbox.y1 + regw * faceInfo[bboxId].regression[1];
		faceRect.x2 = faceInfo[bboxId].bbox.x2 + regw * faceInfo[bboxId].regression[2];
		faceRect.y2 = faceInfo[bboxId].bbox.y2 + regh * faceInfo[bboxId].regression[3];
		faceRect.score = faceInfo[bboxId].bbox.score;

		tempFaceInfo.bbox = faceRect;
		tempFaceInfo.regression = faceInfo[bboxId].regression;

		bboxes.push_back(tempFaceInfo);
	}
	return bboxes;
}

// compute the padding coordinates (pad the bounding boxes to square)
void MTCNNHead::Padding(int img_w, int img_h){
	for (int i = 0; i<regressed_rects_.size(); i++){
		FaceInfo tempFaceInfo;
		tempFaceInfo = regressed_rects_[i];
		tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_h) ? img_h : regressed_rects_[i].bbox.y2;
		tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_w) ? img_w : regressed_rects_[i].bbox.x2;
		tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 <1) ? 1 : regressed_rects_[i].bbox.y1;
		tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 <1) ? 1 : regressed_rects_[i].bbox.x1;
		regressed_pading_.push_back(tempFaceInfo);
	}
}

void MTCNNHead::GenerateBoundingBox(Blob<float>* confidence, Blob<float>* reg,
	float scale, float thresh, int image_width, int image_height){
	int stride = 2;
	int cellSize = 12;

	int curr_feature_map_w_ = std::ceil((image_width - cellSize)*1.0 / stride) + 1;
	int curr_feature_map_h_ = std::ceil((image_height - cellSize)*1.0 / stride) + 1;

	//std::cout << "Feature_map_size:"<< curr_feature_map_w_ <<" "<<curr_feature_map_h_<<std::endl;
	int regOffset = curr_feature_map_w_*curr_feature_map_h_;
	// the first count numbers are confidence of face
	int count = confidence->count() / 2;
	const float* confidence_data = confidence->cpu_data();
	confidence_data += count;
	const float* reg_data = reg->cpu_data();
	char szBuf[256];
	condidate_rects_.clear();
	for (int i = 0; i<count; i++){
		if (*(confidence_data + i) >= thresh){
			int y = i / curr_feature_map_w_;
			int x = i - curr_feature_map_w_ * y;

			float xTop =((x*stride + 1) / scale);
			float yTop = ((y*stride + 1) / scale);
			float xBot = ((x*stride + cellSize - 1 + 1) / scale);
			float yBot = ((y*stride + cellSize - 1 + 1) / scale);
			FaceRect faceRect;
			faceRect.x1 = xTop;
			faceRect.y1 = yTop;
			faceRect.x2 = xBot;
			faceRect.y2 = yBot;
			faceRect.score = *(confidence_data + i);

			sprintf(szBuf,"%.1f %.1f",xBot-xTop,yBot-yTop);
			FaceInfo faceInfo;
			faceInfo.bbox = faceRect;
			faceInfo.regression = cv::Vec4f(reg_data[i + 0 * regOffset], reg_data[i + 1 * regOffset], reg_data[i + 2 * regOffset], reg_data[i + 3 * regOffset]);
			condidate_rects_.push_back(faceInfo);
		}
	}
}

MTCNNHead::MTCNNHead(){
	mbLoad = false;

}

void MTCNNHead::load(const std::string &proto_model_dir)
{
	if (!mbLoad)
	{


		/* Load the network. */
		PNet_.reset(new Net<float>((proto_model_dir + "/p.prototxt"), TEST));
		PNet_->CopyTrainedLayersFrom(proto_model_dir + "/pnet.caffemodel");

		CHECK_EQ(PNet_->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(PNet_->num_outputs(), 2) << "Network should have exactly two output, one"
			" is bbox and another is confidence.";

#ifdef CPU_ONLY
		RNet_.reset(new Net<float>((proto_model_dir + "/r.prototxt"), TEST));
#else
		RNet_.reset(new Net<float>((proto_model_dir + "/det2_input.prototxt"), TEST));
#endif
		RNet_->CopyTrainedLayersFrom(proto_model_dir + "/rnet.caffemodel");

		//  CHECK_EQ(RNet_->num_inputs(), 0) << "Network should have exactly one input.";
		//  CHECK_EQ(RNet_->num_outputs(),3) << "Network should have exactly two output, one"
		//                                     " is bbox and another is confidence.";

#ifdef CPU_ONLY
		ONet_.reset(new Net<float>((proto_model_dir + "/o.prototxt"), TEST));
#else
		ONet_.reset(new Net<float>((proto_model_dir + "/det3_input.prototxt"), TEST));
#endif
		ONet_->CopyTrainedLayersFrom(proto_model_dir + "/onet.caffemodel");

		//  CHECK_EQ(ONet_->num_inputs(), 1) << "Network should have exactly one input.";
		//  CHECK_EQ(ONet_->num_outputs(),3) << "Network should have exactly three output, one"
		//                                     " is bbox and another is confidence.";

		Blob<float>* input_layer;
		input_layer = PNet_->input_blobs()[0];
		num_channels_ = input_layer->channels();
		CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
		mbLoad = true;
		CLogEx::Loger("33..................\n");
	}

}

void MTCNNHead::WrapInputLayer(std::vector<cv::Mat>* input_channels,
	Blob<float>* input_layer, const int height, const int width) {
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void MTCNNHead::ClassifyFace(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,
	caffe::shared_ptr<Net<float> >& net, double thresh, char netName){
	int numBox = regressed_rects.size();
	Blob<float>* crop_input_layer = net->input_blobs()[0];
	int input_channels = crop_input_layer->channels();
	int input_width = crop_input_layer->width();
	int input_height = crop_input_layer->height();
	crop_input_layer->Reshape(1, input_channels, input_width, input_height);
	net->Reshape();

	condidate_rects_.clear();
	// load crop_img data to datum
	for (int i = 0; i<numBox; i++){
		std::vector<cv::Mat> channels;
		WrapInputLayer(&channels, net->input_blobs()[0], input_height,input_width );
		int pad_top = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
		int pad_left = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
		int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
		int pad_bottom = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);
	

		cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1 - 1, regressed_pading_[i].bbox.y2),
			cv::Range(regressed_pading_[i].bbox.x1 - 1, regressed_pading_[i].bbox.x2));
		cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));
#ifdef INTER_FAST
		cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_NEAREST);
#else
		cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
#endif
		crop_img = (crop_img - 127.5)*0.0078125;
		cv::split(crop_img, channels);

		CHECK(reinterpret_cast<float*>(channels.at(0).data) == net->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
		net->Forward();

		int reg_id = 0;
		int confidence_id = 1;

		const Blob<float>* reg = net->output_blobs()[reg_id];
		const Blob<float>* confidence = net->output_blobs()[confidence_id];


	
		
	

		const float* confidence_data = confidence->cpu_data() + confidence->count() / 2;
		const float* reg_data = reg->cpu_data();
	
	
		if (*(confidence_data) > thresh){
			FaceRect faceRect;
			faceRect.x1 = regressed_rects[i].bbox.x1;
			faceRect.y1 = regressed_rects[i].bbox.y1;
			faceRect.x2 = regressed_rects[i].bbox.x2;
			faceRect.y2 = regressed_rects[i].bbox.y2;
			faceRect.score = *(confidence_data);
			FaceInfo faceInfo;
			faceInfo.bbox = faceRect;
			faceInfo.regression = cv::Vec4f(reg_data[0], reg_data[1], reg_data[2], reg_data[3]);

			
			condidate_rects_.push_back(faceInfo);
		}
	}
	regressed_pading_.clear();
}
void MTCNNHead::WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, caffe::shared_ptr<Net<float> >& net)
{
	Blob<float> *input_layer = net->blob_by_name("data").get();

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float *input_data = input_layer->mutable_cpu_data();

	for (int j = 0; j < num; j++) {
		//std::vector<cv::Mat> *input_channels;
		for (int k = 0; k < input_layer->channels(); ++k) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
		cv::Mat img = imgs[j];
		cv::split(img, *input_channels);
		input_channels->clear();
	}
}
// multi test image pass a forward
void MTCNNHead::ClassifyFace_MulImage(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,
	caffe::shared_ptr<Net<float> >& net, double thresh, char netName){
	int numBox = regressed_rects.size();

	int input_width =24;
	int input_height = 24;
	if (netName == 'o')
	{
		 input_width = 48;
		 input_height = 48;
	}
	std::vector<cv::Mat> cur_imgs;
	// load crop_img data to datum
	for (int i = 0; i<numBox; i++) {
		int pad_top = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
		int pad_left = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
		int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
		int pad_bottom = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

		cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1 - 1, regressed_pading_[i].bbox.y2),
			cv::Range(regressed_pading_[i].bbox.x1 - 1, regressed_pading_[i].bbox.x2));
		cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));

#ifdef INTER_FAST
		cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_NEAREST);
#else
		cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
#endif
		crop_img = (crop_img - 127.5)*0.0078125;
		cur_imgs.push_back(crop_img);
	}
	regressed_pading_.clear();

	
	condidate_rects_.clear();
	Blob<float>* input_layer = net->blob_by_name("data").get();
	input_layer->Reshape(cur_imgs.size(), num_channels_,
		input_height, input_width);
	int num = input_layer->num();
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(cur_imgs, &input_channels, net);

	float no_use_loss = 0;
	net->Forward(&no_use_loss);
	//  CHECK(reinterpret_cast<float*>(crop_img_set.at(0).data) == net->input_blobs()[0]->cpu_data())
	//          << "Input channels are not wrapping the input layer of the network.";

	const caffe::shared_ptr<Blob<float> > reg = net->blob_by_name("fc3");
	const caffe::shared_ptr<Blob<float> > confidence = net->blob_by_name("prob");

	const float* confidence_data = confidence->cpu_data();
	const float* reg_data = reg->cpu_data();

	for (int i = 0; i<numBox; i++) {
		if (*(confidence_data + i * 2 + 1) > thresh) {
			FaceRect faceRect;
			faceRect.x1 = regressed_rects[i].bbox.x1;
			faceRect.y1 = regressed_rects[i].bbox.y1;
			faceRect.x2 = regressed_rects[i].bbox.x2;
			faceRect.y2 = regressed_rects[i].bbox.y2;
			faceRect.score = *(confidence_data + i * 2 + 1);
			FaceInfo faceInfo;
			faceInfo.bbox = faceRect;
			faceInfo.regression = cv::Vec4f(reg_data[4 * i + 0], reg_data[4 * i + 1], reg_data[4 * i + 2], reg_data[4 * i + 3]);

			
			condidate_rects_.push_back(faceInfo);
		}
	}


}
bool MTCNNHead::CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum){
	if (cv_mat.empty())
		return false;
	int channels = cv_mat.channels();

	datum->set_channels(cv_mat.channels());
	datum->set_height(cv_mat.rows);
	datum->set_width(cv_mat.cols);
	datum->set_label(0);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);

	int datum_height = datum->height();
	int datum_width = datum->width();
	if (channels == 3){
		for (int c = 0; c < channels; c++){
			for (int h = 0; h < datum_height; ++h){
				for (int w = 0; w < datum_width; ++w){
					const float* ptr = cv_mat.ptr<float>(h);
					datum->add_float_data(ptr[w*channels + c]);
				}
			}
		}
	}

	return true;
}

void MTCNNHead::Detect(const cv::Mat& image, std::vector<FaceInfo>& faceInfo, int minSize, double* threshold, double factor){

	// 2~3ms
	// invert to RGB color space and float type
	cv::Mat sample_single, resized;
	image.convertTo(sample_single, CV_32FC3);
	// cv::cvtColor(sample_single,sample_single,cv::COLOR_BGR2RGB);
	//sample_single = sample_single.t();

	int height = image.rows;
	int width = image.cols;
	int minWH = std::min(height, width);
	int factor_count = 0;
	double m = 12. / minSize;
	minWH *= m;
	std::vector<double> scales;
	while (minWH >= 12)
	{
		scales.push_back(m * std::pow(factor, factor_count));
		minWH *= factor;
		++factor_count;
	}

	// 11ms main consum
	Blob<float>* input_layer = PNet_->input_blobs()[0];
	for (int i = 0; i<factor_count; i++)
	{
		double scale = scales[i];
		int hs = std::ceil(height*scale);
		int ws = std::ceil(width*scale);

		// wrap image and normalization using INTER_AREA method
		cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_AREA);
#ifdef INTER_FAST
		cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_NEAREST);
#else
		cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_AREA);
#endif
		resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5*0.0078125);

		// input data
		input_layer->Reshape(1, 3, hs, ws);
		PNet_->Reshape();
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels, PNet_->input_blobs()[0], hs, ws);
		cv::split(resized, input_channels);

	
		PNet_->Forward();

		// return result
		Blob<float>* reg = PNet_->output_blobs()[1];
		//const float* reg_data = reg->cpu_data();
		Blob<float>* confidence = PNet_->output_blobs()[0];

		
	
		GenerateBoundingBox(confidence, reg, scale, threshold[0], ws, hs);


		std::vector<FaceInfo> bboxes_nms = NonMaximumSuppression(condidate_rects_, 0.5, 'u');
		total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
	}
	
	

	int numBox = total_boxes_.size();


	
//	faceInfo = total_boxes_;
	//return;
	//cv::Mat timg = image.clone();
	//for (int k = 0; k < total_boxes_.size(); ++k)
	//{
	//	cv::rectangle(timg, cv::Rect(total_boxes_[k].bbox.x1, total_boxes_[k].bbox.y1, total_boxes_[k].bbox.x2 - total_boxes_[k].bbox.x1,
	//		total_boxes_[k].bbox.y2 - total_boxes_[k].bbox.y1), cv::Scalar(255, 0, 0));
	//}
	//cv::imwrite("../result/1.jpg", timg);


	if (numBox != 0){

		total_boxes_ = NonMaximumSuppression(total_boxes_, 0.7, 'u');
	
		regressed_rects_ = BoxRegress(total_boxes_, 1);
		total_boxes_.clear();

		Bbox2Square(regressed_rects_);
		Padding(width, height);
	

		ClassifyFace_MulImage(regressed_rects_, sample_single, RNet_, threshold[1], 'r');

		condidate_rects_ = NonMaximumSuppression(condidate_rects_, 0.5, 'm');
		regressed_rects_ = BoxRegress(condidate_rects_, 2);

		Bbox2Square(regressed_rects_);
		Padding(width, height);

	
		/// three stage
		numBox = regressed_rects_.size();
		if (numBox != 0){

			ClassifyFace_MulImage(regressed_rects_, sample_single, ONet_, threshold[2], 'o');

			regressed_rects_ = BoxRegress(condidate_rects_, 3);
			faceInfo = NonMaximumSuppression(regressed_rects_, 0.5, 'm');
		}
	}
	regressed_pading_.clear();
	regressed_rects_.clear();
	condidate_rects_.clear();
}




void caffe_headSetModelPath(char * path)
{
	MTCNNHead::mModelPath = path;
}

class mtcnnHeadMgr
{
public:
	MTCNNHead * GetOneMtcnn()
	{
		MTCNNHead * mFind = NULL;
		mMutex.lock();
		for (int k = 0; k<m_listMtcnn.size(); ++k)
		{
			if (!m_listMtcnn[k]->Busing())
			{
				mFind = m_listMtcnn[k];
				mFind->SetBusing(true);
				break;
			}
		}
		if (mFind == NULL)
		{
			if (m_listMtcnn.size() < 50)
			{
				MTCNNHead * mtcnn = new MTCNNHead();
#ifdef _WIN32
				mtcnn->load(MTCNNHead::mModelPath +"/model/");
#else
				mtcnn->load(MTCNNHead::mModelPath + "/mode_smallface");
#endif
				mtcnn->SetBusing(true);
				m_listMtcnn.push_back(mtcnn);
				mFind = mtcnn;
			}
		}

		mMutex.unlock();
		return mFind;
	}
	void SetMtcnnNotUse(MTCNNHead * obj)
	{
		mMutex.lock();
		for (int k = 0; k<m_listMtcnn.size(); ++k)
		{
			if (m_listMtcnn[k] == obj)
			{
				m_listMtcnn[k]->SetBusing(false);
			}
		}
		mMutex.unlock();
	}
	mtcnnHeadMgr()
	{

	}
private:
	vector<MTCNNHead*> m_listMtcnn;
	std::mutex mMutex;


};
mtcnnHeadMgr gmtcnnHeadMgr;

MTCNNHead *  caffe_InitHeadObj()
{
	return gmtcnnHeadMgr.GetOneMtcnn();
}

void caffe_headSetObjNotUse(MTCNNHead * obj)
{
	gmtcnnHeadMgr.SetMtcnnNotUse(obj);
}

//detect
int caffe_headDetectMarks(char * rgbdata, int w, int h,  float * faceRect)
{

	int outx, outy, outw, outh;
	MTCNNHead * mtcnnObj = caffe_InitHeadObj();
	if (mtcnnObj == NULL) return -50;

	int iCount = caffe_detectHeadObj(mtcnnObj, (unsigned char*)rgbdata, w, h, faceRect);

	caffe_headSetObjNotUse(mtcnnObj);

	return iCount;
}

int caffe_detectHeadObj(MTCNNHead * mtcnnobj, unsigned char * input, int inw, int inh, float* fRect)
{

	//::google::InitGoogleLogging(argv[0]);
	double threshold[3] = { 0.8, 0.75, 0.88};
	double factor = 0.709;
	int minSize = 60;
	if (inw >= 600)
	{
		minSize = 70;
	}

	cv::Mat image(inh, inw, CV_8UC3);
	memcpy(image.data, input, inw*inh * 3);
	std::vector<FaceInfo> faceInfo;
	clock_t t1 = clock();

	mtcnnobj->Detect(image, faceInfo, minSize, threshold, factor);


	int outx, outy, outw, outh;
	int maxw = 0;
	int iCount = 0;
	
		for (int i = 0; i < std::min(50,(int)faceInfo.size()); i++)
		{
			float x = faceInfo[i].bbox.x1;
			float y = faceInfo[i].bbox.y1;
			float w = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
			float h = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;

			outx = x;
			outy = y;
			outw = w;
			outh = h;



			if (outx < 0) outx = 0;
			if (outy < 0) outy = 0;
			if (outx + outw > image.cols) outw = image.cols - outx;
			if (outy + outh > image.rows) outh = image.rows - outy;
			if (outw < 20 || outh < 20)
				continue;

			fRect[iCount * 5 + 0] = outx*1.0 / inw;
			fRect[iCount *5 + 1] = outy*1.0 / inh;
			fRect[iCount * 5 + 2] = outw*1.0 / inw;
			fRect[iCount * 5 + 3] = outh*1.0 / inh;

			iCount++;

		}
	
	return iCount;
}
