#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/ml/ml.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

#define VOCABULARY_SIZE 100

int main(int argc, char** argv)
{
    //variable define
    Mat img, mask;
    vector<KeyPoint> keypoints;
    cout << "create detector and extractor" << endl;
    Ptr<FeatureDetector> detector = new SURF();
    if(NULL == detector)
    {
        cout << "fail to create detector" << endl;
        waitKey(2000);
        return -1;
    }
    Ptr<DescriptorExtractor> extractor = new SIFT();
    if(NULL == extractor)
    {
        cout << "fail to create extractor" << endl;
        waitKey(2000);
        return -1;
    }

    Mat training_descriptors(0, extractor->descriptorSize(), extractor->descriptorType());

    //compute descriptors from images configured in training_bow.txt
    char buf[255];
    string database_dir("/home/cai-mj/Database/");
    ifstream ifs_bow("training_bow.txt");
    cout << "start computing descriptors for bag of words..." << endl;
    while(!ifs_bow.eof())
    {
        ifs_bow.getline(buf, 255);
        string line(buf);
        istringstream iss(line);
        string impath;
        iss >> impath;
        impath = database_dir + impath;
        img = imread(impath, 0);
        if(img.empty())
        {
            cout << "failed to read image " << impath << endl;
            waitKey(2000);
            return -1;
        }
        cout << "keypoint detecting for " << impath << endl;
        detector->detect(img, keypoints, mask);
        Mat descriptors;
        extractor->compute(img, keypoints, descriptors);
        cout << "finish computing keypoint descriptors" << endl;
        training_descriptors.push_back(descriptors);
    }

    //construct the vocabulary
    cout << "start constructing vocabulary..." << endl;
    BOWKMeansTrainer bowtrainer(VOCABULARY_SIZE);
    bowtrainer.add(training_descriptors);
    Mat vocabulary = bowtrainer.cluster();
    cout << "finish constructing vocabulary" << endl;

    //save vocabulary as xml file
    cout << "save codebook as xml file..." << endl;
    FileStorage cvfs("animal_bow.xml", CV_STORAGE_WRITE);
    write(cvfs, "animal_bow", vocabulary);

    Ptr<DescriptorMatcher> matcher(new BruteForceMatcher<L2<float> >());
    BOWImgDescriptorExtractor bowide(extractor, matcher);
    bowide.setVocabulary(vocabulary);

    //compute image descriptor of training set for classifiers
    map<string, Mat> classifier_training_data;
    classifier_training_data.clear();
    Mat imgDescriptor;
    cout << "computing image descriptors of training set..." << endl;
    ifstream ifs_train("training.txt");
    while(!ifs_train.eof())
    {
        ifs_train.getline(buf, 255);
        string line(buf);
        istringstream iss(line);
        string impath, class_name;
        iss >> impath >> class_name;
        impath = database_dir + impath;

        cout << "read image with label: " << class_name << " from " << impath << endl;
        img = imread(impath, 0);
        if(img.empty())
        {
            cout << "failed reading image" << endl;
            waitKey(2000);
            return -1;
        }

        detector->detect(img, keypoints, mask);
        bowide.compute(img, keypoints, imgDescriptor);
        if(0 == classifier_training_data.count(class_name))
        {
            classifier_training_data[class_name].create(0, imgDescriptor.cols, imgDescriptor.type());
        }
        classifier_training_data[class_name].push_back(imgDescriptor);
    }

    //train 1 vs rest svm classifiers
    cout << "start training..." << endl;
    map<string, CvSVM> svm_classifiers;
    for(map<string, Mat>::iterator it1 = classifier_training_data.begin(); it1 != classifier_training_data.end(); ++it1)
    {
        string positive_class = (*it1).first;
        cout << "training svm for class: " << positive_class << endl;
        Mat samples(0, imgDescriptor.cols, imgDescriptor.type());
        Mat labels(0, 1, CV_32F);

        //positive samples and labels
        samples.push_back(classifier_training_data[positive_class]);
        Mat positive_labels = Mat::ones(classifier_training_data[positive_class].rows, 1, CV_32F);
        labels.push_back(positive_labels);

        //rest samples and labels
        for(map<string, Mat>::iterator it2 = classifier_training_data.begin(); it2 != classifier_training_data.end(); ++it2)
        {
            string negtive_class = (*it2).first;
            if(negtive_class == positive_class)
            {
                continue;
            }
            samples.push_back(classifier_training_data[negtive_class]);
            Mat negative_labels = Mat::zeros(classifier_training_data[negtive_class].rows, 1, CV_32F);
            labels.push_back(negative_labels);
        }

        //train svm classifier
        Mat samples_32f;
        samples.convertTo(samples_32f, CV_32F);
        svm_classifiers[positive_class].train(samples_32f, labels);
    }

    //test
    cout << "testing..." << endl;
    ifstream ifs_test("test.txt");
    while(!ifs_test.eof())
    {
        ifs_test.getline(buf, 255);
        string line(buf);
        istringstream iss(line);
        string impath, class_name;
        iss >> impath >> class_name;
        impath = database_dir + impath;

        img = imread(impath, 0);
        if(img.empty())
        {
            cout << "failed reading image from" << impath << endl;
            waitKey(2000);
            return -1;
        }
        
        cout << "image ground-truth: " << class_name << endl;
        detector->detect(img, keypoints, mask);
        bowide.compute(img, keypoints, imgDescriptor);
        for(map<string, CvSVM>::iterator it = svm_classifiers.begin(); it != svm_classifiers.end(); ++it)
        {
            float score = (*it).second.predict(imgDescriptor, true);
            cout << "svm class: " << (*it).first << " distance to the margin: " << score << endl;
        }
    }
    cout << "done!" << endl;

    return 0;
}
