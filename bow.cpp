#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;
#define P_TRAINING_SIZE 10
#define VOCABULARY_SIZE 100

int main(int argc, char** argv)
{
    //variable define
    Mat mask;
    vector<KeyPoint> keypoints;
    SiftDescriptorExtractor extractor;
    Mat training_descriptors(0, extractor.descriptorSize(), extractor.descriptorType());

    string str1 = "cat";
    string str2 = ".jpg";
	string filename;

    //compute descriptors from a loop of images
    for(int i = 0; i < P_TRAINING_SIZE; ++i)
    {
        Mat descriptors;

		// construct filename for loop
		stringstream index;
		index << i;
		filename = str1 + index.str() + str2;
		cout << "read file: " << filename << endl;

		// read image
        Mat img = imread(filename, 0);
        if(img.empty())
        {
            cout << "failed to read image " << filename << endl;
            waitKey(1000);
            return -1;
        }
		imshow("cat", img);
		waitKey(1000);
		cout << "finish read jpg " << i << endl;
        extractor(img, mask, keypoints, descriptors);
		cout << "finish extract descriptor for jpg " << i << endl;
        training_descriptors.push_back(descriptors);
		cout << "finish push back descriptor" << endl;
    }


    //construct the vocabulary
    BOWKMeansTrainer bowtrainer(VOCABULARY_SIZE);
    bowtrainer.add(training_descriptors);
    Mat vocabulary = bowtrainer.cluster();
	cout << "finish construct vocabulary" << endl;

    //save vocabulary as xml file
    FileStorage cvfs("cat.xml", CV_STORAGE_WRITE);
    write(cvfs, "cat", vocabulary);
    return 0;

}
