#include "common.h"
#include "main.h"

std::ofstream myfile;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QString strCont = "D:\\opencvimg\\Lenna.png";
    QString strSign = "D:\\opencvimg\\sign00003.png";
    QString strRes = "D:\\opencvimg\\imgDst.png";

    myfile.open ("D:\\Diplomchik\\fft_wm\\example.txt");
    //embedWmToFFT(strCont, strSign, strRes);
    //extractWmFromFFT(strRes);
    testImageRotation(strRes);
    //rotateImg(strRes);

    if(argc == 4)
    {
        QString fileCont = argv[1],
                fileWm   = argv[2],
                fileRes  = argv[3];
        //embedWmToFFT(fileCont, fileWm, fileRes);
        embedWmToDCT(fileCont, fileWm, fileRes);
    }
    else if(argc == 3)
    {
        QString fileCont = argv[1],
                fileMag  = argv[2];
        rotateImg(fileCont);
    }
    else if(argc == 2)
    {
        QString fileCont = argv[1];
        //extractWmFromFFT(fileCont);
        extractWmFromDCT(fileCont);
    }
    return a.exec();
}

void embedWmToFFT(const QString fileCont, const QString fileWm, QString fileRes)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1), imgDst32, imgDst,
            imgWm = cv::imread(fileWm.toStdString().c_str(), 0), imgWm32,
            imgMag, imgPhase;     // Ампльтуда спектра и Фаза спектра
    imgWm.convertTo(imgWm32, CV_32F, 1.0/255.0);

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);

    // Раскладываем изображение в спектр
    ForwardFFT_Mag_Phase(imgSrcCh[2], imgMag, imgPhase);
    cv::add(imgMag, imgWm32, imgMag);

    // Обратное преобразование
    InverseFFT_Mag_Phase(imgMag, imgPhase, imgDst32);

    double minVal, maxVal;
    cv::minMaxLoc(imgDst32, &minVal, &maxVal);       // минимальное и максимальное значение яркости
    imgDst32 *= maxVal;
    imgDst32.convertTo(imgDst, CV_8U, 255.0/(maxVal-minVal), -minVal);
    cv::normalize(imgDst, imgDst, 0, 255, cv::NORM_MINMAX);

    imgSrcCh[2] = imgDst.clone();
    cv::merge(imgSrcCh, imgDst);

    // Вывод результатов
    cv::Mat imgLogMag;
    imgLogMag.zeros(imgMag.rows, imgMag.cols, CV_32F);
    imgLogMag=(imgMag+1);
    cv::log(imgLogMag, imgLogMag);
    //---------------------------------------------------
    cv::imshow("Логарифм амплитуды", imgLogMag);
    cv::imshow("Фаза", imgPhase);
    cv::imwrite(fileRes.toStdString().c_str(), imgDst);
}

void extractWmFromFFT(const QString fileCont)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1),
            imgMag, imgPhase;

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);
    // Раскладываем изображение в спектр
    ForwardFFT_Mag_Phase(imgSrcCh[2], imgMag, imgPhase);

    cv::Mat imgLogMag;
    imgLogMag.zeros(imgMag.rows, imgMag.cols, CV_32F);
    imgLogMag = (imgMag+1);
    cv::log(imgLogMag, imgLogMag);
    cv::imshow("Логарифм амплитуды 1", imgLogMag);
}

cv::Mat extractWmFromFFT2(cv::Mat blueSrc)
{
    cv::Mat imgMag, imgPhase;
    // Раскладываем изображение в спектр
    ForwardFFT_Mag_Phase(blueSrc, imgMag, imgPhase);

    cv::Mat imgLogMag;
    imgLogMag.zeros(imgMag.rows, imgMag.cols, CV_32F);
    imgLogMag = (imgMag+1);
    cv::log(imgLogMag, imgLogMag);
    // imgLogMag.convertTo(imgLogMag,-1,10,0);

    cv::Rect myROI(0.78125*imgLogMag.cols, 0.78125*imgLogMag.rows, 0.21875*imgLogMag.cols, 0.21875*imgLogMag.rows);
    cv::Mat croppedImage = imgLogMag(myROI).clone();

    cv::normalize(croppedImage, croppedImage, 0, 255, cv::NORM_MINMAX);

    cv::threshold(croppedImage, croppedImage, 63, 255,CV_THRESH_BINARY);
    cv::imwrite("D:\\opencvimg\\forCrop3.png", croppedImage);
    croppedImage.convertTo(croppedImage,CV_8U);
    return croppedImage;

}

// QString fileCont="D:\\opencvimg\\forCrop2.png";
// croppedImage = cv::imread(fileCont.toStdString().c_str(), 0);


void embedWmToDCT(const QString fileCont, const QString fileWm, QString fileRes)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1), imgSrc32, imgDst32, imgDst,
            imgWm = cv::imread(fileWm.toStdString().c_str(), 0), imgWm32,
            imgMag32, imgMag;     // Ампльтуда спектра

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);

    /*imgSrc*/imgSrcCh[0].convertTo(imgSrc32, CV_32F, 1.0/255.0);
    imgWm.convertTo(imgWm32, CV_32F, 1.0/255.0);
    imgWm32 /= 5;

    // Раскладываем изображение в спектр
    cv::dct(imgSrc32, imgMag32, 0);
    cv::add(imgMag32, imgWm32, imgMag32);

    // Обратное DCT и собинаем в изображение
    cv::idct(imgMag32, imgDst32);
    imgDst32.convertTo(imgDst, CV_8U, 255.0, 0);
    cv::normalize(imgDst, imgDst, 0, 255, cv::NORM_MINMAX);
    cv::imwrite(fileRes.toStdString().c_str(), imgDst);

    double minVal, maxVal;
    minMaxLoc(imgMag32, &minVal, &maxVal);       // минимальное и максимальное значение яркости
    imgMag32 = imgMag32*maxVal;
    imgMag32.convertTo(imgMag, CV_8U, 255.0/(maxVal-minVal), -minVal);
    cv::normalize(imgMag, imgMag, 0, 255, cv::NORM_MINMAX);
    cv::imshow("Mag_DST8", imgMag);
}


void extractWmFromDCT(const QString fileCont)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1/*1??*/),
            imgSrc32, imgMag32, imgMag;     // Ампльтуда спектра

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);

    /*imgSrc*/imgSrcCh[0].convertTo(imgSrc32, CV_32F, 1.0/255.0);
    cv::dct(imgSrc32, imgMag32,0);

    double minVal, maxVal;
    minMaxLoc(imgMag32, &minVal, &maxVal);       // минимальное и максимальное значение яркости
    imgMag32 = imgMag32*maxVal;
    imgMag32.convertTo(imgMag, CV_8U, 255.0/(maxVal-minVal), -minVal);
    cv::normalize(imgMag, imgMag, 0, 255, cv::NORM_MINMAX);
    cv::imshow("img Mag", imgMag);
}


void rotateImg(const QString fileCont)
{
    cv::Mat src = cv::imread(fileCont.toStdString().c_str(), 1);
    std::vector <cv::Mat> imgSrcCh;
    cv::split(src, imgSrcCh);
    cv::Mat blueSrc = imgSrcCh[2].clone();
    cv::Mat dst;

    cv::Point2f pc(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pc, 75, 1.0);
    cv::warpAffine(blueSrc, dst, r, src.size());

    extractWmFromFFT2(dst);


    // cv::imshow("imgMagRes", dst);
}
// QString fileCont="D:\\opencvimg\\forCrop2.png";
//croppedImage = cv::imread(fileCont.toStdString().c_str(), 0);

void testImageRotation(const QString fileCont){
    QString idealMat="D:\\opencvimg\\forCompare.png";
    cv::Mat idealSignMat = cv::imread(idealMat.toStdString().c_str(), 0);
    cv::threshold(idealSignMat, idealSignMat, 127, 255,CV_THRESH_BINARY);
    int successResult = 0;
    cv::Mat src = cv::imread(fileCont.toStdString().c_str(), 1),firstRotatedMat;
    std::vector <cv::Mat> imgSrcCh;
    cv::split(src, imgSrcCh);
    src = imgSrcCh[2];
    //simulate first rotate
    for(int firstRotationAngle=0;firstRotationAngle<361;firstRotationAngle++)
    {
     //   double firstRotationAngle = 150;
        cv::Point2f pc(src.cols/2., src.rows/2.);
        cv::Mat r = cv::getRotationMatrix2D(pc, firstRotationAngle, 1.0);
        cv::warpAffine(src, firstRotatedMat, r, src.size());

        int rotationAngleResult = getAngleValue(firstRotatedMat,idealSignMat);
        if (firstRotationAngle==rotationAngleResult){
            successResult++;
            std::cout <<"For "<< firstRotationAngle<< " degree result is OK! \n";
            myfile <<"For "<< firstRotationAngle<< " degree result is OK! \n";
        } else {
            std::cout <<"For "<< firstRotationAngle<< " degree result is WRONG! \n";
            myfile <<"For "<< firstRotationAngle<< " degree result is WRONG! \n";
        }
    }
    std::cout <<"Percentage of true decision is "<< successResult/360<<"\n";
    myfile <<"Percentage of true decision is "<< successResult/360<<"\n";
}

int getAngleValue(cv::Mat src, cv::Mat idealSign){
    cv::Mat dst;
    cv::Point2f pc(src.cols/2., src.rows/2.);
    int resultAngle=1000;

    for (int angle = 0; angle<361;angle++)
    {

        cv::Mat r = cv::getRotationMatrix2D(pc, -angle, 1.0);
        cv::warpAffine(src, dst, r, src.size());

        dst = extractWmFromFFT2(dst);

        //compare dst and idealSign
        double allPixels = dst.rows*dst.cols;
        double samePixels = 0;
        double percentageResult = 0;

        // cv::imshow("lala",dst);
        // cv::imshow("ideal",idealSign);
        if(idealSign.rows==dst.rows && idealSign.cols==dst.cols){
            for(int x = 0; x<dst.cols;x++){
                for(int y = 0; y<dst.rows;y++){
                    if(dst.at<char>(x,y)==idealSign.at<char>(x,y)){
                        samePixels++;
                    }
                }
            }
            percentageResult = samePixels/allPixels;
    //        std::cout<<"same pixels "<<samePixels<<"\n";
        }
        double compareValue = 0.90;
        if(percentageResult>compareValue){
            std::cout <<"Answer is "<< angle<< " degree and "<<100*percentageResult<<" percent. ";
            myfile <<"Answer is "<< angle<< " degree and "<<100*percentageResult<<" percent. ";
            resultAngle=angle;
            break;
        }
    }
    return resultAngle;
}
