import cv2
import numpy as np
import utils

path = "or1.png"
img = cv2.imread(path)
heightImg = 1000
widthImg = 800
questions = 40
choices = 5
ansturkce = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
             0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 4]
anssosyal = [4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0,
             4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0]
ansmat = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3, 3, 4, 0,
          1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
ansfen = [4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0,
          4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0]


#Ön İşleme
img = cv2.resize(img, (widthImg, heightImg))
imgFinal = img.copy()
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

rectCon = utils.rectContour(contours)
cv2.imshow("Sınırları Görüntüleme", imgContours)

#Derslere Yönelik Sınırları Belirleme
turkcesinir = utils.getCornerPoints(rectCon[2])
cv2.drawContours(imgBiggestContours, turkcesinir, -1, (0, 255, 0), 20)#YESIL
turkcedegerlendirme = utils.getCornerPoints(rectCon[10])
cv2.drawContours(imgBiggestContours, turkcedegerlendirme, -1, (0, 255, 0), 20)#YESIL

fensinir = utils.getCornerPoints(rectCon[0])
cv2.drawContours(imgBiggestContours, fensinir, -1, (0, 0, 255), 20)#KIRMIZI
fendegerlendirme = utils.getCornerPoints(rectCon[8])
cv2.drawContours(imgBiggestContours, fendegerlendirme, -1, (0, 0, 255), 20)#KIRMIZI

matsinir = utils.getCornerPoints(rectCon[3])
cv2.drawContours(imgBiggestContours, matsinir, -1, (102, 0, 153), 20)#MOR
matdegerlendirme = utils.getCornerPoints(rectCon[11])
cv2.drawContours(imgBiggestContours, matdegerlendirme, -1, (102, 0, 153), 20)#MOR

sosyalsinir = utils.getCornerPoints(rectCon[1])
cv2.drawContours(imgBiggestContours, sosyalsinir, -1, (255, 203, 219), 20) #GRİ
sosyaldegerlendirme = utils.getCornerPoints(rectCon[9])
cv2.drawContours(imgBiggestContours, sosyaldegerlendirme, -1, (255, 203, 219), 20) #GRİ

'''
Basliklar (Gerekli Değiller)
biggestContour_5 = utils.getCornerPoints(rectCon[4])#sosyalbilimlerbaslik
biggestContour_6 = utils.getCornerPoints(rectCon[5])#turkcebaslik
biggestContour_7 = utils.getCornerPoints(rectCon[6])#matematikbaslik
cv2.drawContours(imgBiggestContours, biggestContour_5, -1, (213, 255, 0), 20)#sosyalbilimlerbaslik
cv2.drawContours(imgBiggestContours, biggestContour_6, -1, (213, 255, 0), 20)#turkcebaslik
cv2.drawContours(imgBiggestContours, biggestContour_7, -1, (213, 255, 0), 20)#matematikbaslik
'''
cv2.imshow("Contour Deneme", imgBiggestContours)

#Turkce Puan Yazma Kısmı
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, turkcedegerlendirme, -1, (0, 255, 0), 20)
turkcedegerlendirme = utils.reorder(turkcedegerlendirme)
pt3 = np.float32(turkcedegerlendirme)
pt4 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
matrix2 = cv2.getPerspectiveTransform(pt3, pt4)
TurkceDegerlendime = cv2.warpPerspective(img, matrix2, (325, 150))
#cv2.imshow("TurkceDegerlendirme", TurkceDegerlendime)

#Sosyal Bilimler Puan Yazma Kısmı
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, sosyaldegerlendirme, -1, (0, 255, 0), 20)
sosyaldegerlendirme = utils.reorder(sosyaldegerlendirme)
pt5 = np.float32(sosyaldegerlendirme)
pt6 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
matrix3 = cv2.getPerspectiveTransform(pt5, pt6)
SosyalDegerlendirme = cv2.warpPerspective(img, matrix3, (325, 150))
#cv2.imshow("SosyalDegerlendirme", SosyalDegerlendirme)

#Matematik Puan Yazma Kısmı
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, matdegerlendirme, -1, (0, 255, 0), 20)
matdegerlendirme = utils.reorder(matdegerlendirme)
pt7 = np.float32(matdegerlendirme)
pt8 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
matrix4 = cv2.getPerspectiveTransform(pt7, pt8)
MatDegerlendirme = cv2.warpPerspective(img, matrix4, (325, 150))
#cv2.imshow("MatDegerlendirme", MatDegerlendirme)

#Fen Puan Yazma Kısmı
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, fendegerlendirme, -1, (0, 255, 0), 20)
fendegerlendirme = utils.reorder(fendegerlendirme)
pt9 = np.float32(fendegerlendirme)
pt10 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
matrix5 = cv2.getPerspectiveTransform(pt9, pt10)
FenDegerlendirme = cv2.warpPerspective(img, matrix5, (325, 150))
#cv2.imshow("FenDegerlendirme", FenDegerlendirme)


#Turkce Icin Degerlendirme
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, turkcesinir, -1, (0, 255, 0), 20)
turkcesinir = utils.reorder(turkcesinir)
pt1 = np.float32(turkcesinir)
pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix1 = cv2.getPerspectiveTransform(pt1, pt2)
TurkceCvp = cv2.warpPerspective(img, matrix1, (widthImg, heightImg))
#cv2.imshow("TurkceCvp", TurkceCvp)


TurkceCvpGray = cv2.cvtColor(TurkceCvp, cv2.COLOR_BGR2GRAY)
TurkceCvpThresh = cv2.threshold(TurkceCvpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("TurkceCvpThresh", TurkceCvpThresh)
boxes0 = utils.splitBoxes(TurkceCvpThresh)
#cv2.imshow("test", boxes0[0])

#print(cv2.countNonZero(boxes0[0]), cv2.countNonZero(boxes0[1]))

#Her Kutunun Sıfır Olmayan Değerlerini Alma
myPixelVal0 = np.zeros((questions, choices)) #Questions,Choices
countC = 0
countR = 0
for image in boxes0:
    totalPixels0 = cv2.countNonZero(image)
    myPixelVal0[countR][countC] = totalPixels0
    countC +=1
    if(countC == choices):countR +=1 ;countC=0
#print(myPixelVal0)

#İşaretlemelerin İndex Değerlerini Bulma
myIndex0 = []
for x in range (0,questions):
    arr = myPixelVal0[x]
    #print("arr",arr)
    myIndexVal0 = np.where(arr == np.amax(arr))
    #print(myIndexVal[0])
    myIndex0.append(myIndexVal0[0][0])
print(myIndex0)

#Değerlendirme
grading0 = []
for x in range (0,questions):
    if ansturkce[x] == myIndex0[x]:
        grading0.append(1)
    else: grading0.append(0)
#print(grading)
scoreturkce = (sum(grading0)/questions) *100 #Final Grade
#print(scoreturkce)
cv2.putText(TurkceDegerlendime, str(float(scoreturkce)), (75,95), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
cv2.imshow("TurkcePuan", TurkceDegerlendime)

#Sosyal Bilimler Icin Degerlendirme
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, sosyalsinir, -1, (255, 203, 219), 20)
sosyalsinir = utils.reorder(sosyalsinir)
pt11 = np.float32(sosyalsinir)
pt12 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix6 = cv2.getPerspectiveTransform(pt11, pt12)
SosyalCvp = cv2.warpPerspective(img, matrix6, (widthImg, heightImg))
#cv2.imshow("SosyalCvp", SosyalCvp)
SosyalCvpGray = cv2.cvtColor(SosyalCvp, cv2.COLOR_BGR2GRAY)
SosyalCvpThresh = cv2.threshold(SosyalCvpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("SosyalCvpThresh", SosyalCvpThresh)
boxes1 = utils.splitBoxes(SosyalCvpThresh)
#cv2.imshow("test",boxes1[0])
#print(cv2.countNonZero(boxes1[0]), cv2.countNonZero(boxes1[1]))
#Getting NonZero Values Of Each Box
myPixelVal1 = np.zeros((questions, choices)) #Questions,Choices
countC = 0
countR = 0
for image in boxes1:
    totalPixels1 = cv2.countNonZero(image)
    myPixelVal1[countR][countC] = totalPixels1
    countC +=1
    if(countC == choices):countR +=1 ;countC=0
#print(myPixelVal1)
#FINDING INDEX VALUES OF THE MARKINGS
myIndex1 = []
for x in range (0,questions):
    arr = myPixelVal1[x]
    #print("arr",arr)
    myIndexVal1 = np.where(arr == np.amax(arr))
    #print(myIndexVal[0])
    myIndex1.append(myIndexVal1[0][0])
#print(myIndex1)
#GRADING
grading1 = []
for x in range (0,questions):
    if anssosyal[x] == myIndex1[x]:
        grading1.append(1)
    else: grading1.append(0)
#print(grading)
scoresosyal = (sum(grading1)/questions) *100 #Final Grade
#print(scoresosyal)
cv2.putText(SosyalDegerlendirme, str(float(scoresosyal)), (40,95), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
cv2.imshow("SosyalPuan", SosyalDegerlendirme)

#Matematik Icin Degerlendirme
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, matsinir, -1, (102, 0, 153), 20)#MOR
matsinir = utils.reorder(matsinir)
pt13 = np.float32(matsinir)
pt14 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix7 = cv2.getPerspectiveTransform(pt13, pt14)
MatCvp = cv2.warpPerspective(img, matrix7, (widthImg, heightImg))
#cv2.imshow("MatCvp", MatCvp)
MatCvpGray = cv2.cvtColor(MatCvp, cv2.COLOR_BGR2GRAY)
MatCvpThresh = cv2.threshold(MatCvpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("MatCvpThresh", MatCvpThresh)
boxes2 = utils.splitBoxes(MatCvpThresh)
#cv2.imshow("test",boxes2[0])
#print(cv2.countNonZero(boxes2[0]), cv2.countNonZero(boxes2[1]))
#Getting NonZero Values Of Each Box
myPixelVal2 = np.zeros((questions, choices)) #Questions,Choices
countC = 0
countR = 0
for image in boxes2:
    totalPixels2 = cv2.countNonZero(image)
    myPixelVal2[countR][countC] = totalPixels2
    countC +=1
    if(countC == choices):countR +=1 ;countC=0
#print(myPixelVal2)
#FINDING INDEX VALUES OF THE MARKINGS
myIndex2 = []
for x in range (0,questions):
    arr = myPixelVal2[x]
    #print("arr",arr)
    myIndexVal2 = np.where(arr == np.amax(arr))
    #print(myIndexVal[0])
    myIndex2.append(myIndexVal2[0][0])
#print(myIndex2)
#GRADING
grading2 = []
for x in range (0,questions):
    if ansmat[x] == myIndex2[x]:
        grading2.append(1)
    else: grading2.append(0)
#print(grading2)
scoremat = (sum(grading2)/questions) *100 #Final Grade
#print(scoremat)
cv2.putText(MatDegerlendirme, str(float(scoremat)), (75,95), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
cv2.imshow("MatPuan", MatDegerlendirme)

#Fen Bilimleri İçin Degerlendirme
imgBiggestContours = img.copy()
cv2.drawContours(imgBiggestContours, fensinir, -1, (102, 0, 153), 20)#MOR
fensinir = utils.reorder(fensinir)

pt15 = np.float32(fensinir)
pt16 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix8 = cv2.getPerspectiveTransform(pt15, pt16)
FenCvp = cv2.warpPerspective(img, matrix8, (widthImg, heightImg))
#cv2.imshow("FenCvp", FenCvp)

FenCvpGray = cv2.cvtColor(FenCvp, cv2.COLOR_BGR2GRAY)
FenCvpThresh = cv2.threshold(FenCvpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("FenCvpThresh", FenCvpThresh)
boxes3 = utils.splitBoxes(FenCvpThresh)
#cv2.imshow("test",boxes3[0])
#print(cv2.countNonZero(boxes3[0]), cv2.countNonZero(boxes3[1]))
#Getting NonZero Values Of Each Box
myPixelVal3 = np.zeros((questions, choices)) #Questions,Choices
countC = 0
countR = 0
for image in boxes3:
    totalPixels3 = cv2.countNonZero(image)
    myPixelVal3[countR][countC] = totalPixels3
    countC +=1
    if(countC == choices):countR +=1 ;countC=0
#print(myPixelVal3)
#FINDING INDEX VALUES OF THE MARKINGS
myIndex3 = []
for x in range (0,questions):
    arr = myPixelVal3[x]
    #print("arr",arr)
    myIndexVal3 = np.where(arr == np.amax(arr))
    #print(myIndexVal[0])
    myIndex3.append(myIndexVal3[0][0])
#print(myIndex3)
#GRADING
grading3 = []
for x in range (0,questions):
    if ansfen[x] == myIndex3[x]:
        grading3.append(1)
    else: grading3.append(0)
#print(grading3)
scorefen = (sum(grading3)/questions) *100 #Final Grade
#print(scorefen)
cv2.putText(FenDegerlendirme, str(float(scorefen)), (45,95), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
cv2.imshow("FenPuan", FenDegerlendirme)

cv2.putText(img, str(float(scorefen)), (468,975), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
cv2.putText(img, str(float(scoremat)), (348,975), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
cv2.putText(img, str(float(scoresosyal)), (218,975), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
cv2.putText(img, str(float(scoreturkce)), (98,975), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imshow("Final Sonuc",img)
cv2.imwrite('DegerlendirmeSonuc.jpg', img)



cv2.waitKey(0)

##atakan.sn##
