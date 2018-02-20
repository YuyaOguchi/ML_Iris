import math
import numpy


#get all data and separate them into diff d-array
# Setosa 0-49       Training: 0-39    Test: 40-49
# Versicolor 50-99  Training: 50-89   Test: 90-99
# Virginica 100.0-149 Training 100.0-139  Test: 140-149

filein = open("iris.txt","r").read().split("\n")
eachinput = [temp.split(",")[:4] for temp in filein]
sets = numpy.array(eachinput, dtype=float)

#each set of flowers
Setosa=sets[0:49]
Versicolor=sets[50:99]
Virginica=sets[100.0:149]

#print each flower
#for flower in Setosa:
  #print(flower)

#for flower in Versicolor:
  #print(flower)

#for flower in Virginica:
  #print(flower)

#Mean of each column for each flower
MeanSetosa=numpy.mean(Setosa[0:39], axis=0, dtype=numpy.float64)
MeanVersicolor=numpy.mean(Versicolor[0:39], axis=0, dtype=numpy.float64)
MeanVirginica=numpy.mean(Virginica[0:39], axis=0, dtype=numpy.float64)

#print mean
print("Mean values:")
print(MeanSetosa)
print(MeanVersicolor)
print(MeanVirginica)


#Covariance
# sum((x-xavg)(y-yavg))/n-1
#Cov method didn't work, sticking to regular
SetosaCov = numpy.cov(Setosa[0:39])
VersicolorCov = numpy.cov(Versicolor[0:39])
VirginicaCov = numpy.cov(Virginica[0:39])

#printer
print("\n")
print(SetosaCov)
print(VersicolorCov)
print(VirginicaCov)


#Range -1 to 1
SetosaCov2 =0;
for x in Setosa[0:39]:
  SetosaCov2 = SetosaCov2 + numpy.outer(x-MeanSetosa, x-MeanSetosa)
SetosaCov2 = SetosaCov2/39


VersicolorCov2 =0;
for x in Versicolor[0:39]:
  VersicolorCov2 = VersicolorCov2 + numpy.outer(x-MeanVersicolor, x-MeanVersicolor)
VersicolorCov2 = VersicolorCov2/39


VirginicaCov2 =0;
for x in Virginica[0:39]:
  VirginicaCov2 = VirginicaCov2 + numpy.outer(x-MeanVirginica, x-MeanVirginica)
VirginicaCov2 = VirginicaCov2/39

totalCov = (SetosaCov2+VersicolorCov2+VirginicaCov2)/3

print("\n")
print("Total cov and cov2s")
print(totalCov)
print(SetosaCov2)
print(VersicolorCov2)
print(VirginicaCov2)



#Assume cov is same for all classes
#Just predict input ones against current cov and mean vals
def LDA(Data, classes):
  error=0
  for item in Data:
    SetosaMatch = numpy.inner(numpy.inner(item - MeanSetosa, numpy.linalg.inv(totalCov)), item - MeanSetosa)* -1/2
    VersicolorMatch=numpy.inner(numpy.inner(item - MeanVersicolor, numpy.linalg.inv(totalCov)), item - MeanVersicolor)* -1/2
    VirginicaMatch=numpy.inner(numpy.inner(item - MeanVirginica, numpy.linalg.inv(totalCov)), item - MeanVirginica)* -1/2

    #print("Printing Match value...........")
    #print(SetosaMatch)
    #print(VersicolorMatch)
    #print(VirginicaMatch)

    if (SetosaMatch > VersicolorMatch):
      if (SetosaMatch > VirginicaMatch):
        if (classes != "Setosa"):
          error+=1
    elif (VersicolorMatch > SetosaMatch):
      if (VersicolorMatch > VirginicaMatch):
        if (classes != "Versicolor"):
          error+=1
    elif (VirginicaMatch > SetosaMatch):
      if (VirginicaMatch > VersicolorMatch):
        if (classes != "Virginica"):
          error+=1
  return error

#Can't assume Cov is same as all others, use its own
def QDA(Data, classes):
  error=0
  for item in Data:
    SetosaMatch=math.log(1/(pow(numpy.linalg.det(SetosaCov2),1/2))) + (-1/2)*numpy.inner(numpy.inner(item - MeanSetosa, numpy.linalg.inv(SetosaCov2)), item-MeanSetosa)
    VersicolorMatch=math.log(1/(pow(numpy.linalg.det(VersicolorCov2),1/2))) + (-1/2)*numpy.inner(numpy.inner(item - MeanVersicolor, numpy.linalg.inv(VersicolorCov2)), item-MeanVersicolor)
    VirginicaMatch=math.log(1/(pow(numpy.linalg.det(VirginicaCov2),1/2))) + (-1/2)*numpy.inner(numpy.inner(item - MeanVirginica, numpy.linalg.inv(VirginicaCov2)), item-MeanVirginica)

    #print("Printing Match value...........")
    #print(SetosaMatch)
    #print(VersicolorMatch)
    #print(VirginicaMatch)

    if (SetosaMatch > VersicolorMatch):
      if (SetosaMatch > VirginicaMatch):
        if (classes != "Setosa"):
          error+=1
    elif (VersicolorMatch > SetosaMatch):
      if (VersicolorMatch > VirginicaMatch):
        if (classes != "Versicolor"):
          error+=1
    elif (VirginicaMatch > SetosaMatch):
      if (VirginicaMatch > VersicolorMatch):
        if (classes != "Virginica"):
          error+=1
  return error


#Part1-4
#errors for training set
#For float calc
ErrorLDA = 1.0
ErrorLDA = (LDA(Setosa[0:39],"Setosa") + LDA(Versicolor[0:39],"Versicolor") + LDA(Virginica[0:39],"Virginica"))/(40.0*3) *100.0
print "LDA Training",
print ErrorLDA,
print "%error"
print LDA(Setosa[0:39],"Setosa"),
print LDA(Versicolor[0:39],"Versicolor"),
print LDA(Virginica[0:39],"Virginica")
#errors for training set
#For float calc
ErrorQDA = 1.0
ErrorQDA = (QDA(Setosa[0:39],"Setosa") + QDA(Versicolor[0:39],"Versicolor") + QDA(Virginica[0:39],"Virginica"))/(40.0*3) *100.0
print "QDA Training",
print ErrorQDA,
print "%error"
print QDA(Setosa[0:39],"Setosa"),
print QDA(Versicolor[0:39],"Versicolor"),
print QDA(Virginica[0:39],"Virginica")


#errors for test set
#For float calc
ErrorLDA = 1.0
ErrorLDA = (LDA(Setosa[40:49],"Setosa") + LDA(Versicolor[40:49],"Versicolor") + LDA(Virginica[40:49],"Virginica"))/(10.0*3) *100.0
print "LDA Test",
print ErrorLDA,
print "%error"
print LDA(Setosa[40:49],"Setosa"),
print LDA(Versicolor[40:49],"Versicolor"),
print LDA(Virginica[40:49],"Virginica")
#errors for test set
#For float calc
ErrorQDA = 1.0
ErrorQDA = (QDA(Setosa[40:49],"Setosa") + QDA(Versicolor[40:49],"Versicolor") + QDA(Virginica[40:49],"Virginica"))/(10.0*3) *100.0
print "QDA Test",
print ErrorQDA,
print "%error"
print QDA(Setosa[40:49],"Setosa"),
print QDA(Versicolor[40:49],"Versicolor"),
print QDA(Virginica[40:49],"Virginica")


#Part6
#Diagonalize Matrix
#Sigma = Cov
NewtotalCov = numpy.diag(numpy.diag(totalCov))
SetosaCov2 = numpy.diag(numpy.diag(SetosaCov2))
VirginicaCov2 = numpy.diag(numpy.diag(VirginicaCov2))
VersicolorCov2 = numpy.diag(numpy.diag(VersicolorCov2))
print("")
print("MATRIX=====================")
#errors for training set
#For float calc
ErrorLDA = 1.0
ErrorLDA = (LDA(Setosa[0:39],"Setosa") + LDA(Versicolor[0:39],"Versicolor") + LDA(Virginica[0:39],"Virginica"))/(40.0*3) *100.0
print "LDA Training",
print ErrorLDA,
print "%error"
print LDA(Setosa[0:39],"Setosa"),
print LDA(Versicolor[0:39],"Versicolor"),
print LDA(Virginica[0:39],"Virginica")
#errors for training set
#For float calc
ErrorQDA = 1.0
ErrorQDA = (QDA(Setosa[0:39],"Setosa") + QDA(Versicolor[0:39],"Versicolor") + QDA(Virginica[0:39],"Virginica"))/(40.0*3) *100.0
print "QDA Training",
print ErrorQDA,
print "%error"
print QDA(Setosa[0:39],"Setosa"),
print QDA(Versicolor[0:39],"Versicolor"),
print QDA(Virginica[0:39],"Virginica")


#errors for test set
#For float calc
ErrorLDA = 1.0
ErrorLDA = (LDA(Setosa[40:49],"Setosa") + LDA(Versicolor[40:49],"Versicolor") + LDA(Virginica[40:49],"Virginica"))/(10.0*3) *100.0
print "LDA Test",
print ErrorLDA,
print "%error"
print LDA(Setosa[40:49],"Setosa"),
print LDA(Versicolor[40:49],"Versicolor"),
print LDA(Virginica[40:49],"Virginica")
#errors for test set
#For float calc
ErrorQDA = 1.0
ErrorQDA = (QDA(Setosa[40:49],"Setosa") + QDA(Versicolor[40:49],"Versicolor") + QDA(Virginica[40:49],"Virginica"))/(10.0*3) *100.0
print "QDA Test",
print ErrorQDA,
print "%error"
print QDA(Setosa[40:49],"Setosa"),
print QDA(Versicolor[40:49],"Versicolor"),
print QDA(Virginica[40:49],"Virginica")

#Part 5
#Delete a col and retry the whole thing
for y in range(4):
  #Delete 1 col and make it new list
  # Del( list, col, rowORcol )
  SetosaTemp = numpy.delete(Setosa, y, 1)
  VersicolorTemp = numpy.delete(Versicolor, y, 1)
  VirginicaTemp = numpy.delete(Virginica, y, 1)

  #New Mean
  #Mean of each column for each flower
  MeanSetosa=numpy.mean(SetosaTemp[0:39], axis=0, dtype=numpy.float64)
  MeanVersicolor=numpy.mean(VersicolorTemp[0:39], axis=0, dtype=numpy.float64)
  MeanVirginica=numpy.mean(VirginicaTemp[0:39], axis=0, dtype=numpy.float64)

  #Recalc Cov & Total
  #Cov var must be the same to keep consistent matrix operation
  #Range -1 to 1
  SetosaCov2 =0;
  for x in SetosaTemp[0:39]:
    SetosaCov2 = SetosaCov2 + numpy.outer(x-MeanSetosa, x-MeanSetosa)
  SetosaCov2 = SetosaCov2/39


  VersicolorCov2 =0;
  for x in VersicolorTemp[0:39]:
    VersicolorCov2 = VersicolorCov2 + numpy.outer(x-MeanVersicolor, x-MeanVersicolor)
  VersicolorCov2 = VersicolorCov2/39


  VirginicaCov2 =0;
  for x in VirginicaTemp[0:39]:
    VirginicaCov2 = VirginicaCov2 + numpy.outer(x-MeanVirginica, x-MeanVirginica)
  VirginicaCov2 = VirginicaCov2/39

  totalCov = (SetosaCov2+VersicolorCov2+VirginicaCov2)/3

  print("")
  if (y==0):
    print("Sepal Length del")
  elif (y==1):
    print("Sepal Width del")
  elif (y==2):
    print("Petal Length del")
  elif (y==3):
    print("Petal Width del")

  #errors for training set
  #For float calc
  ErrorLDA = 1.0
  ErrorLDA = (LDA(SetosaTemp[0:39],"Setosa") + LDA(VersicolorTemp[0:39],"Versicolor") + LDA(VirginicaTemp[0:39],"Virginica"))/(40.0*3) *100.0
  print "LDA Training",
  print ErrorLDA,
  print "%error"
  print LDA(SetosaTemp[0:39],"Setosa"),
  print LDA(VersicolorTemp[0:39],"Versicolor"),
  print LDA(VirginicaTemp[0:39],"Virginica")
  #errors for training set
  #For float calc
  ErrorQDA = 1.0
  ErrorQDA = (QDA(SetosaTemp[0:39],"Setosa") + QDA(VersicolorTemp[0:39],"Versicolor") + QDA(VirginicaTemp[0:39],"Virginica"))/(40.0*3) *100.0
  print "QDA Training",
  print ErrorQDA,
  print "%error"
  print QDA(SetosaTemp[0:39],"Setosa"),
  print QDA(VersicolorTemp[0:39],"Versicolor"),
  print QDA(VirginicaTemp[0:39],"Virginica")


  #errors for test set
  #For float calc
  ErrorLDA = 1.0
  ErrorLDA = (LDA(SetosaTemp[40:49],"Setosa") + LDA(VersicolorTemp[40:49],"Versicolor") + LDA(VirginicaTemp[40:49],"Virginica"))/(10.0*3) *100.0
  print "LDA Test",
  print ErrorLDA,
  print "%error"
  print LDA(SetosaTemp[40:49],"Setosa"),
  print LDA(VersicolorTemp[40:49],"Versicolor"),
  print LDA(VirginicaTemp[40:49],"Virginica")
  #errors for test set
  #For float calc
  ErrorQDA = 1.0
  ErrorQDA = (QDA(SetosaTemp[40:49],"Setosa") + QDA(VersicolorTemp[40:49],"Versicolor") + QDA(VirginicaTemp[40:49],"Virginica"))/(10.0*3) *100.0
  print "QDA Test",
  print ErrorQDA,
  print "%error"
  print QDA(SetosaTemp[40:49],"Setosa"),
  print QDA(VersicolorTemp[40:49],"Versicolor"),
  print QDA(VirginicaTemp[40:49],"Virginica")
