#Michael Talaga's Final Project
#CSC 490 with Dr. Patterson

import random
import math

global beta, eta 

beta, eta = 7, 0.5
#test weights



hidden_layer = [0.1, -0.1, 0]
testW1 = [0, 0, 0, 0]
testW2 = [0, 0, 0, 0]
testWeights =  [testW1, testW2]


##########################
def randomWeights(weights):
  for each in weights:
    for i in range(len(each) - 1):
      each[i] = random.uniform(-0.5, 0.5)
  return weights

def randomHiddenWeights(weights):
  for i in range(len(weights) - 1):
      weights[i] = random.uniform(-0.5, 0.5)
  return weights


def parseData(fileD, boolSig):
  dataFile = open(fileD, "r")
  listData = []
  
  for i in dataFile:
    listData.append(i)
  lengthFile = len(listData)
  data = []
  
  counter2 = 0
  while counter2 < (lengthFile):
    dataRow = []
    stringCurrent = listData[counter2]
    if '\n' in stringCurrent:#taking out \n
      stringCurrent = stringCurrent.replace('\n', '')
    stringCurrent = stringCurrent.split(',')
    data.append(stringCurrent)
    counter2 += 1

  for each in data:
    for element in each:
      if boolSig == 1:
        element = int(element)
      if boolSig == 2:
        element = float(element)

  return data;

#sigmoid calculation
def sigmoid(h, beta):
  #Creates a y
  e = math.e
  denom = -(beta * h)
  calculation = 1 / (1 + e**(denom))
  return calculation
  

#h summation function
def summation(data, weights):
  h = 0.0
  for i in range(len(data) - 1):
    h += float(data[i]) * float(weights[i])
  #bias
  h -= float(data[len(data) - 1]) * float(weights[len(weights) - 1])
  return h

def delta(h, weight, y):
  y = float(y)
  weight = float(weight)
  h = float(h)
  delta = (beta * (y * (1 - y)) * h * weight)
  return delta

#hidden delta behaves differently than regular delta
def hDelta(y, target):
  delta = 0.0
  y = float(y)
  target = float(target)
  delta = beta * (y - target) * (y * (1 - y))
  return delta

#Weight change calculation
def weightChange(delta, y, weight):
  weight = float(weight)
  y = float(y)
  delta = float(delta)
  newWeight = weight - (eta * delta * y)
  return newWeight

#accuracy calculation (% difference)
def difference(value_one, value_two):
  #%error
  difference = abs(value_one - value_two)
  difference = difference * 100
  #difference = format(difference, '.2f')
  return difference

#########################

#Train function will complete an iteration given a data point. It will do h function summations, sigmoid calculations, delta functions, and weight changes
def train(data, tW, hiddenLayer):
  ys = []

  output = data[len(data) - 1]
  output = float(output)

  for i in range(len(tW)):
     #list of y values
    h = summation(data, tW[i])
    y = sigmoid(h, beta)
    ys.append(y)
  hidden_h = 0

  #Hidden node summation + sigmoid
  for i in range(len(hiddenLayer) - 1):
    hidden_h += (float(ys[i]) * float(hiddenLayer[i]))
  hidden_h += (float(hiddenLayer[len(hiddenLayer) - 1]) * -1.0)
  hiddenY = sigmoid(hidden_h, beta)
  ys.append(hiddenY)
  

  hidden_delta = hDelta(hiddenY, output)


  #hidden changes
  for i in range(len(ys) - 1):
    hiddenLayer[i] = weightChange(hidden_delta, ys[i], hiddenLayer[i])
  hiddenLayer[len(hiddenLayer) - 1] = weightChange(hidden_delta, -1, hiddenLayer[len(hiddenLayer) - 1])
  #print(hiddenLayer)

  #regular deltas (input layer)
  for i in range(len(hiddenLayer) - 1):
    input_delta = delta(hidden_delta, hiddenLayer[i], ys[i])
    #print("\nInput Delta: ", input_delta)

    for j in range(len(tW[i]) - 1):
      tW[i][j] = tW[i][j] - (eta * input_delta)
    #bias change
    tW[i][len(tW[i]) - 1]= tW[i][len(tW[i]) - 1] - (eta * input_delta) * -1 
  #print(tW)

  returnList = [hiddenLayer, tW]
  return returnList
  






def validate(inputs, weights, hidden_weights):

  #finding output or y^
  total = 0.0
  #ys is a list of sigmoid functions
  ys = []

  for i in range(len(weights)):
  
    #forward propigation
    h = summation(inputs, weights[i])
    y = sigmoid(h, beta)
    ys.append(y)

  for i in range(len(hidden_weights) - 1):
    total += hidden_weights[i] * ys[i]
    #print(total)

  # add bias
  hidden_summation = summation(ys, hidden_weights)
  yHat2 = sigmoid(hidden_summation, beta)

  return yHat2





def main():
  
  newWeights = randomWeights(testWeights)
  newHidden = randomHiddenWeights(hidden_layer)

  testData = parseData("CVX.data", 2)
  XOMdata = parseData("XOM.data", 2)
  BPdata = parseData("BP.data", 2)
  PSXdata = parseData("PSX.data", 2)
  testDataList = [testData, BPdata, PSXdata]
  
  

  #training set
  done = False
  counter = 0
  while not done:
    counter += 1
    for each in testDataList:
      for i in range(len(each) - 1):
        results = train(each[i], newWeights, newHidden)
        
    if counter == 5:
      done = True
  




  #Validate
  validateCounter = 0
  total_difference = 0
  
  #Validate for the XOM data
  for i in range(len(XOMdata)):
    original = float(XOMdata[i][3])
    inputs = [XOMdata[i][0], XOMdata[i][1], XOMdata[i][2]]
    validateTest = validate(inputs, newWeights, newHidden)
    #print(validateTest)
    diff = difference(validateTest, original)
    total_difference += diff
    #print("Difference:", diff,"%\n", "Original:", original, "\nGuess:", validateTest)
    validateCounter += 1
  #Average difference (Average % difference between guess and actual output)
  averageDifference = total_difference / validateCounter
  print("\nAverage % Difference between guess and actual output in validation set:", averageDifference)

  #Attribute Importance test
  print("")
  peRatio = validate([0.1, 0.0, 0.0], newWeights, newHidden)
  print(peRatio)
  currentRatio = validate([0.0, 0.1, 0.0], newWeights, newHidden)
  print(currentRatio)
  eps = validate([0.0, 0.0, 0.1], newWeights, newHidden)
  print(eps)
  print("\n\nExample:\n")
  exampleValue = validate([0.310, 0.040, 0.040], newWeights, newHidden)
  print(exampleValue)
  
main()