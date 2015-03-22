require "nn"
require "hdf5"
require "image"
require "optim"

torch.setdefaulttensortype('torch.FloatTensor')

local f = hdf5.open('gameData.hdf5', 'r')
local x = f:read('/data/x'):all():float()
local y = f:read('/data/y'):all():float()
f:close()

--[[local y = torch.Tensor(yRaw:size(1), classes)
local elem = 1
local expand = function(label)
	local vector = torch.zeros(classes)
	vector[label+1] = 1
	y[elem] = vector
	elem = elem + 1
end
yRaw:apply(expand)]]--

x = x[{{},{1}}]
--print(x:size())
local width = x:size(4)
local height = x:size(3)
local channels = x:size(2)
local dataSize = x:size(1)

for c = 1, channels do
	d = x[{{}, {c}, {}}]
	--Scaling and mean substraction
	var = d:max()/2
	x[{{}, {c}, {}}] = (d - var)/var
end

local classes = y:size(2)


local trainSize = math.ceil(dataSize*0.9)
trainX = x[{{1, trainSize}, {}}]
trainY = y[{{1, trainSize}}]


local testSize = dataSize - trainSize
testX = x[{{trainSize, dataSize}, {}}]
testY = y[{{trainSize, dataSize}}]

outCh = 2
local convNet = nn.Sequential()


convNet:add(nn.SpatialZeroPadding(1, 1, 2, 3))
convNet:add(nn.SpatialConvolution(channels, outCh, 3, 6))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(4, 1, 4, 1))

convNet:add(nn.SpatialZeroPadding(1, 1, 2, 3))
convNet:add(nn.SpatialConvolution(outCh, outCh*2, 3, 6))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(3, 1, 3, 1))

convNet:add(nn.SpatialZeroPadding(1, 1, 2, 3))
convNet:add(nn.SpatialConvolution(outCh*2, outCh*4, 3, 6))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(2, 1, 2, 1))

convNet:add(nn.SpatialZeroPadding(1, 1, 2, 3))
convNet:add(nn.SpatialConvolution(outCh*4, outCh*8, 3, 6))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(2, 1, 2, 1))

convNet:add(nn.SpatialZeroPadding(1, 1, 2, 3))
convNet:add(nn.SpatialConvolution(outCh*8, outCh*8, 3, 6))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(2, 6, 2, 1))



reshape = outCh*8--150*6*channels
convNet:add(nn.Reshape(reshape))
convNet:add(nn.Linear(reshape, reshape))
convNet:add(nn.ReLU())
convNet:add(nn.Dropout())

convNet:add(nn.Linear(reshape, classes))
convNet:add(nn.Tanh())
--convNet:add(nn.LogSoftMax())

local criterion = nn.MSECriterion()


local counter = 0
local batchSize = 250
local epochs = 15
local iterations = epochs * math.ceil(trainSize / batchSize)

local optimState = {
  learningRate = 1e-2
}
local optimMethod = optim.adagrad

local parameters, gradParameters = convNet:getParameters()

local feval = function (params)
	if params ~= parameters then
    parameters:copy(params)
  end

	local startIndex = counter * batchSize + 1
  local endIndex = math.min(trainSize, (counter + 1) * batchSize + 1)
  if endIndex == trainSize then
    counter = 0
  else
    counter = counter + 1
  end

  local xBatch = trainX[{{startIndex, endIndex}, {}}]
  local yBatch = trainY[{{startIndex, endIndex}}]
  gradParameters:zero()

  local outputs = convNet:forward(xBatch)
	local loss = criterion:forward(outputs, yBatch)
	local dloss = criterion:backward(outputs, yBatch)
	convNet:backward(xBatch, dloss)

	return loss, gradParameters
end


local maxLoops = 1
local sum = 0
local SumIfZero = function(val)
  if val == 0 then
  	sum = sum + 1
  end
  return sum
end

local getError = function()
	local trainLog = convNet:forward(trainX)
	local trainClassProbs = torch.exp(trainLog)
	local _, trainPredictions = torch.max(trainClassProbs, 2)
	local _, trainYMax = torch.max(trainY, 2)


	local testLog = convNet:forward(testX)
	local testClassProbs = torch.exp(testLog)
	local _, testPredictions = torch.max(testClassProbs, 2)
	local _, testYMax = torch.max(testY, 2)


	sum = 0
	--print(trainLog:size(), trainClassProbs:size(), trainY:size())
	(trainPredictions - trainYMax):apply(SumIfZero)
	trainError = 1 - sum/trainPredictions:size(1)

	sum = 0
	(testPredictions - testYMax):apply(SumIfZero)
	testError = 1 - sum/testPredictions:size(1)
	return trainError, testError
end

local printError = function()
	local trainLoss = criterion:forward(convNet:forward(trainX), trainY)
	local testLoss = criterion:forward(convNet:forward(testX), testY)
	print(string.format("Train MSE loss: %6f", trainLoss))
	print(string.format("Test MSE loss: %6f", testLoss))
end

while true do
	for i = 1, iterations do
		local _, minibatchLoss = optimMethod(feval, parameters, optimState)
		--print(string.format("Minibatches Processed: %4d, loss = %6f", i,  minibatchLoss[1]))
	end

	
	printError()

	local trainError, testError = getError()
	print(string.format("Train classification error: %6f%", trainError))
	print(string.format("Test classification error:  %6f%", testError))

	maxLoops = maxLoops - 1
	if maxLoops == 0 or trainError + testError < 0.05 then
		break
	end
end






