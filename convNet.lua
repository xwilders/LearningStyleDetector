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

local width = 40
local height = 6
local channels = 1

for c = 1, channels do
	d = x[{{}, {c}, {}}]
	x[{{}, {c}, {}}] = d/d:max()
end
--print(x[1])

local classes = y:max()

local dataSize = x:size(1)

local trainSize = math.ceil(dataSize*0.9)
trainX = x[{{1, trainSize}, {}}]
trainY = y[{{1, trainSize}}]

local testSize = dataSize - trainSize
testX = x[{{trainSize, dataSize}, {}}]
testY = y[{{trainSize, dataSize}}]
print(x:size())

local convNet = nn.Sequential()
convNet:add(nn.SpatialConvolution(channels, 6, 6, 3))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
convNet:add(nn.SpatialConvolution(6, 12, 6, 2))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialConvolution(12, 24, 6, 1))
convNet:add(nn.ReLU())
convNet:add(nn.Reshape(24*7))
convNet:add(nn.Linear(24*7, classes))
convNet:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

--first256Samples = x[{{1,100},1}]
--print(first256Samples)
--local input = image.toDisplayTensor{input = x, nrow = 100/3/2}
--local image = image.display(input)


--BATCHES COMING NEXT!
local counter = 0
local batchSize = 32
local epochs = 3
local iterations = epochs * math.ceil(trainSize / batchSize)

local optimState = {
  learningRate = 5e-2
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

	local testLog = convNet:forward(testX)
	local testClassProbs = torch.exp(testLog)
	local _, testPredictions = torch.max(testClassProbs, 2)

	sum = 0
	(trainPredictions:long():squeeze() - trainY:long():squeeze()):apply(SumIfZero)
	trainError = 1 - sum/trainPredictions:size(1)
	
	sum = 0
	(testPredictions:long():squeeze() - testY:long():squeeze()):apply(SumIfZero)
	testError = 1 - sum/testPredictions:size(1)
	return trainError, testError
end

while true do
	for i = 1, iterations do
		local _, minibatchLoss = optimMethod(feval, parameters, optimState)
		--print(string.format("Minibatches Processed: %4d, loss = %6f", i,  minibatchLoss[1]))
	end

	trainError, testError = getError()
	print(string.format("Train error: %6f", trainError))
	print(string.format("Test error:  %6f", testError))

	maxLoops = maxLoops - 1
	if maxLoops == 0 or trainError + testError < 0.05 then
		break
	end
end






