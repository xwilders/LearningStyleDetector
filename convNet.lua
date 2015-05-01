require "nn"
require "hdf5"
require "optim"

torch.manualSeed(1)

cmd = torch.CmdLine()
cmd:option("-gpu", false, "Use GPU")
cmd:option("-dataFile", "gameData0.hdf5", "Path to file")
cmd:text()
opt = cmd:parse(arg or {})

WITH_CUDA = opt.gpu
torch.setdefaulttensortype('torch.FloatTensor')

if WITH_CUDA then
  require 'cunn'
end


local startTime = os.time()

local f = hdf5.open(opt.dataFile, 'r')
local x = f:read('/data/x'):all():float()
local y = f:read('/data/y'):all():float()
f:close()

--x = x[{{},{1}}]
--print(x:size())
local width = x:size(4)
local height = x:size(3)
local channels = x:size(2)
local dataSize = x:size(1)

for c = 1, channels do
  local d = x[{{}, {c}, {}}]
  --Scaling and mean substraction
  local mean = d:mean()
  x[{{}, {c}, {}}] = d - mean
end


if WITH_CUDA then 
  x = x:cuda()
  y = y:cuda()
end

local classes = y:size(2)


local trainSize = math.ceil(dataSize*0.9)
trainX = x[{{1, trainSize}, {}}]
trainY = y[{{1, trainSize}}]


local testSize = dataSize - trainSize
testX = x[{{trainSize, dataSize}, {}}]
testY = y[{{trainSize, dataSize}}]

--local outCh = {96, 256, 384, 384, 256}
local outCh = {96/2, 256/2, 384/2, 384/2, 256/2}

local convNet = nn.Sequential()

convNet:add(nn.SpatialZeroPadding(6, 5, 3, 2))
convNet:add(nn.SpatialConvolutionMM(channels, outCh[1], 11, 6, 4)) --> 76 x 6
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(3, 1, 2, 1)) --> 38 x 6

convNet:add(nn.SpatialZeroPadding(3, 2, 3, 2))
convNet:add(nn.SpatialConvolutionMM(outCh[1], outCh[2], 5, 6))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(2, 1, 2, 1)) --> 19 x 6

convNet:add(nn.SpatialZeroPadding(1, 1, 3, 2))
convNet:add(nn.SpatialConvolutionMM(outCh[2], outCh[3], 3, 6))
convNet:add(nn.ReLU())

convNet:add(nn.SpatialZeroPadding(1, 1, 3, 2))
convNet:add(nn.SpatialConvolutionMM(outCh[3], outCh[4], 3, 6))
convNet:add(nn.ReLU())

convNet:add(nn.SpatialZeroPadding(1, 1, 3, 2))
convNet:add(nn.SpatialConvolutionMM(outCh[4], outCh[5], 3, 6))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialMaxPooling(2, 6, 2, 1)) --> 9 x 1

reshape =  outCh[5]*9--300*6*channels
convNet:add(nn.View(reshape))
convNet:add(nn.Dropout(0.5))
convNet:add(nn.Linear(reshape, 4096))
convNet:add(nn.ReLU())
convNet:add(nn.Dropout(0.5))
convNet:add(nn.Linear(4096, 4096))
convNet:add(nn.ReLU())

convNet:add(nn.Linear(4096, classes))

local criterion = nn.MSECriterion()


local counter = 0
local batchSize = 128
local epochs = 4
local gpuNum = 1
local iterations = epochs * math.ceil(trainSize / batchSize)

local optimState = {
  learningRate = 1e-5
}
local optimMethod = optim.adagrad

if WITH_CUDA then 
  convNet:cuda()
  criterion:cuda()
  --gpuNum = 4
  --iterations = epochs * math.ceil(trainSize / (batchSize*gpuNum))
end

local parameters, gradParameters = convNet:getParameters()

local feval = function (params)
  if params ~= parameters then
    parameters:copy(params)
  end

  --for gpu = 1, gpuNum do
    --if WITH_CUDA then cutorch.setDevice(gpu) end
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
  --end
  --if WITH_CUDA then cutorch.synchronize() end

  return loss, gradParameters
end

local printErrorPerPercentile = function()

  local calcError = function(data, labels, p)
    local dataResults = convNet:forward(data)
    local loss = criterion:forward(dataResults, labels)

    local getPercentileScore = function(val)
      if val<=0.01 then p[1] = p[1] + 1
      elseif val<=0.02 then p[2] = p[2] + 1
      elseif val<=0.03 then p[3] = p[3] + 1
      elseif val<=0.05 then p[4] = p[4] + 1
      elseif val<=0.10 then p[5] = p[5] + 1
      elseif val<=0.25 then p[6] = p[6] + 1
      else p[7] = p[7] + 1 end
    end
    
    local results = (dataResults - labels)[{{}, 1}]:abs()
    results:apply(getPercentileScore)

    return loss
  end

  local printError = function(data, labels, intro)
    local p = torch.Tensor(7):zero()
    local max = math.max(data:size(1)/100/5, 100)
    local loss = 0
    local lossNum = 0
    for x = 1, data:size(1)/max do
      local startIndex = (x-1) * max + 1
      local endIndex = startIndex + max - 1
      local batchLoss = calcError(data[{{startIndex, endIndex}, {}}], labels[{{startIndex, endIndex}}], p)
      loss = (loss*lossNum + batchLoss)/(lossNum+1)
      lossNum = lossNum + 1
    end
    for i = 1, 7 do
      p[i] = (p[i]*100.0) / (labels:size(1))
    end
    print(string.format("%s Error  0.01:%2.1f%%, 0.02:%2.1f%%, 0.03:%2.1f%%, 0.05:%2.1f%%, 0.10:%2.1f%%, 0.25:%2.1f%%, 1.00:%2.1f%%", intro, p[1], p[2], p[3], p[4], p[5], p[6], p[7]))
    print("Loss for", intro, loss)
  end

  printError(trainX, trainY, "Training")
  printError(testX, testY, "Testing")

end


local losses = {}

for i = 1, iterations do
  local _, minibatchLoss = optimMethod(feval, parameters, optimState)
  losses[#losses + 1] = minibatchLoss[1]
  print(string.format("Minibatches Processed: %4d, loss = %6f", i,  minibatchLoss[1]))
end


printErrorPerPercentile()

torch.save("lossCurve.t7", losses)

print(string.format("Time elapsed: %5is", (os.time() - startTime)))






