require "nn"
require "hdf5"
require "optim"
require "nnx"

cmd = torch.CmdLine()
cmd:option("-gpu", false, "Use GPU")
cmd:option("-dataFile", "gameData.hdf5", "Path to file")
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

--[[local y = torch.Tensor(yRaw:size(1), classes)
local elem = 1
local expand = function(label)
  local vector = torch.zeros(classes)
  vector[label+1] = 1
  y[elem] = vector
  elem = elem + 1
end
yRaw:apply(expand)]]--

--x = x[{{},{1}}]
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

outCh = 2

batchSize = 8
rho = 5
hiddenSize = 10
nIndex = 10000
-- RNN
r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, nIndex))
rnn:add(nn.LogSoftMax())



reshape = outCh*8--150*6*channels
convNet:add(nn.Reshape(reshape))
convNet:add(nn.Linear(reshape, reshape))
convNet:add(nn.ReLU())
convNet:add(nn.Dropout())

convNet:add(nn.Linear(reshape, classes))

--convNet:add(nn.Tanh())
--convNet:add(nn.LogSoftMax())

local criterion = nn.MSECriterion()

if WITH_CUDA then 
  convNet:cuda()
  criterion:cuda()
end


local counter = 0
local batchSize = 256
local epochs = 3
local iterations = epochs * math.ceil(trainSize / batchSize)

local optimState = {
  learningRate = 1e-1

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

local printErrorPerPercentile = function()

  local printError = function(data, labels, intro)
    local dataResults = convNet:forward(data)
    
    local p = torch.Tensor(7):zero()

    local getPercentileScore = function(val)
      if val<=0.01 then p[1] = p[1] + 1 end
      if val<=0.02 then p[2] = p[2] + 1 end
      if val<=0.03 then p[3] = p[3] + 1 end
      if val<=0.05 then p[4] = p[4] + 1 end
      if val<=0.10 then p[5] = p[5] + 1 end
      if val<=0.25 then p[6] = p[6] + 1 end
      p[7] = p[7] + 1
    end
    
    local results = (dataResults - labels)[{{}, 1}]:abs()
    results:apply(getPercentileScore)
    for i = 1, 7 do
      p[i] = p[i] / labels:size(1)
      p[i] = math.floor(p[i]*100)
    end
    print(string.format("%s Error  0.01:%2i%%, 0.02:%2i%%, 0.03:%2i%%, 0.05:%2i%%, 0.10:%2i%%, 0.25:%2i%%", intro, p[1], p[2], p[3], p[4], p[5], p[6]))
  end

  printError(trainX, trainY, "Training")
  printError(testX, testY, "Testing")
end

local printError = function()
  local trainLoss = criterion:forward(convNet:forward(trainX), trainY)
  local testLoss = criterion:forward(convNet:forward(testX), testY)
  print(string.format("Train MSE loss: %6f", trainLoss))
  print(string.format("Test MSE loss: %6f", testLoss))
end

local maxLoops = 1

while true do
  for i = 1, iterations do
    local _, minibatchLoss = optimMethod(feval, parameters, optimState)
    print(string.format("Minibatches Processed: %4d, loss = %6f", i,  minibatchLoss[1]))
  end

  
  printError()
  printErrorPerPercentile()

  --local trainError, testError = getError()
  --print(string.format("Train classification error: %6f", trainError))
  --print(string.format("Test classification error:  %6f", testError))

  maxLoops = maxLoops - 1
  if maxLoops == 0 or trainError + testError < 0.05 then
    break
  end
end

print(string.format("Time elapsed: %5is", (os.time() - startTime)))






