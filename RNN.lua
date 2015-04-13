require "nn"
require "hdf5"
require "image"
require "optim"
require "nnx"

cmd = torch.CmdLine()
cmd:option("-gpu", false, "Use GPU")
cmd:option("-dataFile", "gameData1.hdf5", "Path to file")
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
local dataSize = x:size(2)
local states = x:size(3)

--for s = 1, states do
  --d = x[{{}, {}, s}]:max()
  --Scaling and mean substraction
  --x[{{}, {}, s}]:div(d)
--end

if WITH_CUDA then 
  x = x:cuda()
  y = y:cuda()
end

local classes = 4

local trainX = x[{1, {}}]
local trainY = y[{1, {}}]

for c = 2, x:size(1)-1 do
  trainX = torch.cat(trainX, x[{c, {}}], 1)
  trainY = torch.cat(trainY, y[{c, {}}], 1)
end

local trainSize = trainX:size(1)

local testSize = dataSize
testX = x[{x:size(1), {}}]
testY = y[{x:size(1), {}}]

local outCh = {48, 128, 192, 192, 128}

local rho = 5
local updateInterval = 5
local lr = 0.1

-- RNN
local rec = nn.Recurrent(
   states, nn.View(states), 
   nn.Linear(states, states), nn.Tanh(), 
   rho
)

convNet = nn.Sequential()
convNet:add(rec)
convNet:add(nn.Linear(states, classes))
convNet:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()


local counter = 0
local batchSize = 1
local epochs = 1
local gpuNum = 1
local iterations = epochs * math.ceil(trainSize / batchSize)

local optimState = {
  learningRate = 1
}
local optimMethod = optim.adagrad

if WITH_CUDA then 
  convNet:cuda()
  criterion:cuda()
  --gpuNum = 4
  --iterations = epochs * math.ceil(trainSize / (batchSize*gpuNum))
end

local parameters, gradParameters = convNet:getParameters()

local i = 0

local feval = function (params)
  if params ~= parameters then
    parameters:copy(params)
  end

  --for gpu = 1, gpuNum do
    --if WITH_CUDA then cutorch.setDevice(gpu) end
    local xBatch
    local yBatch = torch.Tensor({5})
    while yBatch[1] == 5 do
      local startIndex = counter * batchSize + 1
      local endIndex = math.min(trainSize, (counter + 1) * batchSize)

      if endIndex == trainSize then
        counter = 0
        rec:forget()
      else
        counter = counter + 1
      end

      xBatch = trainX[{{startIndex, endIndex}, {}}]
      yBatch = trainY[{{startIndex, endIndex}}]

      gradParameters:zero()
      --print(yBatch, xBatch, counter)
      if yBatch[1] == 5 then
        --rec:forget()
      end
    end

    local outputs = convNet:forward(xBatch)
    local loss = criterion:forward(outputs, yBatch)
    local dloss = criterion:backward(outputs, yBatch)
    convNet:backward(xBatch, dloss)

    i = i + 1
    -- note that updateInterval < rho
    if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      rec:updateParameters(lr)
      rec:forget()
    end
  --end
  --if WITH_CUDA then cutorch.synchronize() end

  return loss, gradParameters
end

local printErrorPerPercentile = function()

  local calcError = function(data, labels, p)
    local dataResults = convNet:forward(data)
    local _, maxResults = torch.max(dataResults, 2)

    local getPercentileScore = function(val)
      if val==0 then p[1] = p[1]+1 end
      p[2] = p[2] + 1
    end
    if labels[1] == 5 then
      --rec:forget()
      return
    end
    local results = maxResults:long() - labels:long()
    --print(maxResults[1][1])
    results:apply(getPercentileScore)
  end

  local printError = function(data, labels, intro)
    rec:forget()
    local p = torch.Tensor(2)
    for x = 1, data:size(1)/batchSize do
      local startIndex = (x-1) * batchSize + 1
      local endIndex = startIndex + batchSize - 1
      --print(startIndex, endIndex, data:size(), labels:size())
      calcError(data[{{startIndex, endIndex}, {}}], labels[{{startIndex, endIndex}}], p)
    end
    p[1] = (p[1]*100.0) / p[2]
    print(string.format("%s Score  %3i%%", intro, p[1]))
  end

  printError(trainX, trainY, "Training")
  printError(testX, testY, "Testing")

end


for i = 1, iterations do
  local _, minibatchLoss = optimMethod(feval, parameters, optimState)
  --print(string.format("Minibatches Processed: %4d, loss = %6f", i,  minibatchLoss[1]))
end
print(parameters[100])

printErrorPerPercentile()


print(string.format("Time elapsed: %5is", (os.time() - startTime)))






