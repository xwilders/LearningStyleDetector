require "nn"
require "hdf5"
require "image"
require "optim"
require "nnx"

torch.include('rnn', 'SequencerCriterion.lua')

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
local x = f:read('/data/x'):all():float() -- ALL?
local y = f:read('/data/y'):all():float() -- ALL ?
local players = f:read('/data/players'):all():float()
f:close()

if WITH_CUDA then 
  x = x:cuda()
  y = y:cuda()
end

local totalPlayers = 1--players:size(1))

local allParams = nil
local originalParameters = nil

local dataSize = x:size(1)/players:size(1)
local states = x:size(3)
local classes = 5

local trainNum = dataSize-1
local testNum = 1
local trainSize = trainNum * x:size(2)
local testSize = testNum * x:size(2)

for s = 1, states do
  t = x[{{}, {}, {s}}]:max()
  --Scaling and mean substraction
  x[{{}, {}, {s}}] = x[{{}, {}, {s}}]/t
end
--print(x[1])

local rho = 5
local updateInterval = 5
local lr = 0.1--1--0.000000000000000000000001

-- RNN
local rec = nn.Recurrent(
   states, nn.Identity(), 
   nn.Linear(states, states), nn.ReLU(), 
   rho)

local convNet = nn.Sequential()
--convNet:add(nn.Linear(states, states))
--convNet:add(nn.ReLU())
convNet:add(rec)
--convNet:add(nn.Tanh())
convNet:add(nn.Linear(states, states))
convNet:add(nn.ReLU())
--local lin = nn.Linear(states, classes)
convNet:add(nn.Linear(states, classes))
convNet:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()
--local sequencerCriterion = nn.SequencerCriterion(criterion)


local counter = 0
local batchSize = 300
local epochs = 20
local gpuNum = 1
local iterations = epochs * math.ceil(trainSize / batchSize)

local optimState = {
  learningRate = 5e-2
}
local optimMethod = optim.adagrad

if WITH_CUDA then 
  convNet:cuda()
  criterion:cuda()
  --gpuNum = 4
  --iterations = epochs * math.ceil(trainSize / (batchSize*gpuNum))
end

local printErrorPerPercentile = function(player, startTime, trainX, trainY, testX, testY)

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
    --print(maxResults)
    results:apply(getPercentileScore)
  end

  local printError = function(data, labels, intro)
    --rec:forget()
    --local p = torch.Tensor(2)
    --for x = 1, data:size(1)/batchSize do
    --  local startIndex = (x-1) * batchSize + 1
    --  local endIndex = startIndex + batchSize - 1
      --print(startIndex, endIndex, data:size(), labels:size())
    --  calcError(data[{{startIndex, endIndex}, {}}], labels[{{startIndex, endIndex}}], p)
    --end
    --p[1] = (p[1]*100.0) / p[2]
    --print(string.format("%s Score  %3i%%", intro, p[1]))

    stats = torch.Tensor(5):zero()
    local getStats = function(state)
      stats[state] = stats[state] + 1
    end

    labels:apply(getStats)
    oldStats = stats:clone()
    stats:zero()
    local dataResults = convNet:forward(data)
    local _, maxResults = torch.max(dataResults, 2)
    print(dataResults[1])
    maxResults:apply(getStats)
    print(oldStats, stats)
    


  end

  if player%100 == 0 then
    print(player, string.format("Time: %4i%%", os.time() - startTime))
    if player==0 then
      printError(trainX[{{1,trainSize/trainNum},{}}], trainY[{{1,trainSize/trainNum}}], "Training")
      printError(testX, testY, "Testing")
    end
  end

end

for player = 0, totalPlayers-1 do--players:size(1)-1 do
  local startTimePlayer = os.time()

  local trainX = x[{1 + player*dataSize, {}}]
  local trainY = y[{1 + player*dataSize, {}}]

  for c = 2, dataSize-1 do
    trainX = torch.cat(x[{c + player*dataSize, {}}], trainX, 1)
    trainY = torch.cat(y[{c + player*dataSize, {}}], trainY, 1)
  end
  
  local testX = x[{(player+1)*dataSize, {}}]
  local testY = y[{(player+1)*dataSize, {}}]

  
  local parameters, gradParameters = convNet:getParameters()
  if allParams == nil then
    --local pL, gpL = lin:getParameters()
    allParams = torch.Tensor(totalPlayers, parameters:size(1))
    originalParameters = parameters
    --print(pL:size(1))
  else
    parameters:clone(originalParameters)
  end

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
          --rec:forget()
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

      --print(xBatch)
      --print(yBatch)

      local outputs = convNet:forward(xBatch)--convNet:forward(xBatch)
      local loss = criterion:forward(outputs, yBatch)
      local dloss = criterion:backward(outputs, yBatch)
      convNet:backward(xBatch, dloss)--convNet:backward(xBatch, dloss)

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


  for i = 1, iterations do
    rec:forget()
    local _, minibatchLoss = optimMethod(feval, parameters, optimState)
    --print(string.format("Minibatches Processed: %4d, loss = %6f", i,  minibatchLoss[1]))
  end

  
  printErrorPerPercentile(player, startTime, trainX, trainY, testX, testY)

  --p, gp = lin:getParameters()
  allParams[player+1] = parameters:clone()
  --print(allParams)
  
end


print(string.format("Time elapsed: %5is", (os.time() - startTime)))
--print(allParams)
torch.save("allParams.t7", {["params"]=allParams, ["players"]=players[{{1, totalPlayers}}]})




