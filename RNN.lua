require "nn"
require "hdf5"
require "image"
require "optim"
require "nnx"

torch.include('rnn', 'SequencerCriterion.lua')

cmd = torch.CmdLine()
cmd:option("-gpu", false, "Use GPU")
cmd:option("-dataFile", "gameData1*100000.hdf5", "Path to file")
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
local players = f:read('/data/players'):all():float()
f:close()

if WITH_CUDA then 
  x = x:cuda()
  y = y:cuda()
end

local allParams = nil
local originalParameters = nil

local dataSize = x:size(1)/players:size(1)
local states = x:size(3)
local classes = 4

local trainNum = dataSize-1
local testNum = 1
local trainSize = trainNum * x:size(2)
local testSize = testNum * x:size(2)

local rho = 5
local updateInterval = 5
local lr = 0.1

-- RNN

local lin = nn.Linear(states, classes)
local rec = nn.Recurrent(
   states, nn.Identity(), 
   nn.Linear(states, states), nn.Tanh(), 
   rho
)
local convNet = nn.Sequencer(rec)

--local convNet = nn.Sequential()
--convNet:add(nn.Sequencer(rec))
--convNet:add(rec)
--local lin = nn.Linear(states, classes)
--convNet:add(lin)
convNet:add(nn.LogSoftMax())


local criterion = nn.ClassNLLCriterion()
local sequencerCriterion = nn.SequencerCriterion(criterion)


local counter = 0
local batchSize = 5
local epochs = 1
local gpuNum = 1
local iterations = epochs * math.ceil(trainSize / batchSize)

local optimState = {
  learningRate = 1e-1
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
    data = {[1]=data[1], [2]=data[2], [3]=data[3], [4]=data[4], [5]=data[5]}
    local dataResultsTable = convNet:forward(data)
    local dataResults = torch.Tensor(batchSize, states)
    for c = 1, batchSize do
      dataResults[c] = dataResultsTable[c]
    end
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
    print(string.format("%s Score  %3i%%", intro, p[1]), os.time() - startTime)
  end

  print(player+1, "Active %", players[player+1])
  if player==0 then
    printError(trainX[{{1,trainSize/trainNum},{}}], trainY[{{1,trainSize/trainNum}}], "Training")
    printError(testX, testY, "Testing")
  end

end

for player = 0, players:size(1)-1 do
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
    allParams = torch.Tensor(players:size(1), 44)--pL:size(1))
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
          rec:forget()
        else
          counter = counter + 1
        end

        xBatch = trainX[{{startIndex, endIndex}, {}}]
        yBatch = trainY[{{startIndex, endIndex}}]
        xBatch = {[1]=xBatch[1], [2]=xBatch[2], [3]=xBatch[3], [4]=xBatch[4], [5]=xBatch[5]}
        yBatch = {[1]=yBatch[1], [2]=yBatch[2], [3]=yBatch[3], [4]=yBatch[4], [5]=yBatch[5]}

        gradParameters:zero()
        --print(yBatch, xBatch, counter)
        if yBatch[1] == 5 then
          --rec:forget()
        end
      end

      local outputs = convNet:forward(xBatch)--convNet:forward(xBatch)
      local loss = sequencerCriterion:forward(outputs, yBatch)
      local dloss = sequencerCriterion:backward(outputs, yBatch)
      convNet:backward(xBatch, dloss)--convNet:backward(xBatch, dloss)

      i = i + 1
      -- note that updateInterval < rho
      if i % updateInterval == -1 then
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
    local _, minibatchLoss = optimMethod(feval, parameters, optimState)
    print(string.format("Minibatches Processed: %4d, loss = %6f", i,  minibatchLoss[1]))
  end

  
  printErrorPerPercentile(player, startTimePlayer, trainX, trainY, testX, testY)

  p, gp = lin:getParameters()
  allParams[player+1] = p:clone()
  
end


print(string.format("Time elapsed: %5is", (os.time() - startTime)))
torch.save("allParams.t7", allParams)




