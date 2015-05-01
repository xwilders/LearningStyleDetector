require 'unsup'

torch.setdefaulttensortype('torch.FloatTensor')

local table = torch.load("allParams.t7")
local allParams = table["params"]
--print(allParams)
local players = table["players"]

local classifiers = torch.Tensor(players:size(1))

local centroids, counts = unsup.kmeans(allParams, 2, 1000)
for i = 1, players:size(1) do
	classifiers[i] = torch.norm(allParams[i] - centroids[1]) / torch.norm((centroids[2] - centroids[1]))
end

classifiers = torch.clamp(classifiers, 0, 1)
local res1 = classifiers - players
local res2 = classifiers - (torch.Tensor(players:size(1)):fill(1) - players)


local printErrorPerPercentile = function(data)
	

  local calcError = function(data, p)

    local getPercentileScore = function(val)
      if val<=0.05 then p[1] = p[1] + 1 end
      if val<=0.15 then p[2] = p[2] + 1 end
      if val<=0.30 then p[3] = p[3] + 1 end
      if val<=0.50 then p[4] = p[4] + 1 end
      p[5] = p[5] + 1
    end
    
    data:apply(getPercentileScore)
  end

  local p = torch.Tensor(5):zero()
	calcError(data:abs(), p)

	for i = 1, p:size(1) do
    p[i] = (p[i]*100.0) / (data:size(1))
  end
  print(string.format("Error  0.05:%2.1f%%, 0.15:%2.1f%%, 0.30:%2.1f%%, 0.50:%2.1f%%, 1.00:%2.1f%%", p[1], p[2], p[3], p[4], p[5]))

end

print(allParams:size())
printErrorPerPercentile(res1)
printErrorPerPercentile(res2)

