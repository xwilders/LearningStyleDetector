------------------------------------------------------------------------
--[[ SequencerCriterion ]]--
-- Applies a criterion to each of the inputs and targets in the 
-- corresponding input and target Tables.
-- Useful for nn.Repeater and nn.Sequencer.
------------------------------------------------------------------------
local SequencerCriterion, parent = torch.class('nn.SequencerCriterion', 'nn.Criterion')

function SequencerCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   self.gradInput = {}
end

function SequencerCriterion:forward(inputTable, targetTable)
   self.output = 0
   for i,input in ipairs(inputTable) do
      self.output = self.output + self.criterion:forward(input, targetTable[i])
   end
   return self.output
end

function SequencerCriterion:backward(inputTable, targetTable)
   for i,input in ipairs(inputTable) do
      self.gradInput[i] = recursiveCopy(self.gradInput[i], self.criterion:backward(input, targetTable[i]))
   end
   return self.gradInput
end

function SequencerCriterion:type(type)
   self.gradInput = recursiveType(self.gradInput)
   return self.criterion:type(type)
end


function recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2):copy(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for i = 1, #param do
         param[i] = recursiveType(param[i], type_str)
      end
   else
      if torch.typename(param) and 
        torch.typename(param):find('torch%..+Tensor') then
         param = param:type(type_str)
      end
   end
   return param
end