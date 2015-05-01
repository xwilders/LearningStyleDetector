require "gnuplot"

local loss = torch.load("lossCurve.t7")
print(torch.Tensor(loss):size())

gnuplot.axis(auto)
gnuplot.plot({
  torch.range(1, #loss),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  torch.Tensor(loss),           -- y-coordinates (the training losses)
  '-'})

io.read()