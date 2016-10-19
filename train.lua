require 'nn'
local M = {}
local grid = dofile 'grid.lua'
local net = nn.Sequential()
local loss = nn.MSECriterion()
net:add(nn.Linear(64, 164))
net:add(nn.ReLU())
net:add(nn.Linear(164, 150))
net:add(nn.ReLU())
net:add(nn.Linear(150, 4))


function M.train(epochs, alpha, epsilon, gamma) 
      local state = grid.initGrid()
      local status = 1
      local action = 0
      local avgLoss = 0
      local totalMoves = 0
      for i = 1, epochs do
      	 state = grid.initGrid()
      	 status = 1
      	 while (status == 1) do
      	    qval = net:forward(state:view(-1)) -- qval has size 4
	    if torch.rand(1)[1] < epsilon then -- random action
	       action = torch.random(1, 4)
	    else
	       _, action = torch.max(qval, 1)
	       action = action[1]
	    end
	    local new_state = grid.makeMove(state, action)
	    local reward = grid.getReward(new_state)
	    local newQ = net:forward(new_state:view(-1))
	    local maxQ = torch.max(newQ) 
	    local y = qval:clone()
	    local update = reward
	    if reward == -1 then -- non-terminal
	       update = update + (gamma * maxQ)
	    end
	    y[action] = update -- target output for the taken action
	    -- backprop for this experience
	    local J = loss:forward(qval, y)
	    net:zeroGradParameters()
	    net:backward(state:view(-1), loss:backward(qval, y))
	    net:updateParameters(alpha)
	    state = new_state
	    avgLoss = avgLoss + J
	    totalMoves = totalMoves + 1
	    if reward ~= -1 then
	       status = 0
	    end
	 end
	 print(string.format("Average loss after game %s: %.4f", i,
			     avgLoss / totalMoves))
	 if epsilon > 0.1 then
	    epsilon = epsilon - (1 / epochs)
	 end
	 -- if epsilon > 0.1 then
	 --    epsilon = 1 / i
	 -- end
      end
end

-- train with experience replay
function M.trainER(epochs, alpha, epsilon, gamma, init) 
      local state = grid.initGrid()
      local status = 1
      local action = 0
      local avgLoss = 0
      local totalMoves = 0
      -- ER things
      local buffer = 80
      local replay = {}
      local h = 0
      for i = 1, epochs do
	 if init == 0 then
	    state = grid.initGrid()
	 elseif init == 1 then
	    state = grid.initGridPlayer()
	 elseif init == 2 then
	    state = grid.initGridRand()
	 end
      	 status = 1
      	 while (status == 1) do
      	    qval = net:forward(state:view(-1)) -- qval has size 4
	    if torch.rand(1)[1] < epsilon then -- random action
	       action = torch.random(1, 4)
	    else
	       _, action = torch.max(qval, 1)
	       action = action[1]
	    end
	    local new_state = grid.makeMove(state, action)
	    local reward = grid.getReward(new_state)
	    -- ER
	    if #replay < buffer then -- if buffer not full
	       table.insert(replay, {state, action, reward, new_state})
	    else -- buffer full, so overwrite
	       if (h < buffer) then
		  h = h + 1
	       else
		  h = 1
	       end
	       replay[h] = {state, action, reward, new_state}
	       local X_train = torch.zeros(#replay, 64)
	       local y_train = torch.zeros(#replay, 4)
	       for j = 1, #replay do
		  local old_state_rep, action_rep, reward_rep, new_state_rep = table.unpack(replay[j])
		  local old_qval = net:forward(old_state_rep:view(-1))
		  local newQ = net:forward(new_state_rep:view(-1))
		  local maxQ = torch.max(newQ)
		  local y = old_qval:clone()
		  local update = reward_rep
		  if reward_rep == -1 then
		     update = update + (gamma * maxQ)
		  end
		  y[action_rep] = update
		  X_train[j] = old_state_rep:view(-1)
		  y_train[j] = y
	       end
	       local h = net:forward(X_train)
	       local J = loss:forward(h, y_train)
	       net:zeroGradParameters()
	       net:backward(X_train, loss:backward(h, y_train))
	       net:updateParameters(alpha)
	       avgLoss = avgLoss + J
	    end   
	    state = new_state
	    totalMoves = totalMoves + 1
	    if reward ~= -1 then
	       status = 0
	    end
	 end
	 print(string.format("Average loss after game %s: %.4f", i,
			     avgLoss / totalMoves))
	 if epsilon > 0.1 then
	    epsilon = epsilon - (1 / epochs)
	 end
      end
end

function M.play(init, games)
   local state = nil
   local numWin = 0
   local numTime = 0
   local numLoss = 0
   for k = 1, games do
      if init == 0 then
	 state = grid.initGrid()
      elseif init == 1 then
	 state = grid.initGridPlayer()
      elseif init == 2 then
	 state = grid.initGridRand()
      end
      print("Initial State:")
      grid.dispGrid(state)
      local status = 1
      local moves = 0
      while (status == 1) do
	 local qval = net:forward(state:view(-1))
	 print(qval)
	 local _, action = torch.max(qval, 1)
	 action = action[1]
	 state = grid.makeMove(state, action)
	 grid.dispGrid(state)
	 local reward = grid.getReward(state)
	 if reward ~= -1 then
	    if reward == 10 then
	       numWin = numWin + 1
	    elseif reward == -10 then
	       numLoss = numLoss + 1
	    end
	    status = 0
	    print(string.format("Received terminal reward %d", reward))
	 end
	 moves = moves + 1
	 if (moves > 10) then
	    print("Game lost; too many moves")
	    numTime = numTime + 1
	    break
	 end
      end
   end
   print(string.format("Total Wins: %d\nTotal Loss: %d\nTotal Time: %d", numWin, numLoss, numTime))
end

M.net = net
M.grid = grid
return M
