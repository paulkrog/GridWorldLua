-- The game state is represented as a 4x4x4 torch tensor.

IO = require 'io'
-- torch.setdefaulttensortype('torch.ByteTensor')

local M = {}

-- Return a coordinate table on [s, e] 
function M.randPair(s, e) 
   return {torch.random(s, e), torch.random(s, e)}
end

function M.findLoc(state, obj)
   for i = 1, 4 do
      for j = 1, 4 do
	 if (torch.eq(state[{{}, {i}, {j}}], obj):sum() == 4) then
	    return {i, j}
	 end
      end
   end
end

function M.initGrid()
   local state = torch.zeros(4, 4, 4)
   -- place player
   state[{{}, {1}, {2}}] = torch.Tensor({0, 0, 0, 1})
   -- place wall
   state[{{}, {3}, {3}}] = torch.Tensor({0, 0, 1, 0})
   -- place pit
   state[{{}, {2}, {2}}] = torch.Tensor({0, 1, 0, 0})
   -- place goal
   state[{{}, {4}, {4}}] = torch.Tensor({1, 0, 0, 0})
   
   return state
end

-- place player with random coordinates

function M.initGridPlayer()
    local state = torch.zeros(4, 4, 4)
    local player_x = torch.random(1, 4)
    local player_y = torch.random(1, 4)
   state[{{}, {player_x}, {player_y}}] = torch.Tensor({0, 0, 0, 1})
   -- place wall
   state[{{}, {3}, {3}}] = torch.Tensor({0, 0, 1, 0})
   -- place pit
   state[{{}, {2}, {2}}] = torch.Tensor({0, 1, 0, 0})
   -- place goal
   state[{{}, {4}, {4}}] = torch.Tensor({1, 0, 0, 0})

   -- find grid position of player
   local a = M.findLoc(state, torch.Tensor({0, 0, 0, 1}))
   local w = M.findLoc(state, torch.Tensor({0, 0, 1, 0}))  -- find wall
   local g = M.findLoc(state, torch.Tensor({1, 0, 0, 0}))  -- find goal
   local p = M.findLoc(state, torch.Tensor({0, 1, 0, 0}))  -- find pit
   if (not a or not w or not g or not p) then
      print("Invalid grid. Rebuilding..")
      return M.initGridPlayer()
   end
   return state
end

-- Initialize grid so that goal, pit, wall, player are all randomly placed

function M.initGridRand()
    local state = torch.zeros(4, 4, 4)
    local player_x = torch.random(1, 4)
    local player_y = torch.random(1, 4)
    local wall_x = torch.random(1, 4)
    local wall_y = torch.random(1, 4)
    local pit_x = torch.random(1, 4)
    local pit_y = torch.random(1, 4)
    local goal_x = torch.random(1, 4)
    local goal_y = torch.random(1, 4)
   state[{{}, {player_x}, {player_y}}] = torch.Tensor({0, 0, 0, 1})
   -- place wall
   state[{{}, {wall_x}, {wall_y}}] = torch.Tensor({0, 0, 1, 0})
   -- place pit
   state[{{}, {pit_x}, {pit_y}}] = torch.Tensor({0, 1, 0, 0})
   -- place goal
   state[{{}, {goal_x}, {goal_y}}] = torch.Tensor({1, 0, 0, 0})

   -- find grid position of player
   local a = M.findLoc(state, torch.Tensor({0, 0, 0, 1}))
   local w = M.findLoc(state, torch.Tensor({0, 0, 1, 0}))  -- find wall
   local g = M.findLoc(state, torch.Tensor({1, 0, 0, 0}))  -- find goal
   local p = M.findLoc(state, torch.Tensor({0, 1, 0, 0}))  -- find pit
   if (not a or not w or not g or not p) then
      print("Invalid grid. Rebuilding..")
      return M.initGridRand()
   end
   return state
end
    
function M.makeMove(state, action)
   -- need to locate player in grid
   -- need to determine what object (if any) is in the new grid spot
   -- the player is moving to
   local player_loc = M.findLoc(state, torch.Tensor({0, 0, 0, 1}))
   local wall = M.findLoc(state, torch.Tensor({0, 0, 1, 0}))
   local goal = M.findLoc(state, torch.Tensor({1, 0, 0, 0}))
   local pit = M.findLoc(state, torch.Tensor({0, 1, 0, 0}))
   state = torch.zeros(4, 4, 4)

   -- up (row - 1)
   if action == 1 then
      local new_loc = {player_loc[1] - 1, player_loc[2]}
      if not (new_loc[1] == wall[1] and new_loc[2] == wall[2]) then
	 if new_loc[1] <= 4 and new_loc[2] <= 4 and new_loc[1] >= 1
	 and new_loc[2] >= 1 then
	    state[4][new_loc[1]][new_loc[2]] = 1
	 end
      end
   -- down (row + 1)
   elseif action == 2 then
      local new_loc = {player_loc[1] + 1, player_loc[2]}
      if not (new_loc[1] == wall[1] and new_loc[2] == wall[2]) then
	 if new_loc[1] <= 4 and new_loc[2] <= 4 and new_loc[1] >= 1
	 and new_loc[2] >= 1 then
	    state[4][new_loc[1]][new_loc[2]] = 1
	 end
      end
   -- left (column - 1)
   elseif action == 3 then
      local new_loc = {player_loc[1], player_loc[2] - 1}
      if not (new_loc[1] == wall[1] and new_loc[2] == wall[2]) then
	 if new_loc[1] <= 4 and new_loc[2] <= 4 and new_loc[1] >= 1
	 and new_loc[2] >= 1 then
	    state[4][new_loc[1]][new_loc[2]] = 1
	 end
      end
   -- right (column + 1)
   elseif action == 4 then
      local new_loc = {player_loc[1], player_loc[2] + 1}
      if not (new_loc[1] == wall[1] and new_loc[2] == wall[2]) then
	 if new_loc[1] <= 4 and new_loc[2] <= 4 and new_loc[1] >= 1
	 and new_loc[2] >= 1 then
	    state[4][new_loc[1]][new_loc[2]] = 1
	 end
      end
   end

   local new_player_loc = M.findLoc(state, torch.Tensor({0, 0, 0, 1}))
   if new_player_loc == nil then
      state[{{}, {player_loc[1]}, {player_loc[2]}}] = torch.Tensor({0,
								    0,
								    0, 1})
   end
   -- re-place pit
   state[2][pit[1]][pit[2]] = 1
   -- re-place wall
   state[3][wall[1]][wall[2]] = 1
   -- re-place goal
   state[1][goal[1]][goal[2]] = 1
   
   return state
end

function M.getLoc(state, level)
   for i = 1, 4 do
      for j = 1, 4 do
	 if (state[level][i][j] == 1) then
	    return {i, j}
	 end
      end
   end
end

function M.getReward(state)
   local player_loc = M.getLoc(state, 4)
   local pit = M.getLoc(state, 2)
   local goal = M.getLoc(state, 1)
   if (player_loc[1] == pit[1] and player_loc[2] == pit[2]) then
      return -10
   elseif (player_loc[1] == goal[1] and player_loc[2] == goal[2]) then
      return 10
   else
      return -1
   end
end

function M.dispGrid(state)
   local grid = torch.Tensor(4, 4)
   local player_loc = M.findLoc(state, torch.Tensor({0, 0, 0, 1}))
   local wall = M.findLoc(state, torch.Tensor({0, 0, 1, 0}))
   local goal = M.findLoc(state, torch.Tensor({1, 0, 0, 0}))
   local pit = M.findLoc(state, torch.Tensor({0, 1, 0, 0}))
   for i = 1, 4 do
      for j = 1, 4 do
	 grid[i][j] = 0
      end
   end

   if player_loc ~= nil then
      grid[player_loc[1]][player_loc[2]] = 1 -- player
   end
   if wall ~= nil then
      grid[wall[1]][wall[2]] = 8 -- wall
   end
   if goal ~= nil then
      grid[goal[1]][goal[2]] = 3 -- goal
   end
   if pit ~= nil then
      grid[pit[1]][pit[2]] = 7 -- pit
   end

   print(grid)
end

return M


