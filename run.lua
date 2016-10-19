local M = dofile 'train.lua'

M.trainER(10000, 0.01, 1, 0.9, 2)
M.play(2, 20)
